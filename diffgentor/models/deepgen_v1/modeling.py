# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen model - Qwen2.5-VL + SD3.5 for unified image generation and editing.

This module provides the main DeepGen model class that combines:
- Qwen2.5-VL as the language/vision encoder
- SD3.5 Transformer as the diffusion backbone
- A connector module to bridge LLM hidden states to DiT

The model supports both text-to-image generation and image editing tasks.
"""

import math
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig
from safetensors.torch import load_file as load_safetensors

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from .connector import ConnectorConfig, ConnectorEncoder
from .transformer import SD3Transformer2DModel
from .pipeline import StableDiffusion3Pipeline, calculate_shift


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def multi_apply(func, *args, **kwargs):
    """Apply function to multiple arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    """Find linear layer names for LoRA."""
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def load_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint from file.

    Supports both .pt/.pth and .safetensors formats.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        State dict
    """
    if checkpoint_path.endswith(".safetensors"):
        return load_safetensors(checkpoint_path)
    else:
        return torch.load(checkpoint_path, map_location="cpu")


def load_config(config_name: str = "deepgen_v1") -> Dict[str, Any]:
    """Load configuration by name from configs/ folder.

    Args:
        config_name: Name of the config file (without .py extension)

    Returns:
        Dict containing prompt_template, connector_config, and model_config

    Raises:
        ValueError: If config file not found
    """
    import importlib

    try:
        config_module = importlib.import_module(f".configs.{config_name}", package="diffgentor.models.deepgen_v1")
    except ModuleNotFoundError:
        raise ValueError(
            f"Config '{config_name}' not found. "
            f"Available configs are in diffgentor/models/deepgen_v1/configs/"
        )

    return {
        "prompt_template": getattr(config_module, "prompt_template"),
        "connector_config": getattr(config_module, "connector_config"),
        "model_config": getattr(config_module, "model_config"),
    }


class DeepGenModel(nn.Module):
    """DeepGen model combining Qwen2.5-VL and SD3.5 for image generation/editing.

    This model uses Qwen2.5-VL as the language/vision encoder and SD3.5 as the
    diffusion backbone. A connector module bridges the LLM hidden states to the
    DiT model.

    The model supports:
    - Text-to-image generation (T2I)
    - Image editing with text instructions (I2I)

    Args:
        diffusion_path: Path to diffusion model (transformer, vae, scheduler)
        qwen_path: Path to Qwen2.5-VL model
        config_name: Name of config to load from configs/ folder (default: deepgen_v1)
        lora_modules: LoRA target modules (default: 'auto')
        lora_rank: LoRA rank (default: 64)
        lora_alpha: LoRA alpha (default: 128)
        unconditional: Unconditional dropout rate (default: 0.1)
        weighting_scheme: Loss weighting scheme (default: 'none')
        logit_mean: Logit mean for timestep sampling (default: 0.0)
        logit_std: Logit std for timestep sampling (default: 1.0)
        device: Device to load model on
        torch_dtype: Data type for model weights
    """

    def __init__(
        self,
        diffusion_path: str,
        qwen_path: str,
        config_name: str = "deepgen_v1",
        lora_modules: Optional[Union[str, List[str]]] = 'auto',
        lora_rank: int = 64,
        lora_alpha: int = 128,
        unconditional: float = 0.1,
        weighting_scheme: str = 'none',
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_impl: str = "flash_attention_2",
    ):
        super().__init__()

        self._device = device or "cuda"
        self._dtype = torch_dtype

        # Load config from configs/ folder
        config = load_config(config_name)
        self.prompt_template = config["prompt_template"]
        connector_config = config["connector_config"]
        model_config = config["model_config"]

        # Extract model config values
        num_queries = model_config["num_queries"]
        vit_input_size = model_config["vit_input_size"]
        max_length = model_config["max_length"]
        freeze_lmm = model_config["freeze_lmm"]
        freeze_transformer = model_config["freeze_transformer"]

        # Load Qwen2.5-VL
        print(f"Loading Qwen2.5-VL from: {qwen_path}", flush=True)
        print(f"  device_map: {'auto' if device is None else device}", flush=True)
        print(f"  torch_dtype: {torch_dtype}", flush=True)
        print(f"  attn_implementation: {attn_impl}", flush=True)

        # Determine device_map strategy
        if device is None:
            # Check number of available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # Use auto device_map for multi-GPU
                device_map = "auto"
            else:
                # Single GPU - load directly to cuda:0
                device_map = {"": 0}
            print(f"  Using device_map: {device_map} (num_gpus={num_gpus})", flush=True)
        else:
            device_map = device

        self.lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        print(f"  Qwen2.5-VL loaded successfully", flush=True)
        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        # Load tokenizer
        print(f"Loading tokenizer from: {qwen_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            qwen_path,
            trust_remote_code=True,
            padding_side='right',
        )
        print(f"  Tokenizer loaded successfully", flush=True)

        # Load diffusion Transformer
        print(f"Loading diffusion Transformer from: {diffusion_path}", flush=True)
        self.transformer = SD3Transformer2DModel.from_pretrained(
            diffusion_path,
            subfolder="transformer",
            torch_dtype=torch_dtype,
        )
        print(f"  Transformer loaded successfully", flush=True)
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer

        # Load VAE
        print(f"Loading VAE from: {diffusion_path}", flush=True)
        self.vae = AutoencoderKL.from_pretrained(
            diffusion_path,
            subfolder="vae",
            torch_dtype=torch_dtype,
        )
        print(f"  VAE loaded successfully", flush=True)
        self.vae.requires_grad_(False)

        # Load schedulers
        self.train_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            diffusion_path,
            subfolder="scheduler",
        )
        self.test_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            diffusion_path,
            subfolder="scheduler",
        )

        # Store config
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGE_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGE_STD), persistent=False)

        # Initialize connector
        self.num_queries = num_queries
        self.connector = ConnectorEncoder(ConnectorConfig(**connector_config))

        # Initialize projectors
        # projector_1: LLM hidden states (6 layers concatenated) -> connector hidden size
        self.projector_1 = nn.Linear(self.llm.config.hidden_size * 6, self.connector.config.hidden_size)
        # projector_2: connector output -> pooled projection for DiT
        self.projector_2 = nn.Linear(self.connector.config.hidden_size, self.transformer.config.pooled_projection_dim)
        # projector_3: connector output -> sequence embedding for DiT
        self.projector_3 = nn.Linear(self.connector.config.hidden_size, self.transformer.config.joint_attention_dim)

        # Zero initialization for projector_2 and projector_3
        nn.init.zeros_(self.projector_2.weight)
        nn.init.zeros_(self.projector_3.weight)
        nn.init.zeros_(self.projector_2.bias)
        nn.init.zeros_(self.projector_3.bias)

        # Meta queries
        self.meta_queries = nn.Parameter(
            torch.zeros(num_queries, self.llm.config.hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(self.llm.config.hidden_size))

        # Training config
        self.unconditional = unconditional
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        # Add LoRA if specified
        if lora_modules is not None and freeze_lmm:
            self.llm.config.tie_word_embeddings = False
            if lora_modules == 'auto':
                lora_modules = find_target_linear_names(self.lmm)
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
                lora_dropout=0.05,
            )
            self.lmm.add_adapter(transformer_lora_config)

    @property
    def llm(self):
        """Get the language model."""
        return self.lmm.language_model

    @property
    def device(self):
        """Get the device."""
        return self.llm.device

    @property
    def dtype(self):
        """Get the data type."""
        return self.llm.dtype

    def llm2dit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform LLM hidden states to DiT embeddings.

        Args:
            x: LLM hidden states of shape (batch, seq_len, hidden_size * 6)

        Returns:
            Tuple of (pooled_out, seq_out) for DiT
        """
        x = self.connector(self.projector_1(x))
        pooled_out = self.projector_2(x.mean(1))
        seq_out = self.projector_3(x)
        return pooled_out, seq_out

    def load_checkpoint(
        self,
        checkpoint_path: str,
        debug_level: int = 0,
        debug_log_dir: Optional[str] = None,
    ) -> Optional[str]:
        """Load a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint (.pt, .pth, or .safetensors)
            debug_level: Debug level for checkpoint loading report
                0 = off (default), 1 = basic summary, 2 = detailed, 3 = verbose
            debug_log_dir: Directory for debug log file (default: current dir or log_dir)

        Returns:
            Path to debug log file if debug_level > 0, otherwise None
        """
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = load_checkpoint(checkpoint_path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:10]}...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}...")

        # Write debug report if enabled
        debug_log_path = None
        if debug_level > 0:
            debug_log_path = self._write_debug_report(
                checkpoint_path=checkpoint_path,
                checkpoint_state_dict=state_dict,
                missing_keys=missing,
                unexpected_keys=unexpected,
                debug_level=debug_level,
                debug_log_dir=debug_log_dir,
            )

        return debug_log_path

    def _write_debug_report(
        self,
        checkpoint_path: str,
        checkpoint_state_dict: Dict[str, torch.Tensor],
        missing_keys: List[str],
        unexpected_keys: List[str],
        debug_level: int,
        debug_log_dir: Optional[str] = None,
    ) -> str:
        """Write checkpoint load debug report to a separate log file.

        Args:
            checkpoint_path: Path to checkpoint file
            checkpoint_state_dict: Loaded state dict from checkpoint
            missing_keys: Keys missing from checkpoint
            unexpected_keys: Keys in checkpoint but not in model
            debug_level: Debug verbosity level (1-3)
            debug_log_dir: Directory for debug log file

        Returns:
            Path to the debug log file
        """
        from datetime import datetime

        # Determine log directory
        if debug_log_dir:
            log_dir = Path(debug_log_dir)
        else:
            log_dir = Path(".")

        log_dir.mkdir(parents=True, exist_ok=True)
        debug_log_path = log_dir / "deepgen_checkpoint_debug.log"

        # Get model's current state_dict (for verification)
        model_state_dict = self.state_dict()

        # Calculate successfully loaded keys
        loaded_keys = [k for k in checkpoint_state_dict.keys() if k not in unexpected_keys]

        # Detect LoRA keys
        lora_keys_in_ckpt = [k for k in checkpoint_state_dict.keys() if "lora_" in k]
        lora_keys_loaded = [k for k in loaded_keys if "lora_" in k]
        lora_keys_missing = [k for k in lora_keys_in_ckpt if k not in lora_keys_loaded]

        with open(debug_log_path, "w", encoding="utf-8") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("DeepGen Checkpoint Load Debug Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Debug Level: {debug_level}\n\n")

            # Basic info
            f.write("=== Basic Info ===\n")
            try:
                ckpt_size = Path(checkpoint_path).stat().st_size / (1024**3)
                f.write(f"Checkpoint file size: {ckpt_size:.2f} GB\n")
            except OSError:
                f.write("Checkpoint file size: N/A\n")
            f.write(f"Checkpoint keys: {len(checkpoint_state_dict)}\n")
            f.write(f"Model trainable keys: {len(model_state_dict)}\n\n")

            # Load summary
            f.write("=== Load Summary ===\n")
            f.write(f"Successfully loaded: {len(loaded_keys)} keys\n")
            f.write(f"Missing keys: {len(missing_keys)}\n")
            f.write(f"Unexpected keys: {len(unexpected_keys)}\n\n")

            # Level 2+: Full missing/unexpected keys list
            if debug_level >= 2:
                if missing_keys:
                    f.write(f"=== Missing Keys ({len(missing_keys)}) ===\n")
                    for k in sorted(missing_keys):
                        f.write(f"  {k}\n")
                    f.write("\n")

                if unexpected_keys:
                    f.write(f"=== Unexpected Keys ({len(unexpected_keys)}) ===\n")
                    for k in sorted(unexpected_keys):
                        f.write(f"  {k}\n")
                    f.write("\n")

            # LoRA check
            f.write("=== LoRA Weights ===\n")
            f.write(f"LoRA modules in checkpoint: {len(lora_keys_in_ckpt)}\n")
            f.write(f"LoRA modules loaded: {len(lora_keys_loaded)}\n")
            lora_all_loaded = len(lora_keys_in_ckpt) == len(lora_keys_loaded) and len(lora_keys_in_ckpt) > 0
            if lora_keys_in_ckpt:
                f.write(f"All LoRA weights loaded: {'YES' if lora_all_loaded else 'NO'}\n")
            else:
                f.write("No LoRA weights in checkpoint\n")

            if debug_level >= 2 and lora_keys_in_ckpt:
                f.write("\nLoRA modules:\n")
                for k in sorted(lora_keys_in_ckpt):
                    status = "✓" if k in lora_keys_loaded else "✗"
                    f.write(f"  [{status}] {k}\n")
            f.write("\n")

            # Level 3: All successfully loaded keys
            if debug_level >= 3:
                f.write(f"=== Successfully Loaded Keys ({len(loaded_keys)}) ===\n")
                for k in sorted(loaded_keys):
                    f.write(f"  {k}\n")
                f.write("\n")

            # Weight statistics and non-zero verification
            f.write("=== Weight Statistics ===\n")
            zero_weights = []

            # Select keys to check based on debug level
            key_components = ["projector_1", "projector_2", "projector_3", "meta_queries", "connector"]
            if debug_level >= 3:
                # Level 3: Check all loaded weights
                check_keys = [k for k in loaded_keys if k in model_state_dict]
            else:
                # Level 1-2: Only check key components
                check_keys = [k for k in loaded_keys if any(c in k for c in key_components)]

            for key in sorted(check_keys):
                if key in model_state_dict:
                    tensor = model_state_dict[key]
                    if tensor.numel() > 0:
                        # Convert to float for statistics
                        tensor_float = tensor.float()
                        mean_val = tensor_float.mean().item()
                        std_val = tensor_float.std().item()
                        min_val = tensor_float.min().item()
                        max_val = tensor_float.max().item()
                        nonzero_count = (tensor != 0).sum().item()
                        nonzero_pct = nonzero_count / tensor.numel() * 100

                        f.write(f"{key}:\n")
                        f.write(f"  shape: {tuple(tensor.shape)}, dtype: {tensor.dtype}\n")
                        f.write(f"  mean: {mean_val:.6f}, std: {std_val:.6f}\n")
                        f.write(f"  min: {min_val:.6f}, max: {max_val:.6f}\n")
                        f.write(f"  non-zero: {nonzero_pct:.1f}% ({nonzero_count}/{tensor.numel()})\n")

                        if nonzero_pct == 0:
                            zero_weights.append(key)
            f.write("\n")

            # Verification result
            f.write("=== Verification Result ===\n")
            if zero_weights:
                f.write(f"[FAIL] Found {len(zero_weights)} zero-weight tensor(s):\n")
                for k in zero_weights:
                    f.write(f"  - {k}\n")
            else:
                f.write("[PASS] All checked weights are non-zero\n")

            if lora_keys_in_ckpt:
                if lora_all_loaded:
                    f.write("[PASS] All LoRA weights loaded successfully\n")
                else:
                    f.write(f"[FAIL] {len(lora_keys_missing)} LoRA weight(s) failed to load:\n")
                    for k in lora_keys_missing:
                        f.write(f"  - {k}\n")

            if missing_keys:
                f.write(f"[WARN] {len(missing_keys)} missing key(s) (may be expected for frozen components)\n")

            if unexpected_keys:
                f.write(f"[WARN] {len(unexpected_keys)} unexpected key(s) in checkpoint\n")

            f.write("=" * 80 + "\n")

        return str(debug_log_path)

    def state_dict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Get state dict excluding frozen components."""
        state_dict = super().state_dict(*args, **kwargs)
        # Exclude VAE, LMM base weights, and EMA
        state_dict = {k: v for k, v in state_dict.items()
                      if 'vae.' not in k and 'lmm.' not in k and 'ema.' not in k}
        return state_dict

    @torch.no_grad()
    def pixels_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latents using VAE.

        Args:
            x: Pixel values of shape (batch, channels, height, width) in [-1, 1]

        Returns:
            Latents
        """
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixels using VAE.

        Args:
            z: Latents

        Returns:
            Pixel values in [-1, 1]
        """
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x_rec = self.vae.decode(z).sample
        return x_rec

    @torch.no_grad()
    def get_semantic_features_dynamic(
        self,
        pixel_values: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Get semantic features from images with dynamic resolution.

        Args:
            pixel_values: List of pixel tensors

        Returns:
            Tuple of (image_embeds list, image_grid_thw tensor)
        """
        # Get vision config parameters
        patch_size = self.lmm.config.vision_config.patch_size
        spatial_merge_size = self.lmm.config.vision_config.spatial_merge_size

        # The image dimensions must be divisible by (patch_size * spatial_merge_size)
        align_size = patch_size * spatial_merge_size

        # Resize images to ensure proper alignment
        resized_pixel_values = []
        for p in pixel_values:
            # p shape: (C, H, W)
            _, h, w = p.shape

            # Scale by 28/32 first
            new_h = int(h * 28 / 32)
            new_w = int(w * 28 / 32)

            # Align to patch_size * spatial_merge_size
            new_h = (new_h // align_size) * align_size
            new_w = (new_w // align_size) * align_size

            # Ensure minimum size
            new_h = max(new_h, align_size)
            new_w = max(new_w, align_size)

            # Resize
            resized = F.interpolate(p[None], size=(new_h, new_w), mode='bilinear', align_corners=False)
            resized_pixel_values.append(resized)

        image_embeds, image_grid_thw = multi_apply(self.get_semantic_features, resized_pixel_values, resize=False)
        image_embeds = [x[0] for x in image_embeds]
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
        return image_embeds, image_grid_thw

    @torch.no_grad()
    def get_semantic_features(
        self,
        pixel_values: torch.Tensor,
        resize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get semantic features from a single image.

        Args:
            pixel_values: Pixel values in [-1, 1]
            resize: Whether to resize to vit_input_size

        Returns:
            Tuple of (image_embeds, image_grid_thw)
        """
        # Normalize to [0, 1] then apply VIT normalization
        pixel_values = (pixel_values + 1.0) / 2
        # Ensure vit_mean and vit_std are on the same device as pixel_values
        vit_mean = self.vit_mean.to(pixel_values.device).view(1, 3, 1, 1)
        vit_std = self.vit_std.to(pixel_values.device).view(1, 3, 1, 1)
        pixel_values = pixel_values - vit_mean
        pixel_values = pixel_values / vit_std

        if resize:
            pixel_values = F.interpolate(pixel_values, size=(self.vit_input_size, self.vit_input_size),
                                         mode='bilinear')
        b, c, h, w = pixel_values.shape

        patch_size = self.lmm.config.vision_config.patch_size
        spatial_merge_size = self.lmm.config.vision_config.spatial_merge_size
        temporal_patch_size = self.lmm.config.vision_config.temporal_patch_size

        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)

        grid_t = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        pixel_values = pixel_values.view(
            b,
            grid_t,
            temporal_patch_size,
            c,
            grid_h // spatial_merge_size,
            spatial_merge_size,
            patch_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
            patch_size,
        )

        pixel_values = rearrange(
            pixel_values, 'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(self.device).long()

        image_embeds = self.lmm.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, image_grid_thw

    def prepare_forward_input(
        self,
        query_embeds: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for LLM forward pass.

        Args:
            query_embeds: Query embeddings
            input_ids: Input token IDs
            image_embeds: Image embeddings
            image_grid_thw: Image grid dimensions
            attention_mask: Attention mask
            past_key_values: Past key values for caching

        Returns:
            Dict with inputs_embeds, attention_mask, position_ids, past_key_values
        """
        b, l, _ = query_embeds.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)

        assert l == self.num_queries

        input_ids = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

        position_ids, _ = self.lmm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        if past_key_values is not None:
            inputs_embeds = query_embeds
            position_ids = position_ids[..., -l:]
        else:
            input_ids = input_ids[:, :-l]

            if image_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                            device=self.device, dtype=self.dtype)
                inputs_embeds[input_ids == self.image_token_id] = \
                    image_embeds.contiguous().view(-1, self.llm.config.hidden_size)
                inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
                    input_ids[input_ids != self.image_token_id]
                )

            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

        inputs = dict(inputs_embeds=inputs_embeds,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values)

        return inputs

    @torch.no_grad()
    def prepare_text2image_prompts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Prepare prompts for text-to-image generation.

        Args:
            texts: List of text prompts

        Returns:
            Tokenized inputs
        """
        texts = [self.prompt_template['GENERATION'].format(input=text) for text in texts]
        texts = [self.prompt_template['INSTRUCTION'].format(input=text) for text in texts]

        return self.tokenizer(
            texts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)

    @torch.no_grad()
    def prepare_image2image_prompts(
        self,
        texts: List[str],
        num_refs: List[int],
        ref_lens: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Prepare prompts for image-to-image editing.

        Args:
            texts: List of text instructions
            num_refs: Number of reference images per sample
            ref_lens: Length of each reference image embedding

        Returns:
            Tokenized inputs
        """
        prompts = []
        cnt = 0
        for text, num_ref in zip(texts, num_refs):
            image_tokens = ''
            for _ in range(num_ref):
                image_tokens += self.prompt_template['IMG_START_TOKEN'] + \
                               self.prompt_template['IMG_CONTEXT_TOKEN'] * ref_lens[cnt] + \
                               self.prompt_template['IMG_END_TOKEN']
                cnt += 1

            prompts.append(self.prompt_template['INSTRUCTION'].format(input=f'{image_tokens}\n{text}'))

        return self.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        cfg_prompt: Union[str, List[str]] = "",
        pixel_values_src: Optional[List[List[torch.Tensor]]] = None,
        cfg_scale: float = 4.5,
        num_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        height: int = 512,
        width: int = 512,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """Generate images from text prompts or edit existing images.

        Args:
            prompt: Text prompt(s) for generation/editing
            cfg_prompt: CFG prompt(s) for classifier-free guidance
            pixel_values_src: Source images for editing (list of lists)
            cfg_scale: Classifier-free guidance scale
            num_steps: Number of diffusion steps
            generator: Random generator for reproducibility
            height: Output image height
            width: Output image width
            progress_bar: Whether to show progress bar

        Returns:
            Generated images as tensor in [-1, 1]
        """
        # Normalize inputs
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(cfg_prompt, str):
            cfg_prompt = [cfg_prompt] * len(prompt)

        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)

        # Prepare inputs based on whether we have source images
        if pixel_values_src is not None:
            # Image editing mode
            num_refs = [len(ref_images) for ref_images in pixel_values_src]
            pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_imgs]
                                for ref_imgs in pixel_values_src]
            image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
                [img for ref_images in pixel_values_src for img in ref_images])
            ref_lens = [len(x) for x in image_embeds]

            text_inputs = self.prepare_image2image_prompts(prompt + cfg_prompt, num_refs=num_refs * 2,
                                                           ref_lens=ref_lens * 2)
            text_inputs.update(image_embeds=torch.cat(image_embeds * 2),
                               image_grid_thw=torch.cat([image_grid_thw] * 2))
            cond_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_imgs]
                            for ref_imgs in pixel_values_src]
            cond_latents = cond_latents * 2
        else:
            # Text-to-image mode
            text_inputs = self.prepare_text2image_prompts(prompt + cfg_prompt)
            cond_latents = None

        # Forward through LLM
        # Ensure meta_queries is on the correct device
        meta_queries = self.meta_queries.to(self.device)
        hidden_states = meta_queries[None].expand(2 * b, self.num_queries, -1)
        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        output = self.llm(**inputs, return_dict=True, output_hidden_states=True)

        # Extract and merge hidden states from multiple layers
        hidden_states = output.hidden_states
        num_layers = len(hidden_states) - 1

        # Select layers: every 6th layer from the end
        selected_layers = list(range(num_layers - 1, 0, -6))
        selected_hiddens = [hidden_states[i] for i in selected_layers]
        merged_hidden = torch.cat(selected_hiddens, dim=-1)

        # Transform to DiT embeddings
        pooled_out, seq_out = self.llm2dit(merged_hidden)

        # Create pipeline and generate
        pipeline = StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.test_scheduler,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
            text_encoder_3=None,
            tokenizer_3=None,
        )

        pipeline.set_progress_bar_config(disable=not progress_bar)

        samples = pipeline(
            height=height,
            width=width,
            guidance_scale=cfg_scale,
            num_inference_steps=num_steps,
            prompt_embeds=seq_out[:b],
            pooled_prompt_embeds=pooled_out[:b],
            negative_prompt_embeds=seq_out[b:],
            negative_pooled_prompt_embeds=pooled_out[b:],
            generator=generator,
            output_type='latent',
            cond_latents=cond_latents,
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples)


def resize_image(x, image_size: int, unit_image_size: int = 32):
    """Resize image while maintaining aspect ratio.

    Args:
        x: PIL Image
        image_size: Target size for the longer edge
        unit_image_size: Size unit for rounding

    Returns:
        Resized PIL Image
    """
    w, h = x.size
    if w >= h and w >= image_size:
        target_w = image_size
        target_h = h * (target_w / w)
        target_h = math.ceil(target_h / unit_image_size) * unit_image_size

    elif h >= w and h >= image_size:
        target_h = image_size
        target_w = w * (target_h / h)
        target_w = math.ceil(target_w / unit_image_size) * unit_image_size

    else:
        target_h = math.ceil(h / unit_image_size) * unit_image_size
        target_w = math.ceil(w / unit_image_size) * unit_image_size

    x = x.resize(size=(target_w, target_h))

    return x
