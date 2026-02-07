# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen model implementation.

This module implements the DeepGen model for unified visual generation,
supporting both text-to-image generation and image editing.

The model architecture consists of:
- Qwen2.5-VL as the language/vision understanding module
- SD3.5 Transformer as the image generation module
- Connector module to bridge LLM and DiT
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from diffgentor.models.deepgen.connector import ConnectorConfig, ConnectorEncoder

if TYPE_CHECKING:
    from diffgentor.utils.env import DeepGenEnv


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def multi_apply(func, *args, **kwargs):
    """Apply function to multiple arguments."""
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    """Find target linear layer names for LoRA."""
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


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate shift for flow matching scheduler."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@dataclass
class DeepGenModelConfig:
    """Configuration for DeepGen model.

    Args:
        diffusion_model_path: Path to diffusion model (SD3.5)
        ar_model_path: Path to AR model (Qwen2.5-VL)
        num_queries: Number of query tokens (default: 128)
        connector_hidden_size: Connector hidden size (default: 2048)
        connector_intermediate_size: Connector intermediate size (default: 11946)
        connector_num_layers: Number of connector layers (default: 6)
        connector_num_heads: Number of connector attention heads (default: 32)
        connector_attn_impl: Connector attention implementation (default: flash_attention_2)
        max_length: Maximum sequence length (default: 1024)
        freeze_lmm: Whether to freeze LMM (default: True)
        freeze_transformer: Whether to freeze transformer (default: True)
        lora_rank: LoRA rank (default: 64)
        lora_alpha: LoRA alpha (default: 128)
        unconditional_prob: Unconditional probability for CFG training (default: 0.1)
        prompt_template: Prompt template dict (loaded from config file)
    """

    diffusion_model_path: str
    ar_model_path: str
    num_queries: int = 128
    connector_hidden_size: int = 2048
    connector_intermediate_size: int = 11946
    connector_num_layers: int = 6
    connector_num_heads: int = 32
    connector_attn_impl: str = "flash_attention_2"
    max_length: int = 1024
    freeze_lmm: bool = True
    freeze_transformer: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    unconditional_prob: float = 0.1
    prompt_template: Optional[Dict[str, Any]] = None

    @classmethod
    def from_env(cls, env: "DeepGenEnv") -> "DeepGenModelConfig":
        """Create config from DeepGenEnv.

        Args:
            env: DeepGenEnv instance with loaded configuration

        Returns:
            DeepGenModelConfig instance

        Raises:
            ValueError: If required model paths are not provided
        """
        diffusion_path = env.diffusion_model_path
        ar_path = env.ar_model_path

        if not diffusion_path:
            raise ValueError(
                "Diffusion model path not specified. "
                "Set DG_DEEPGEN_DIFFUSION_MODEL_PATH environment variable."
            )
        if not ar_path:
            raise ValueError(
                "AR model path not specified. "
                "Set DG_DEEPGEN_AR_MODEL_PATH environment variable."
            )

        return cls(
            diffusion_model_path=diffusion_path,
            ar_model_path=ar_path,
            num_queries=env.num_queries,
            connector_hidden_size=env.connector_hidden_size,
            connector_intermediate_size=env.connector_intermediate_size,
            connector_num_layers=env.connector_num_layers,
            connector_num_heads=env.connector_num_heads,
            connector_attn_impl=env.connector_attn_impl,
            max_length=env.max_length,
            freeze_lmm=env.freeze_lmm,
            freeze_transformer=env.freeze_transformer,
            lora_rank=env.lora_rank,
            lora_alpha=env.lora_alpha,
            prompt_template=env.prompt_template,
        )


class DeepGenModel(nn.Module):
    """DeepGen model for unified visual generation.

    This model combines Qwen2.5-VL for language/vision understanding with
    SD3.5 Transformer for image generation, connected via a Connector module.

    Supports:
    - Text-to-image generation
    - Image editing with instruction

    Args:
        config: Model configuration
        pretrained_pth: Path to pretrained weights (optional)
        use_activation_checkpointing: Enable gradient checkpointing (default: False)
    """

    # Default prompt template (used if not provided in config)
    DEFAULT_PROMPT_TEMPLATE = {
        "IMG_START_TOKEN": "<|vision_start|>",
        "IMG_END_TOKEN": "<|vision_end|>",
        "IMG_CONTEXT_TOKEN": "<|image_pad|>",
        "IMG_START_TOKEN_FOR_GENERATION": False,
        "SYSTEM": "<|im_start|>system\n{system}<|im_end|>\n",
        "INSTRUCTION": "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
        "SUFFIX": "<|im_end|>",
        "SUFFIX_AS_EOS": True,
        "SEP": "\n",
        "STOP_WORDS": ["<|im_end|>", "<|endoftext|>"],
        "GENERATION": "Generate an image: {input}",
        "CFG": "Generate an image.",
    }

    # Hardcoded ViT input size
    VIT_INPUT_SIZE = 448

    def __init__(
        self,
        config: DeepGenModelConfig,
        pretrained_pth: Optional[str] = None,
        use_activation_checkpointing: bool = False,
        debug_checkpoint: bool = False,
    ):
        super().__init__()
        self.config = config
        self._debug_checkpoint = debug_checkpoint

        # Use prompt template from config or default
        self.prompt_template = config.prompt_template or self.DEFAULT_PROMPT_TEMPLATE

        # Load Qwen2.5-VL (AR model)
        self.lmm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.ar_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        if config.freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = config.freeze_lmm

        # Load SD3.5 Transformer (Diffusion model)
        self.transformer = SD3Transformer2DModel.from_pretrained(
            config.diffusion_model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        if config.freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = config.freeze_transformer

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            config.diffusion_model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        self.vae.requires_grad_(False)

        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            config.diffusion_model_path,
            subfolder="scheduler",
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.ar_model_path,
            trust_remote_code=True,
            padding_side="right",
        )

        # Setup image token (hardcoded vit_input_size)
        self.vit_input_size = self.VIT_INPUT_SIZE
        self.max_length = config.max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_template["IMG_CONTEXT_TOKEN"])
        self.register_buffer("vit_mean", torch.tensor(IMAGE_MEAN), persistent=False)
        self.register_buffer("vit_std", torch.tensor(IMAGE_STD), persistent=False)

        # Build connector
        self.num_queries = config.num_queries
        connector_config = ConnectorConfig(
            hidden_size=config.connector_hidden_size,
            intermediate_size=config.connector_intermediate_size,
            num_hidden_layers=config.connector_num_layers,
            num_attention_heads=config.connector_num_heads,
            _attn_implementation=config.connector_attn_impl,
        )
        self.connector = ConnectorEncoder(connector_config)

        # Build projectors
        llm_hidden_size = self.llm.config.hidden_size
        self.projector_1 = nn.Linear(llm_hidden_size * 6, config.connector_hidden_size)
        self.projector_2 = nn.Linear(config.connector_hidden_size, self.transformer.config.pooled_projection_dim)
        self.projector_3 = nn.Linear(config.connector_hidden_size, self.transformer.config.joint_attention_dim)

        # Zero initialize output projectors
        nn.init.zeros_(self.projector_2.weight)
        nn.init.zeros_(self.projector_3.weight)
        nn.init.zeros_(self.projector_2.bias)
        nn.init.zeros_(self.projector_3.bias)

        # Meta queries
        self.meta_queries = nn.Parameter(torch.zeros(config.num_queries, llm_hidden_size))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(llm_hidden_size))

        self.unconditional_prob = config.unconditional_prob

        # Setup activation checkpointing
        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        # Add LoRA if LMM is frozen
        if config.freeze_lmm and config.lora_rank > 0:
            self.llm.config.tie_word_embeddings = False
            lora_modules = find_target_linear_names(self.lmm)
            transformer_lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
                lora_dropout=0.05,
            )
            self.lmm.add_adapter(transformer_lora_config)

        # Load pretrained weights
        if pretrained_pth is not None:
            self._load_pretrained(pretrained_pth)

    def _load_pretrained(self, pretrained_pth: str):
        """Load pretrained weights.

        If debug_checkpoint is enabled (DG_DEEPGEN_DEBUG_CHECKPOINT=1),
        writes detailed checkpoint loading info to {log_dir}/checkpoint_debug.log
        """
        if pretrained_pth.endswith(".pt"):
            state_dict = torch.load(pretrained_pth, map_location="cpu")
        else:
            from safetensors.torch import load_file

            state_dict = load_file(pretrained_pth, device="cpu")

        # Get model and checkpoint keys
        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # Compute differences
        matched_keys = model_keys & ckpt_keys
        missing_keys = model_keys - ckpt_keys  # In model but not in checkpoint
        unexpected_keys = ckpt_keys - model_keys  # In checkpoint but not in model

        # Load state dict
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_pth}")
        print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        # Write debug log if enabled
        if self._debug_checkpoint:
            self._write_checkpoint_debug_log(
                pretrained_pth=pretrained_pth,
                ckpt_keys=ckpt_keys,
                model_keys=model_keys,
                matched_keys=matched_keys,
                missing_keys=missing_keys,
                unexpected_keys=unexpected_keys,
            )

    def _write_checkpoint_debug_log(
        self,
        pretrained_pth: str,
        ckpt_keys: set,
        model_keys: set,
        matched_keys: set,
        missing_keys: set,
        unexpected_keys: set,
    ):
        """Write checkpoint debug log to file (rank 0 only).

        Args:
            pretrained_pth: Path to checkpoint file
            ckpt_keys: Keys in checkpoint
            model_keys: Keys in model
            matched_keys: Keys in both
            missing_keys: Keys in model but not in checkpoint
            unexpected_keys: Keys in checkpoint but not in model
        """
        from datetime import datetime
        from pathlib import Path

        from diffgentor.utils.distributed import is_main_process
        from diffgentor.utils.logging import get_log_dir

        # Only rank 0 writes debug log
        if not is_main_process():
            return

        log_dir = get_log_dir()
        if not log_dir:
            print("Warning: log_dir not set, skipping checkpoint debug log")
            return

        debug_log_path = Path(log_dir) / "checkpoint_debug.log"

        def categorize_keys(keys):
            """Categorize keys by component."""
            categories = {
                "lora": [],
                "connector": [],
                "projector": [],
                "meta_queries": [],
                "transformer": [],
                "lmm": [],
                "vae": [],
                "other": [],
            }
            for k in sorted(keys):
                if "lora" in k.lower():
                    categories["lora"].append(k)
                elif "connector" in k:
                    categories["connector"].append(k)
                elif "projector" in k:
                    categories["projector"].append(k)
                elif "meta_queries" in k:
                    categories["meta_queries"].append(k)
                elif k.startswith("transformer."):
                    categories["transformer"].append(k)
                elif k.startswith("lmm."):
                    categories["lmm"].append(k)
                elif k.startswith("vae."):
                    categories["vae"].append(k)
                else:
                    categories["other"].append(k)
            return categories

        def write_category_section(f, title, keys_set):
            """Write a categorized section to file."""
            f.write(f"\n{'=' * 80}\n")
            f.write(f"{title}\n")
            f.write(f"{'=' * 80}\n")
            categories = categorize_keys(keys_set)
            for cat_name, cat_keys in categories.items():
                f.write(f"\n[{cat_name}] ({len(cat_keys)} keys)\n")
                for key in cat_keys:
                    f.write(f"{key}\n")

        with open(debug_log_path, "w") as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("DeepGen Checkpoint Debug Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"Checkpoint: {pretrained_pth}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Summary
            f.write(f"\n{'=' * 80}\n")
            f.write("Summary\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Total checkpoint keys: {len(ckpt_keys)}\n")
            f.write(f"Total model keys: {len(model_keys)}\n")
            f.write(f"Matched keys: {len(matched_keys)}\n")
            f.write(f"Missing keys (in model but not in checkpoint): {len(missing_keys)}\n")
            f.write(f"Unexpected keys (in checkpoint but not in model): {len(unexpected_keys)}\n")

            # Detailed sections
            write_category_section(f, "Checkpoint Keys by Category", ckpt_keys)
            write_category_section(f, "Model Keys by Category", model_keys)
            write_category_section(f, "Missing Keys by Category (in model but not in checkpoint)", missing_keys)
            write_category_section(f, "Unexpected Keys by Category (in checkpoint but not in model)", unexpected_keys)
            write_category_section(f, "Matched Keys (successfully loaded)", matched_keys)

        print(f"Checkpoint debug log saved to: {debug_log_path}")

    def llm2dit(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform LLM hidden states to DiT embeddings.

        Args:
            x: LLM hidden states of shape (batch, seq_len, hidden_size * 6)

        Returns:
            Tuple of (pooled_out, seq_out) for DiT conditioning
        """
        x = self.connector(self.projector_1(x))
        pooled_out = self.projector_2(x.mean(1))
        seq_out = self.projector_3(x)
        return pooled_out, seq_out

    @property
    def llm(self):
        """Get the language model (returns full LMM for compatibility with DPO Fusion)."""
        return self.lmm

    @property
    def device(self):
        """Get model device."""
        return self.llm.device

    @property
    def dtype(self):
        """Get model dtype."""
        return self.llm.dtype

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.llm.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.llm.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()
        return self

    @torch.no_grad()
    def pixels_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixels to latents.

        Args:
            x: Pixel values of shape (batch, 3, height, width) in range [-1, 1]

        Returns:
            Latents of shape (batch, channels, height/8, width/8)
        """
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixels.

        Args:
            z: Latents of shape (batch, channels, height/8, width/8)

        Returns:
            Pixel values of shape (batch, 3, height, width) in range [-1, 1]
        """
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x_rec = self.vae.decode(z).sample
        return x_rec

    @torch.no_grad()
    def get_semantic_features_dynamic(self, pixel_values: List[torch.Tensor]):
        """Get semantic features with dynamic resolution.

        Note: Input images MUST have dimensions that are multiples of 32.
        Use resize_for_deepgen() in the backend to ensure this.

        Args:
            pixel_values: List of pixel tensors, each of shape (C, H, W)
                         Dimensions must be multiples of 32.

        Returns:
            Tuple of (image_embeds, image_grid_thw)
        """
        # Scale from input size to 28/32 of input size
        # e.g., 512 -> 448, 480 -> 420
        # Input must be multiples of 32 so output is multiples of 28
        pixel_values = [F.interpolate(p[None], scale_factor=28 / 32, mode="bilinear") for p in pixel_values]
        image_embeds, image_grid_thw = multi_apply(self.get_semantic_features, pixel_values, resize=False)
        image_embeds = [x[0] for x in image_embeds]
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
        return image_embeds, image_grid_thw

    @torch.no_grad()
    def get_semantic_features(self, pixel_values: torch.Tensor, resize: bool = True):
        """Get semantic features from pixel values.

        Args:
            pixel_values: Pixel values in range [-1, 1]
            resize: Whether to resize to vit_input_size

        Returns:
            Tuple of (image_embeds, image_grid_thw)
        """
        # Normalize to [0, 1] then apply ViT normalization
        pixel_values = (pixel_values + 1.0) / 2
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        if resize:
            pixel_values = F.interpolate(
                pixel_values, size=(self.vit_input_size, self.vit_input_size), mode="bilinear"
            )
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

        pixel_values = rearrange(pixel_values, "b t tp c h m p w n q -> (b t h w m n) (c tp p q)")

        image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(self.device).long()

        image_embeds = self.lmm.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, "(b l) d -> b l d", b=b)

        return image_embeds, image_grid_thw

    @torch.no_grad()
    def prepare_text2image_prompts(self, texts: List[str]):
        """Prepare prompts for text-to-image generation.

        Args:
            texts: List of text prompts

        Returns:
            Tokenized inputs
        """
        texts = [self.prompt_template["GENERATION"].format(input=text) for text in texts]
        texts = [self.prompt_template["INSTRUCTION"].format(input=text) for text in texts]

        return self.tokenizer(texts, add_special_tokens=True, return_tensors="pt", padding=True, padding_side="left").to(
            self.device
        )

    @torch.no_grad()
    def prepare_image2image_prompts(self, texts: List[str], num_refs: List[int], ref_lens: List[int]):
        """Prepare prompts for image editing.

        Args:
            texts: List of instruction texts
            num_refs: Number of reference images per sample
            ref_lens: Length of each reference image embedding

        Returns:
            Tokenized inputs
        """
        prompts = []
        cnt = 0
        for text, num_ref in zip(texts, num_refs):
            image_tokens = ""
            for _ in range(num_ref):
                image_tokens += (
                    self.prompt_template["IMG_START_TOKEN"]
                    + self.prompt_template["IMG_CONTEXT_TOKEN"] * ref_lens[cnt]
                    + self.prompt_template["IMG_END_TOKEN"]
                )
                cnt += 1

            prompts.append(self.prompt_template["INSTRUCTION"].format(input=f"{image_tokens}\n{text}"))

        return self.tokenizer(prompts, add_special_tokens=True, return_tensors="pt", padding=True, padding_side="left").to(
            self.device
        )

    def prepare_forward_input(
        self,
        query_embeds: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for forward pass.

        Args:
            query_embeds: Query embeddings
            input_ids: Input token IDs
            image_embeds: Image embeddings
            image_grid_thw: Image grid dimensions
            attention_mask: Attention mask
            past_key_values: Past key values for caching

        Returns:
            Dictionary of inputs for LLM forward
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

        # Prepare context
        if past_key_values is not None:
            inputs_embeds = query_embeds
            position_ids = position_ids[..., -l:]
        else:
            input_ids = input_ids[:, :-l]  # context input_ids

            if image_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = torch.zeros(
                    *input_ids.shape, self.llm.config.hidden_size, device=self.device, dtype=self.dtype
                )
                inputs_embeds[input_ids == self.image_token_id] = image_embeds.contiguous().view(
                    -1, self.llm.config.hidden_size
                )
                inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
                    input_ids[input_ids != self.image_token_id]
                )

            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

        inputs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )

        return inputs

    @torch.no_grad()
    def generate(
        self,
        prompt: List[str],
        cfg_prompt: List[str],
        pixel_values_src: Optional[List[List[torch.Tensor]]] = None,
        cfg_scale: float = 4.5,
        num_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        height: int = 512,
        width: int = 512,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """Generate images.

        Args:
            prompt: List of prompts
            cfg_prompt: List of CFG prompts (negative prompts)
            pixel_values_src: Optional list of source images for editing
            cfg_scale: CFG guidance scale
            num_steps: Number of inference steps
            generator: Random generator
            height: Output height
            width: Output width
            progress_bar: Show progress bar

        Returns:
            Generated images as tensor in range [-1, 1]
        """
        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)

        if pixel_values_src is not None:
            # Image editing mode
            num_refs = [len(ref_images) for ref_images in pixel_values_src]
            pixel_values_src = [
                [img.to(dtype=self.dtype, device=self.device) for img in ref_imgs] for ref_imgs in pixel_values_src
            ]
            image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
                [img for ref_images in pixel_values_src for img in ref_images]
            )
            ref_lens = [len(x) for x in image_embeds]

            text_inputs = self.prepare_image2image_prompts(
                prompt + cfg_prompt, num_refs=num_refs * 2, ref_lens=ref_lens * 2
            )
            text_inputs.update(
                image_embeds=torch.cat(image_embeds * 2),
                image_grid_thw=torch.cat([image_grid_thw] * 2),
            )
            cond_latents = [
                [self.pixels_to_latents(img[None])[0] for img in ref_imgs] for ref_imgs in pixel_values_src
            ]
            cond_latents = cond_latents * 2
        else:
            # Text-to-image mode
            text_inputs = self.prepare_text2image_prompts(prompt + cfg_prompt)
            cond_latents = None

        hidden_states = self.meta_queries[None].expand(2 * b, self.num_queries, -1)
        inputs = self.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

        output = self.llm(**inputs, return_dict=True, output_hidden_states=True)

        # Extract hidden states from multiple layers
        hidden_states = output.hidden_states
        num_layers = len(hidden_states) - 1

        # Select layers: every 6th layer from the end
        selected_layers = list(range(num_layers - 1, 0, -6))
        selected_hiddens = [hidden_states[i] for i in selected_layers]
        merged_hidden = torch.cat(selected_hiddens, dim=-1)
        pooled_out, seq_out = self.llm2dit(merged_hidden)

        # Create pipeline for inference
        pipeline = StableDiffusion3Pipeline(
            transformer=self.transformer,
            scheduler=self.scheduler,
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
            output_type="latent",
        ).images.to(self.dtype)

        return self.latents_to_pixels(samples)

    @torch.no_grad()
    def generate_t2i(
        self,
        prompt: Union[str, List[str]],
        cfg_prompt: Optional[Union[str, List[str]]] = None,
        cfg_scale: float = 4.5,
        num_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        height: int = 512,
        width: int = 512,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            cfg_prompt: CFG prompt(s) for negative guidance
            cfg_scale: CFG guidance scale
            num_steps: Number of inference steps
            generator: Random generator
            height: Output height
            width: Output width
            progress_bar: Show progress bar

        Returns:
            Generated images as tensor in range [-1, 1]
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        if cfg_prompt is None:
            cfg_prompt = [""] * len(prompt)
        elif isinstance(cfg_prompt, str):
            cfg_prompt = [cfg_prompt] * len(prompt)

        return self.generate(
            prompt=prompt,
            cfg_prompt=cfg_prompt,
            pixel_values_src=None,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            generator=generator,
            height=height,
            width=width,
            progress_bar=progress_bar,
        )

    @torch.no_grad()
    def generate_edit(
        self,
        images: List[torch.Tensor],
        instruction: Union[str, List[str]],
        cfg_instruction: Optional[Union[str, List[str]]] = None,
        cfg_scale: float = 4.5,
        num_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """Edit images with instruction.

        Args:
            images: List of input images as tensors in range [-1, 1]
            instruction: Editing instruction(s)
            cfg_instruction: CFG instruction(s) for negative guidance
            cfg_scale: CFG guidance scale
            num_steps: Number of inference steps
            generator: Random generator
            height: Output height (default: same as input)
            width: Output width (default: same as input)
            progress_bar: Show progress bar

        Returns:
            Edited images as tensor in range [-1, 1]
        """
        if isinstance(instruction, str):
            instruction = [instruction] * len(images)

        if cfg_instruction is None:
            cfg_instruction = [""] * len(instruction)
        elif isinstance(cfg_instruction, str):
            cfg_instruction = [cfg_instruction] * len(instruction)

        # Determine output size from first image if not specified
        if height is None or width is None:
            h, w = images[0].shape[-2:]
            height = height or h
            width = width or w

        # Wrap each image in a list (single reference per sample)
        pixel_values_src = [[img] for img in images]

        return self.generate(
            prompt=instruction,
            cfg_prompt=cfg_instruction,
            pixel_values_src=pixel_values_src,
            cfg_scale=cfg_scale,
            num_steps=num_steps,
            generator=generator,
            height=height,
            width=width,
            progress_bar=progress_bar,
        )
