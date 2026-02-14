# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""BAGEL backend using ByteDance's BAGEL model."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.distributed import get_local_rank, get_world_size, is_distributed
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.env import BagelEnv


class BagelBackend(BaseEditingBackend):
    """BAGEL backend using ByteDance's BAGEL multimodal model.

    Uses data parallelism (torchrun) for multi-GPU inference - each GPU loads
    a complete model instance and processes different data.

    Requires the bagel submodule to be initialized:
        git submodule update --init diffgentor/models/third_party/bagel

    Model-specific parameters via environment variables:
        DG_BAGEL_CFG_TEXT_SCALE: Text CFG scale (default: 3.0)
        DG_BAGEL_CFG_IMG_SCALE: Image CFG scale (default: 1.5)
        DG_BAGEL_CFG_INTERVAL: CFG interval as comma-separated floats (default: 0.4,1.0)
        DG_BAGEL_TIMESTEP_SHIFT: Timestep shift value (default: 3.0)
        DG_BAGEL_NUM_TIMESTEPS: Number of denoising timesteps (default: 50)
        DG_BAGEL_THINK: Enable thinking mode (default: false)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize BAGEL backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._inferencer = None
        self._model = None
        self._vae_model = None
        self._tokenizer = None
        self._cfg_text_scale = BagelEnv.cfg_text_scale()
        self._cfg_img_scale = BagelEnv.cfg_img_scale()
        self._cfg_interval = BagelEnv.cfg_interval()
        self._timestep_shift = BagelEnv.timestep_shift()

    def load_model(self, **kwargs) -> None:
        """Load BAGEL model.

        Uses data parallelism - each GPU loads a complete model instance.
        Model is loaded to the local GPU based on LOCAL_RANK.

        Model-specific parameters are read from environment variables:
            DG_BAGEL_CFG_TEXT_SCALE: Text CFG scale
            DG_BAGEL_CFG_IMG_SCALE: Image CFG scale
            DG_BAGEL_CFG_INTERVAL: CFG interval range (comma-separated)
            DG_BAGEL_TIMESTEP_SHIFT: Timestep shift value

        Args:
            **kwargs: Additional arguments
        """
        # Read from environment variables
        self._cfg_text_scale = BagelEnv.cfg_text_scale()
        self._cfg_img_scale = BagelEnv.cfg_img_scale()
        self._cfg_interval = BagelEnv.cfg_interval()
        self._timestep_shift = BagelEnv.timestep_shift()

        # Add third-party path
        third_party_path = Path(__file__).parent.parent.parent / "models" / "third_party" / "bagel"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
        else:
            raise ImportError(
                f"BAGEL vendored code not found at {third_party_path}. "
                "Please reinstall diffgentor: pip install diffgentor"
            )

        try:
            from safetensors.torch import load_file
            from data.data_utils import add_special_tokens
            from data.transforms import ImageTransform
            from inferencer import InterleaveInferencer
            from modeling.autoencoder import load_ae
            from modeling.bagel import (
                BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
                SiglipVisionConfig, SiglipVisionModel
            )
            from modeling.qwen2 import Qwen2Tokenizer

            print_rank0(f"Loading BAGEL from: {self.model_name}")

            # Determine device based on distributed mode
            env_world_size = int(os.environ.get("WORLD_SIZE", 1))
            env_local_rank = os.environ.get("LOCAL_RANK")
            if is_distributed() or env_world_size > 1 or env_local_rank is not None:
                local_rank = get_local_rank()
                device = f"cuda:{local_rank}"
                torch.cuda.set_device(local_rank)
                print_rank0(f"Distributed mode: device {device} (world_size={get_world_size()})")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            model_path = self.model_name

            # Load configs
            llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
            llm_config.qk_norm = True
            llm_config.tie_word_embeddings = False
            llm_config.layer_module = "Qwen2MoTDecoderLayer"

            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
            vit_config.rope = False
            vit_config.num_hidden_layers -= 1

            # Load VAE
            self._vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

            # Create model config
            config = BagelConfig(
                visual_gen=True,
                visual_und=True,
                llm_config=llm_config,
                vit_config=vit_config,
                vae_config=vae_config,
                vit_max_num_patch_per_side=70,
                connector_act="gelu_pytorch_tanh",
                latent_patch_size=2,
                max_latent_size=64,
            )

            # Build model
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            self._model = Bagel(language_model, vit_model, config)
            self._model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

            # Load tokenizer
            self._tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
            self._tokenizer, new_token_ids, _ = add_special_tokens(self._tokenizer)

            # Load model weights
            model_state_dict = load_file(os.path.join(model_path, "ema.safetensors"), device="cpu")
            msg = self._model.load_state_dict(model_state_dict, strict=False)
            print_rank0(f"Model loading: {msg}")
            del model_state_dict

            # Move to device
            self._model = self._model.to(device).eval()
            self._vae_model = self._vae_model.to(device).eval()

            # Create transforms
            vae_transform = ImageTransform(1024, 512, 16)
            vit_transform = ImageTransform(980, 224, 14)

            # Create inferencer
            self._inferencer = InterleaveInferencer(
                model=self._model,
                vae_model=self._vae_model,
                tokenizer=self._tokenizer,
                vae_transform=vae_transform,
                vit_transform=vit_transform,
                new_token_ids=new_token_ids,
            )

            self._initialized = True
            print_rank0("BAGEL model loaded")

        except ImportError as e:
            raise ImportError(
                f"Failed to import BAGEL module: {e}. "
                "Make sure the bagel submodule is properly initialized."
            )

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using BAGEL.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Number of inference steps (default: 50)
            guidance_scale: Not directly used (use cfg_text_scale, cfg_img_scale)
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 50

        # Normalize images
        if isinstance(images, Image.Image):
            images = [images]

        # Convert images to RGB
        images = [img.convert("RGB") for img in images]

        # Build input list for interleave_inference
        # Format: [image1, image2, ..., text_instruction]
        input_list = images + [instruction]

        # Get env settings
        think = os.environ.get("DG_BAGEL_THINK", "false").lower() in ("true", "1", "yes")

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            import random
            random.seed(seed)

        # Get image shape from first image
        image_shapes = (images[0].height, images[0].width) if images else (1024, 1024)

        # Run inference
        output_list = self._inferencer.interleave_inference(
            input_lists=input_list,
            think=think,
            understanding_output=False,
            cfg_text_scale=self._cfg_text_scale,
            cfg_img_scale=self._cfg_img_scale,
            cfg_interval=list(self._cfg_interval),
            timestep_shift=self._timestep_shift,
            num_timesteps=num_inference_steps,
            image_shapes=image_shapes,
        )

        # Extract images from output
        result_images = []
        for item in output_list:
            if isinstance(item, Image.Image):
                result_images.append(item)

        return result_images if result_images else [output_list[-1]]

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._inferencer = None
        if self._model is not None:
            del self._model
            self._model = None
        if self._vae_model is not None:
            del self._vae_model
            self._vae_model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
