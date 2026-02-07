# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen backend for image editing.

DeepGen is a unified visual generation model based on AR (Qwen2.5-VL) + Diffusion (SD3.5),
supporting both text-to-image generation and image editing.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.env import DeepGenEnv
from diffgentor.utils.logging import print_rank0


def resize_for_deepgen(image: Image.Image, max_size: int = 512, unit_size: int = 32) -> Tuple[Image.Image, int, int]:
    """Resize image for DeepGen model.

    Ensures the image dimensions are multiples of unit_size (32) for proper
    processing by the vision transformer. After 28/32 scaling in the model,
    the dimensions will be multiples of 28, which is divisible by patch_size (14)
    and spatial_merge_size (2).

    Args:
        image: Input PIL image
        max_size: Maximum size for the longer edge (should be multiple of 32)
        unit_size: Unit size for alignment (default: 32)

    Returns:
        Tuple of (resized_image, target_width, target_height)
    """
    w, h = image.size

    if w >= h and w >= max_size:
        target_w = max_size
        target_h = h * (target_w / w)
        target_h = math.ceil(target_h / unit_size) * unit_size
    elif h >= w and h >= max_size:
        target_h = max_size
        target_w = w * (target_h / h)
        target_w = math.ceil(target_w / unit_size) * unit_size
    else:
        target_h = math.ceil(h / unit_size) * unit_size
        target_w = math.ceil(w / unit_size) * unit_size

    target_w = int(target_w)
    target_h = int(target_h)

    resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return resized, target_w, target_h


class DeepGenEditingBackend(BaseEditingBackend):
    """DeepGen backend for image editing.

    Uses AR model (Qwen2.5-VL) for vision-language understanding and
    Diffusion model (SD3.5) for image generation.

    Configuration is loaded from a Python config file. Model paths can be
    overridden via environment variables.

    CLI parameters:
        --model_name: Path to model checkpoint (.safetensors or .pt file)
        --guidance_scale: CFG guidance scale (default: 4.0)
        --num_inference_steps: Number of inference steps
        --height: Output image height
        --width: Output image width
        --negative_prompt: Negative prompt for CFG

    Environment variables:
        DG_DEEPGEN_CONFIG: Config file name (default: "deepgen")
        DG_DEEPGEN_DIFFUSION_MODEL_PATH: Override diffusion model path (SD3.5)
        DG_DEEPGEN_AR_MODEL_PATH: Override AR model path (Qwen2.5-VL)
        DG_DEEPGEN_MAX_LENGTH: Maximum sequence length (default: 1024)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize DeepGen editing backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._model = None
        self._env = DeepGenEnv.load()

    def load_model(self, **kwargs) -> None:
        """Load DeepGen model.

        The model configuration is loaded from the config file specified by
        DG_DEEPGEN_CONFIG (default: "deepgen"). Model paths can be overridden
        via environment variables.

        GPU assignment is handled by the Launcher based on DG_DEEPGEN_GPUS_PER_MODEL.
        Each instance runs on its assigned GPU(s) via CUDA_VISIBLE_DEVICES.

        Args:
            **kwargs: Additional arguments
        """
        import os

        import torch

        from diffgentor.models.deepgen import DeepGenModel
        from diffgentor.models.deepgen.model import DeepGenModelConfig

        # Log GPU configuration
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        num_gpus = torch.cuda.device_count()
        print_rank0(f"[DeepGen] CUDA_VISIBLE_DEVICES: {visible_gpus}")
        print_rank0(f"[DeepGen] Available CUDA devices: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print_rank0(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

        # Create model config from environment
        config = DeepGenModelConfig.from_env(self._env)

        print_rank0(f"Loading DeepGen model:")
        print_rank0(f"  Config: {self._env.config_name}")
        print_rank0(f"  Diffusion model: {config.diffusion_model_path}")
        print_rank0(f"  AR model: {config.ar_model_path}")
        print_rank0(f"  Device: {self.device}")
        if self.model_name:
            print_rank0(f"  Checkpoint: {self.model_name}")

        # Load model (use model_name as checkpoint path)
        self._model = DeepGenModel(
            config=config,
            pretrained_pth=self.model_name,
            use_activation_checkpointing=self._env.use_activation_checkpointing,
        )

        # Move to device
        self._model = self._model.to(self.device)
        self._model = self._model.to(self._model.dtype)
        self._model.eval()

        self._initialized = True
        print_rank0("DeepGen model loaded")

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using DeepGen.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Number of inference steps (default: 50)
            guidance_scale: Guidance scale (default: 4.0)
            negative_prompt: Negative prompt for CFG (default: "")
            seed: Random seed
            height: Output height (default: aligned to 32)
            width: Output width (default: aligned to 32)
            **kwargs: Additional arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize images
        if isinstance(images, Image.Image):
            images = [images]

        # Convert images to RGB
        images = [img.convert("RGB") for img in images]

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 50
        if guidance_scale is None:
            guidance_scale = 4.0
        if negative_prompt is None:
            negative_prompt = ""

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._model.device).manual_seed(seed)

        # Convert PIL images to tensors with proper resizing
        # Images must be resized to dimensions that are multiples of 32
        pixel_values = []
        for img in images:
            # Resize image to proper dimensions (multiples of 32)
            img_resized, target_w, target_h = resize_for_deepgen(img, max_size=512, unit_size=32)
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img_resized)).to(
                dtype=self._model.dtype, device=self._model.device
            )
            img_tensor = rearrange(img_tensor, "h w c -> c h w")
            # Normalize to [-1, 1]
            img_tensor = 2 * (img_tensor / 255) - 1
            pixel_values.append(img_tensor)

        # Use the first image's resized dimensions for output if not specified
        if height is None or width is None:
            _, out_w, out_h = resize_for_deepgen(images[0], max_size=512, unit_size=32)
            if height is None:
                height = out_h
            if width is None:
                width = out_w

        # Run inference
        output = self._model.generate_edit(
            images=pixel_values,
            instruction=instruction,
            cfg_instruction=negative_prompt,
            cfg_scale=guidance_scale,
            num_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            progress_bar=False,
        )

        # Convert output tensor to PIL images
        # Output is in range [-1, 1]
        output = torch.clamp(127.5 * output + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()

        result_images = []
        for i in range(output.shape[0]):
            img_array = rearrange(output[i], "c h w -> h w c")
            result_images.append(Image.fromarray(img_array))

        return result_images

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()
