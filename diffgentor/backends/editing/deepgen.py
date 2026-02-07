# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen backend for unified image generation and editing.

DeepGen combines Qwen2.5-VL and SD3.5 for both text-to-image generation
and image editing tasks.
"""

import os
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.env import DeepGenEnv


class DeepGenBackend(BaseEditingBackend):
    """DeepGen backend supporting both T2I and image editing.

    DeepGen is a unified image generation model combining Qwen2.5-VL and SD3.5.
    It supports both text-to-image generation (when no input image is provided)
    and image editing (when input images are provided).

    Supports multi-GPU tensor parallelism for large model inference via device_map="auto".

    CLI arguments (common parameters):
        --guidance_scale: Classifier-free guidance scale (default: 4.0)
        --num_inference_steps: Number of inference steps (default: 50)

    Model-specific parameters via environment variables:
        DG_DEEPGEN_DIFFUSION_PATH: Path to diffusion model (transformer, vae, scheduler) - REQUIRED
        DG_DEEPGEN_QWEN_PATH: Path to Qwen2.5-VL model - REQUIRED
        DG_DEEPGEN_CONFIG: Config name from configs/ folder (default: deepgen_v1)
        DG_DEEPGEN_GPUS_PER_MODEL: Number of GPUs per model instance (default: 0, use all visible)
        DG_DEEPGEN_CFG_PROMPT: CFG prompt for unconditional generation (default: "")
        DG_DEEPGEN_DEBUG: Debug level for checkpoint loading (default: 0)

    Config-based architecture:
        Model-specific parameters (connector config, num_queries, etc.) are defined in
        config files located at diffgentor/models/deepgen_v1/configs/.

    Multi-GPU Usage:
        The Launcher automatically handles GPU assignment based on DG_DEEPGEN_GPUS_PER_MODEL.
        Model is distributed across visible GPUs via device_map="auto".

        Examples:
            # Single model on 2 GPUs
            CUDA_VISIBLE_DEVICES=0,1 diffgentor edit --backend deepgen ...

            # 4 model instances, each on 2 GPUs (8 GPUs total)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_DEEPGEN_GPUS_PER_MODEL=2 \\
                diffgentor edit --backend deepgen --model_type deepgen ...
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize DeepGen backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._model = None
        # Load env config
        self._env = DeepGenEnv.load()

    def load_model(self, **kwargs) -> None:
        """Load DeepGen model.

        Model-specific parameters are read from environment variables.
        Model automatically distributed across visible GPUs via device_map="auto".
        GPU assignment is handled by the Launcher based on DG_DEEPGEN_GPUS_PER_MODEL.

        Args:
            **kwargs: Additional arguments
        """
        # Reload env config to get latest values
        self._env = DeepGenEnv.load()

        # Validate required paths
        if not self._env.diffusion_path:
            raise ValueError(
                "DG_DEEPGEN_DIFFUSION_PATH environment variable is required. "
                "Set it to the path of your diffusion model (transformer, vae, scheduler)."
            )
        if not self._env.qwen_path:
            raise ValueError(
                "DG_DEEPGEN_QWEN_PATH environment variable is required. "
                "Set it to the path of your Qwen2.5-VL model."
            )

        # Log GPU configuration
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        num_gpus = torch.cuda.device_count()
        print_rank0(f"[DeepGen] CUDA_VISIBLE_DEVICES: {visible_gpus}")
        print_rank0(f"[DeepGen] Available CUDA devices: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print_rank0(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

        # Import model class
        from diffgentor.models.deepgen_v1 import DeepGenModel

        print_rank0(f"Loading DeepGen model...")
        print_rank0(f"  Config: {self._env.config}")
        print_rank0(f"  Diffusion path: {self._env.diffusion_path}")
        print_rank0(f"  Qwen path: {self._env.qwen_path}")
        print_rank0(f"  Checkpoint: {self.model_name}")
        print_rank0(f"  Attention impl: {self._env.attn_impl}")

        # Determine dtype
        torch_dtype = torch.bfloat16
        if self.optimization_config and self.optimization_config.torch_dtype:
            dtype_str = self.optimization_config.torch_dtype
            if dtype_str == "float16":
                torch_dtype = torch.float16
            elif dtype_str == "float32":
                torch_dtype = torch.float32

        # Initialize model with config from configs/ folder
        self._model = DeepGenModel(
            diffusion_path=self._env.diffusion_path,
            qwen_path=self._env.qwen_path,
            config_name=self._env.config,
            torch_dtype=torch_dtype,
            attn_impl=self._env.attn_impl,
        )

        # Load checkpoint if provided
        if self.model_name and os.path.exists(self.model_name):
            from diffgentor.utils.logging import get_log_dir
            from diffgentor.utils.distributed import is_main_process

            # Only rank 0 writes debug report
            debug_level = self._env.debug_level if is_main_process() else 0
            debug_log_path = self._model.load_checkpoint(
                self.model_name,
                debug_level=debug_level,
                debug_log_dir=get_log_dir(),
            )

            if debug_level > 0 and debug_log_path:
                print_rank0(f"[DeepGen] Checkpoint debug report written to: {debug_log_path}")

        # Move non-LMM components to the correct device
        # LMM uses device_map="auto", but other components need explicit device placement
        device = self._model.device  # Get device from LMM
        print_rank0(f"[DeepGen] Moving model components to device: {device}")

        # Move connector and projectors
        self._model.connector = self._model.connector.to(device=device, dtype=torch_dtype)
        self._model.projector_1 = self._model.projector_1.to(device=device, dtype=torch_dtype)
        self._model.projector_2 = self._model.projector_2.to(device=device, dtype=torch_dtype)
        self._model.projector_3 = self._model.projector_3.to(device=device, dtype=torch_dtype)

        # Move transformer and VAE to cuda:0 (they don't use device_map)
        self._model.transformer = self._model.transformer.to(device="cuda:0", dtype=torch_dtype)
        self._model.vae = self._model.vae.to(device="cuda:0", dtype=torch_dtype)

        # Move meta_queries parameter
        self._model.meta_queries.data = self._model.meta_queries.data.to(device=device, dtype=torch_dtype)

        # Move buffers
        self._model.vit_mean = self._model.vit_mean.to(device=device)
        self._model.vit_std = self._model.vit_std.to(device=device)

        self._model.eval()

        self._initialized = True
        print_rank0("DeepGen model loaded successfully")

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using DeepGen.

        If no images are provided, this performs text-to-image generation instead.

        Args:
            images: Input image(s) to edit (can be None for T2I)
            instruction: Editing instruction or generation prompt
            num_inference_steps: Number of inference steps (default: 50)
            guidance_scale: Guidance scale (default: from env or 4.0)
            seed: Random seed
            height: Output image height (default: from input or 1024)
            width: Output image width (default: from input or 1024)
            **kwargs: Additional arguments

        Returns:
            List of edited/generated PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 50
        if guidance_scale is None:
            guidance_scale = 4.0  # Default CFG scale for DeepGen

        # Normalize images
        if images is None:
            images = []
        elif isinstance(images, Image.Image):
            images = [images]

        # Determine output size
        if images:
            # Use first image size as reference
            input_image = images[0].convert("RGB")
            if height is None:
                height = input_image.height
            if width is None:
                width = input_image.width
        else:
            # Default size for T2I
            if height is None:
                height = 1024
            if width is None:
                width = 1024

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._model.device).manual_seed(seed)

        # Prepare source images if provided
        pixel_values_src = None
        if images:
            pixel_values_src = []
            for img in images:
                img = img.convert("RGB")
                # Resize if needed
                from diffgentor.models.deepgen_v1 import resize_image
                img = resize_image(img, max(height, width), 32)

                # Convert to tensor
                img_tensor = torch.from_numpy(np.array(img)).to(
                    dtype=self._model.dtype, device=self._model.device
                )
                img_tensor = rearrange(img_tensor, 'h w c -> c h w')
                img_tensor = 2 * (img_tensor / 255) - 1  # Normalize to [-1, 1]
                pixel_values_src.append(img_tensor)

            # Wrap in list (batch of 1, with multiple reference images)
            pixel_values_src = [pixel_values_src]

        # Get CFG prompt
        cfg_prompt = self._env.cfg_prompt or ""

        # Generate
        with torch.no_grad():
            samples = self._model.generate(
                prompt=[instruction],
                cfg_prompt=[cfg_prompt],
                pixel_values_src=pixel_values_src,
                cfg_scale=guidance_scale,
                num_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                progress_bar=True,
            )

        # Convert to PIL images
        # samples shape: (batch, channels, height, width) in [-1, 1]
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
        samples = samples.numpy()

        result_images = []
        for sample in samples:
            # Convert from CHW to HWC
            sample = np.transpose(sample, (1, 2, 0))
            result_images.append(Image.fromarray(sample))

        return result_images

    def generate(
        self,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        height: int = 1024,
        width: int = 1024,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from text prompt (T2I mode).

        This is a convenience method that calls edit() without input images.

        Args:
            prompt: Text prompt for generation
            num_inference_steps: Number of inference steps (default: 50)
            guidance_scale: Guidance scale (default: from env or 4.0)
            seed: Random seed
            height: Output image height (default: 1024)
            width: Output image width (default: 1024)
            **kwargs: Additional arguments

        Returns:
            List of generated PIL Images
        """
        return self.edit(
            images=None,
            instruction=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            height=height,
            width=width,
            **kwargs,
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._model is not None:
            # Clean up model components
            if hasattr(self._model, 'lmm'):
                del self._model.lmm
            if hasattr(self._model, 'transformer'):
                del self._model.transformer
            if hasattr(self._model, 'vae'):
                del self._model.vae
            del self._model
            self._model = None
        torch.cuda.empty_cache()
