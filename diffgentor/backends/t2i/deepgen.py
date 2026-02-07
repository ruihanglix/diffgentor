# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen backend for text-to-image generation.

DeepGen is a unified visual generation model based on AR (Qwen2.5-VL) + Diffusion (SD3.5),
supporting both text-to-image generation and image editing.
"""

from typing import List, Optional, Union

import torch
from einops import rearrange
from PIL import Image

from diffgentor.backends.base import BaseBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.env import DeepGenEnv
from diffgentor.utils.logging import print_rank0


class DeepGenT2IBackend(BaseBackend):
    """DeepGen backend for text-to-image generation.

    Uses AR model (Qwen2.5-VL) for language understanding and
    Diffusion model (SD3.5) for image generation.

    Configuration is loaded from a Python config file. Model paths can be
    overridden via environment variables.

    CLI parameters:
        --model_name: Path to model checkpoint (.safetensors or .pt file)
        --guidance_scale: CFG guidance scale (default: 4.0)
        --num_inference_steps: Number of inference steps (default: 50)
        --height: Output image height (default: 512)
        --width: Output image width (default: 512)
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
        """Initialize DeepGen T2I backend.

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

        Args:
            **kwargs: Additional arguments
        """
        from diffgentor.models.deepgen import DeepGenModel
        from diffgentor.models.deepgen.model import DeepGenModelConfig

        # Create model config from environment
        config = DeepGenModelConfig.from_env(self._env)

        print_rank0(f"Loading DeepGen model:")
        print_rank0(f"  Config: {self._env.config_name}")
        print_rank0(f"  Diffusion model: {config.diffusion_model_path}")
        print_rank0(f"  AR model: {config.ar_model_path}")
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

    def generate(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            num_inference_steps: Number of inference steps (default: 50)
            guidance_scale: Guidance scale (default: 4.0)
            negative_prompt: Negative prompt(s) for CFG (default: "")
            seed: Random seed
            height: Output height (default: 512)
            width: Output width (default: 512)
            **kwargs: Additional arguments

        Returns:
            List of generated PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize prompts
        if isinstance(prompt, str):
            prompt = [prompt]

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 50
        if guidance_scale is None:
            guidance_scale = 4.0
        if height is None:
            height = 512
        if width is None:
            width = 512

        # Handle negative prompt
        if negative_prompt is None:
            cfg_prompt = [""] * len(prompt)
        elif isinstance(negative_prompt, str):
            cfg_prompt = [negative_prompt] * len(prompt)
        else:
            cfg_prompt = negative_prompt

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._model.device).manual_seed(seed)

        # Run inference
        output = self._model.generate_t2i(
            prompt=prompt,
            cfg_prompt=cfg_prompt,
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
