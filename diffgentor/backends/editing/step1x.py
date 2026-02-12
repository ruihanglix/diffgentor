# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Step1X-Edit backend."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.env import Step1XEnv


class Step1XBackend(BaseEditingBackend):
    """Step1X-Edit backend.

    Requires the step1x_edit submodule to be initialized:
        git submodule update --init diffgentor/models/third_party/step1x_edit

    Model-specific parameters via environment variables:
        DG_STEP1X_VERSION: Model version (default: v1.1)
        DG_STEP1X_SIZE_LEVEL: Size level for image processing (default: 512)
        DG_STEP1X_OFFLOAD: Enable CPU offload (default: false)
        DG_STEP1X_QUANTIZED: Use fp8 quantization (default: false)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize Step1X backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._generator = None
        self._size_level = Step1XEnv.size_level()
        self._version = Step1XEnv.version()

    def load_model(self, **kwargs) -> None:
        """Load Step1X-Edit model.

        Model-specific parameters are read from environment variables:
            DG_STEP1X_VERSION: Model version (v1.0 or v1.1)
            DG_STEP1X_SIZE_LEVEL: Size level for image processing
            DG_STEP1X_OFFLOAD: Enable CPU offload
            DG_STEP1X_QUANTIZED: Use fp8 quantization

        Args:
            **kwargs: Additional arguments
        """
        # Read from environment variables
        self._size_level = Step1XEnv.size_level()
        self._version = Step1XEnv.version()
        offload = os.environ.get("DG_STEP1X_OFFLOAD", "false").lower() in ("true", "1", "yes")
        quantized = os.environ.get("DG_STEP1X_QUANTIZED", "false").lower() in ("true", "1", "yes")

        # Add third-party path
        third_party_path = Path(__file__).parent.parent.parent / "models" / "third_party" / "step1x_edit"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
        else:
            raise ImportError(
                f"Step1X-Edit vendored code not found at {third_party_path}. "
                "Please reinstall diffgentor: pip install diffgentor"
            )

        try:
            from inference import ImageGenerator

            print_rank0(f"Loading Step1X-Edit from: {self.model_name}")
            print_rank0(f"Version: {self._version}, Size level: {self._size_level}")
            print_rank0(f"Offload: {offload}, Quantized: {quantized}")

            # Determine checkpoint name based on version
            if self._version == "v1.0":
                ckpt_name = "step1x-edit-i1258.safetensors"
            else:
                ckpt_name = "step1x-edit-v1p1-official.safetensors"

            # Initialize generator
            self._generator = ImageGenerator(
                ae_path=os.path.join(self.model_name, "vae.safetensors"),
                dit_path=os.path.join(self.model_name, ckpt_name),
                qwen2vl_model_path=os.path.join(self.model_name, "Qwen2.5-VL-7B-Instruct"),
                max_length=640,
                quantized=quantized,
                offload=offload,
                mode="flash",
                version=self._version,
            )

            self._initialized = True
            print_rank0(f"Step1X-Edit model loaded (version={self._version})")

        except ImportError as e:
            raise ImportError(
                f"Failed to import Step1X-Edit module: {e}. "
                "Make sure the step1x_edit submodule is properly initialized."
            )

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using Step1X-Edit.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Number of inference steps (default: 28)
            guidance_scale: Guidance scale (default: 6.0)
            negative_prompt: Negative prompt
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 28
        if guidance_scale is None:
            guidance_scale = 6.0

        # Normalize images
        if isinstance(images, Image.Image):
            images = [images]

        # Use first image
        input_image = images[0] if images else None
        if input_image is None:
            raise ValueError("No input image provided")

        # Convert to RGB
        input_image = input_image.convert("RGB")

        # Use seed or generate random one
        if seed is None:
            seed = torch.Generator(device="cpu").seed()

        # Run inference using generate_image method
        result = self._generator.generate_image(
            prompt=instruction,
            negative_prompt=negative_prompt,
            ref_images=input_image,
            num_steps=num_inference_steps,
            cfg_guidance=guidance_scale,
            seed=seed,
            num_samples=1,
            show_progress=True,
            size_level=self._size_level,
        )

        # Result is already a list of PIL Images
        if isinstance(result, list):
            return result
        else:
            return [result]

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._generator is not None:
            # Clean up generator resources
            if hasattr(self._generator, "ae"):
                del self._generator.ae
            if hasattr(self._generator, "dit"):
                del self._generator.dit
            if hasattr(self._generator, "llm_encoder"):
                del self._generator.llm_encoder
        self._generator = None
        torch.cuda.empty_cache()
