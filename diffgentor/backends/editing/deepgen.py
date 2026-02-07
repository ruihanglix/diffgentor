# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen backend for image editing.

DeepGen is a unified visual generation model based on Qwen2.5-VL + SD3.5,
supporting both text-to-image generation and image editing.
"""

from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.env import DeepGenEnv
from diffgentor.utils.logging import print_rank0


class DeepGenEditingBackend(BaseEditingBackend):
    """DeepGen backend for image editing.

    Uses Qwen2.5-VL for vision-language understanding and SD3.5 for image generation.

    Model-specific parameters via environment variables:
        DG_DEEPGEN_SD3_MODEL_PATH: Path to SD3.5 model (required if not using model_name)
        DG_DEEPGEN_QWEN_MODEL_PATH: Path to Qwen2.5-VL model (required if not using model_name)
        DG_DEEPGEN_CHECKPOINT: Path to model checkpoint (optional)
        DG_DEEPGEN_CFG_SCALE: CFG guidance scale (default: 4.0)
        DG_DEEPGEN_CFG_PROMPT: CFG negative prompt (default: "")
        DG_DEEPGEN_HEIGHT: Output image height (default: 512)
        DG_DEEPGEN_WIDTH: Output image width (default: 512)
        DG_DEEPGEN_NUM_STEPS: Number of inference steps (default: 50)
        DG_DEEPGEN_NUM_QUERIES: Number of query tokens (default: 128)
        DG_DEEPGEN_CONNECTOR_HIDDEN_SIZE: Connector hidden size (default: 2048)
        DG_DEEPGEN_CONNECTOR_NUM_LAYERS: Number of connector layers (default: 6)
        DG_DEEPGEN_VIT_INPUT_SIZE: ViT input size (default: 448)
        DG_DEEPGEN_LORA_RANK: LoRA rank (default: 64)
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

        The model requires both SD3.5 and Qwen2.5-VL paths. These can be specified via:
        1. model_name in backend_config (format: "sd3_path,qwen_path")
        2. Environment variables DG_DEEPGEN_SD3_MODEL_PATH and DG_DEEPGEN_QWEN_MODEL_PATH

        Args:
            **kwargs: Additional arguments
        """
        from diffgentor.models.deepgen import DeepGenModel
        from diffgentor.models.deepgen.model import DeepGenModelConfig

        # Get model paths
        sd3_path = self._env.sd3_model_path
        qwen_path = self._env.qwen_model_path

        # If not set via env, try to parse from model_name
        if not sd3_path or not qwen_path:
            if "," in self.model_name:
                parts = self.model_name.split(",")
                if len(parts) >= 2:
                    sd3_path = sd3_path or parts[0].strip()
                    qwen_path = qwen_path or parts[1].strip()
            else:
                # Assume model_name is the base path containing both models
                sd3_path = sd3_path or self.model_name
                qwen_path = qwen_path or self.model_name

        if not sd3_path:
            raise ValueError(
                "SD3 model path not specified. Set DG_DEEPGEN_SD3_MODEL_PATH or use "
                "--model_name 'sd3_path,qwen_path'"
            )
        if not qwen_path:
            raise ValueError(
                "Qwen model path not specified. Set DG_DEEPGEN_QWEN_MODEL_PATH or use "
                "--model_name 'sd3_path,qwen_path'"
            )

        print_rank0(f"Loading DeepGen model:")
        print_rank0(f"  SD3 path: {sd3_path}")
        print_rank0(f"  Qwen path: {qwen_path}")

        # Create model config
        config = DeepGenModelConfig(
            sd3_model_path=sd3_path,
            qwen_model_path=qwen_path,
            num_queries=self._env.num_queries,
            connector_hidden_size=self._env.connector_hidden_size,
            connector_num_layers=self._env.connector_num_layers,
            vit_input_size=self._env.vit_input_size,
            lora_rank=self._env.lora_rank,
        )

        # Load model
        checkpoint = self._env.checkpoint_path()
        self._model = DeepGenModel(
            config=config,
            pretrained_pth=checkpoint,
            use_activation_checkpointing=False,
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
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using DeepGen.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Number of inference steps (default from env)
            guidance_scale: Guidance scale (default from env)
            seed: Random seed
            height: Output height (default from env or input size)
            width: Output width (default from env or input size)
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
            num_inference_steps = self._env.default_num_steps()
        if guidance_scale is None:
            guidance_scale = self._env.cfg_scale_default()

        # Determine output size
        if height is None:
            height = self._env.default_height() or images[0].height
        if width is None:
            width = self._env.default_width() or images[0].width

        # Get CFG prompt
        cfg_prompt = self._env.cfg_prompt_default()

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._model.device).manual_seed(seed)

        # Convert PIL images to tensors
        pixel_values = []
        for img in images:
            # Resize to target size
            img_resized = img.resize((width, height))
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img_resized)).to(
                dtype=self._model.dtype, device=self._model.device
            )
            img_tensor = rearrange(img_tensor, "h w c -> c h w")
            # Normalize to [-1, 1]
            img_tensor = 2 * (img_tensor / 255) - 1
            pixel_values.append(img_tensor)

        # Run inference
        output = self._model.generate_edit(
            images=pixel_values,
            instruction=instruction,
            cfg_instruction=cfg_prompt,
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
