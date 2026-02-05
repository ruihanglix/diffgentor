# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""HunyuanImage-3.0-Instruct backend using Tencent's model."""

import os
from typing import List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.env import HunyuanImage3Env
from diffgentor.utils.logging import print_rank0


class HunyuanImage3Backend(BaseEditingBackend):
    """HunyuanImage-3.0-Instruct backend for image editing.

    Supports HunyuanImage-3.0-Instruct and HunyuanImage-3.0-Instruct-Distil models.
    This is a native multimodal model that unifies multimodal understanding and
    generation within an autoregressive framework.

    Features:
    - Image editing (TI2I - Text-Image-to-Image)
    - Multi-image fusion (up to 3 images)
    - CoT reasoning (think_recaption mode)
    - Prompt self-rewrite

    Requires transformers with trust_remote_code=True.
    Model weights can be downloaded with:
        hf download tencent/HunyuanImage-3.0-Instruct-Distil --local-dir ./HunyuanImage-3-Instruct-Distil

    Model-specific parameters via environment variables:
        DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL: Number of GPUs per model instance (default: 0, use all visible)
        DG_HUNYUAN_IMAGE_3_ATTN_IMPL: Attention implementation (default: sdpa)
        DG_HUNYUAN_IMAGE_3_MOE_IMPL: MoE implementation, "eager" or "flashinfer" (default: eager)
        DG_HUNYUAN_IMAGE_3_MOE_DROP_TOKENS: Enable MoE token dropping (default: true)
        DG_HUNYUAN_IMAGE_3_USE_SYSTEM_PROMPT: System prompt type (default: en_unified)
            Options: None, dynamic, en_vanilla, en_recaption, en_think_recaption, en_unified, custom
        DG_HUNYUAN_IMAGE_3_BOT_TASK: Task type (default: think_recaption)
            Options: image (direct), auto (text), recaption (rewrite->image), think_recaption (think->rewrite->image)
        DG_HUNYUAN_IMAGE_3_INFER_ALIGN_IMAGE_SIZE: Align output size to input size (default: true)
        DG_HUNYUAN_IMAGE_3_MAX_NEW_TOKENS: Maximum new tokens for text generation (default: 2048)
        DG_HUNYUAN_IMAGE_3_USE_TAYLOR_CACHE: Use Taylor Cache when sampling (default: false)

    Multi-GPU Usage:
        The Launcher automatically handles GPU assignment based on DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL.
        Model is distributed across visible GPUs via device_map="auto".

        Examples:
            # Single model on all 8 GPUs
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 diffgentor edit --backend hunyuan_image_3 ...

            # 2 model instances, each on 4 GPUs (8 GPUs total)
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL=4 \\
                diffgentor edit --backend hunyuan_image_3 --model_type hunyuan_image_3 ...
            # Instance 0: GPU 0,1,2,3 | Instance 1: GPU 4,5,6,7
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize HunyuanImage-3.0 backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._model = None
        self._env = HunyuanImage3Env.load()
        # Store the last CoT/recaption text for retrieval by workers
        self.last_cot_text: Optional[str] = None

    def load_model(self, **kwargs) -> None:
        """Load HunyuanImage-3.0 model.

        Model-specific parameters are read from environment variables.
        Model automatically distributed across visible GPUs via device_map="auto".
        GPU assignment is handled by the Launcher based on DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL.

        Args:
            **kwargs: Additional arguments
        """
        # Reload env config to get latest values
        self._env = HunyuanImage3Env.load()

        # Log GPU configuration
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        num_gpus = torch.cuda.device_count()
        print_rank0(f"[HunyuanImage3] CUDA_VISIBLE_DEVICES: {visible_gpus}")
        print_rank0(f"[HunyuanImage3] Available CUDA devices: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print_rank0(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

        print_rank0(f"[HunyuanImage3] Configuration:")
        print_rank0(f"  attn_impl: {self._env.attn_impl}")
        print_rank0(f"  moe_impl: {self._env.moe_impl}")
        print_rank0(f"  moe_drop_tokens: {self._env.moe_drop_tokens}")
        print_rank0(f"  use_system_prompt: {self._env.use_system_prompt}")
        print_rank0(f"  bot_task: {self._env.bot_task}")
        print_rank0(f"  infer_align_image_size: {self._env.infer_align_image_size}")
        print_rank0(f"  max_new_tokens: {self._env.max_new_tokens}")
        print_rank0(f"  use_taylor_cache: {self._env.use_taylor_cache}")

        try:
            from transformers import AutoModelForCausalLM

            print_rank0(f"Loading HunyuanImage-3.0 from: {self.model_name}")

            # Build model kwargs
            model_kwargs = dict(
                attn_implementation=self._env.attn_impl,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
                moe_impl=self._env.moe_impl,
                moe_drop_tokens=self._env.moe_drop_tokens,
            )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            # Load tokenizer
            self._model.load_tokenizer(self.model_name)

            self._initialized = True
            print_rank0("HunyuanImage-3.0 model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers: {e}. "
                "Make sure transformers is installed: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HunyuanImage-3.0 model: {e}. "
                "Make sure the model path is correct and you have sufficient GPU memory."
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
        """Edit images using HunyuanImage-3.0.

        Supports single image editing and multi-image fusion (up to 3 images).

        Args:
            images: Input image(s) to edit. Can be a single Image or list of up to 3 Images.
            instruction: Editing instruction in natural language.
            num_inference_steps: Number of diffusion inference steps (default: 8 for distilled model).
            guidance_scale: Not directly used (model handles internally).
            seed: Random seed for reproducibility.
            **kwargs: Additional arguments passed to generate_image.

        Returns:
            List of edited PIL Images.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize images to list
        if isinstance(images, Image.Image):
            images = [images]

        # Convert to RGB, max 3 images supported
        imgs_input = [img.convert("RGB") for img in images[:3]]

        if not imgs_input:
            raise ValueError("No input images provided")

        # Set defaults for distilled model
        if num_inference_steps is None:
            num_inference_steps = 8  # Distilled version recommended

        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            import random
            random.seed(seed)

        print_rank0(f"[HunyuanImage3] Editing with {len(imgs_input)} input image(s)")
        print_rank0(f"[HunyuanImage3] Instruction: {instruction[:100]}...")
        print_rank0(f"[HunyuanImage3] Inference steps: {num_inference_steps}")

        try:
            # Generate edited image
            cot_text, samples = self._model.generate_image(
                prompt=instruction,
                image=imgs_input,
                seed=seed if seed is not None else 42,
                image_size="auto",
                use_system_prompt=self._env.use_system_prompt,
                bot_task=self._env.bot_task,
                infer_align_image_size=self._env.infer_align_image_size,
                diff_infer_steps=num_inference_steps,
                max_new_tokens=self._env.max_new_tokens,
                use_taylor_cache=self._env.use_taylor_cache,
                verbose=0,
            )

            # Store CoT/recaption text for retrieval by workers
            self.last_cot_text = cot_text if cot_text else None

            # Log CoT reasoning if available
            if cot_text:
                print_rank0(f"[HunyuanImage3] CoT reasoning: {cot_text[:200]}...")

            # Ensure output is a list
            if samples is None:
                print_rank0("[HunyuanImage3] Warning: No output images generated")
                return []

            result = samples if isinstance(samples, list) else [samples]
            print_rank0(f"[HunyuanImage3] Generated {len(result)} output image(s)")

            return result

        except Exception as e:
            print_rank0(f"[HunyuanImage3] Edit failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()
