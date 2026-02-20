# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Unified diffusers editing backend using strategy pattern."""

from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.backends.editing.strategies import (
    ModelStrategy,
    get_model_strategy,
)
from diffgentor.backends.editing.strategies.registry import detect_editing_model_type
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.exceptions import EditingError, ModelLoadError, log_error
from diffgentor.utils.logging import print_rank0


class DiffusersEditingBackend(BaseEditingBackend):
    """Unified diffusers editing backend using strategy pattern.

    This backend supports multiple model types through pluggable strategies:
    - qwen: QwenImageEditPlusPipeline (multi-image)
    - qwen_singleimg: QwenImageEditPipeline (single-image)
    - flux2: Flux2Pipeline
    - flux2_klein: Flux2 Klein (4B/9B)
    - flux1_kontext: FluxKontextPipeline
    - longcat: LongCatImageEditPipeline
    - glm_image: GlmImagePipeline

    Each model type has a corresponding strategy that handles:
    - Pipeline class selection
    - Input image preparation
    - Default parameters
    - Batch inference support
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize diffusers editing backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._strategy: Optional[ModelStrategy] = None
        self._effective_model_type: Optional[str] = None

    @property
    def strategy(self) -> ModelStrategy:
        """Get current model strategy."""
        if self._strategy is None:
            self._strategy = get_model_strategy(None)
        return self._strategy

    def load_model(self, **kwargs) -> None:
        """Load diffusers editing pipeline.

        Args:
            **kwargs: Additional arguments for pipeline loading
        """
        from diffusers import DiffusionPipeline

        # Determine model type
        model_type = self.model_type
        if model_type is None:
            model_type = detect_editing_model_type(self.model_name)

        if model_type is None:
            print_rank0(f"Auto-detecting pipeline for: {self.model_name}")
            model_type = "auto"

        self._effective_model_type = model_type
        self._strategy = get_model_strategy(model_type)

        print_rank0(f"Using strategy for model type: {model_type}")

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.optimization_config.torch_dtype, torch.bfloat16)

        # Load pipeline
        load_kwargs = {"torch_dtype": torch_dtype, **kwargs}
        pipeline_class_name = self.strategy.pipeline_class_name

        if pipeline_class_name:
            try:
                pipeline_class = self._get_pipeline_class(pipeline_class_name)
                print_rank0(f"Loading {pipeline_class_name}")
                self.pipe = pipeline_class.from_pretrained(self.model_name, **load_kwargs)
            except (ImportError, AttributeError) as e:
                log_error(
                    ModelLoadError(f"Failed to load {pipeline_class_name}", cause=e),
                    context="load_model",
                    include_traceback=True,
                )
                print_rank0("Falling back to DiffusionPipeline auto-detection")
                self.pipe = DiffusionPipeline.from_pretrained(self.model_name, **load_kwargs)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(self.model_name, **load_kwargs)

        print_rank0(f"Loaded pipeline: {type(self.pipe).__name__}")

        # Move to device
        self._setup_device()

        # Apply optimizations
        self.apply_optimizations()

        self._initialized = True
        print_rank0("Editing pipeline initialization complete")

    def _setup_device(self) -> None:
        """Setup device and move pipeline."""
        import os

        from diffgentor.utils.distributed import get_local_rank, get_world_size, is_distributed

        device = self.backend_config.device
        if device == "cuda" and torch.cuda.is_available():
            env_world_size = int(os.environ.get("WORLD_SIZE", 1))
            env_local_rank = os.environ.get("LOCAL_RANK")

            if is_distributed() or env_world_size > 1 or env_local_rank is not None:
                local_rank = get_local_rank()
                device = f"cuda:{local_rank}"
                torch.cuda.set_device(local_rank)
                print_rank0(f"Distributed mode: device {device} (world_size={get_world_size()})")

            self.pipe = self.pipe.to(device)

    def _get_pipeline_class(self, class_name: str):
        """Get pipeline class by name."""
        import diffusers

        if hasattr(diffusers, class_name):
            return getattr(diffusers, class_name)
        raise AttributeError(f"Pipeline class {class_name} not found in diffusers")

    def apply_optimizations(self) -> None:
        """Apply optimizations to the pipeline."""
        if self.pipe is None:
            return

        from diffgentor.optimizations.manager import OptimizationManager

        manager = OptimizationManager(self.optimization_config)
        self.pipe = manager.apply_all(self.pipe)

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images based on instruction.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            true_cfg_scale: True CFG scale (for Qwen models)
            negative_prompt: Negative prompt
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        kwargs.pop("quality", None)

        # Prepare generator
        generator = None
        if seed is not None:
            device = self.pipe.device if hasattr(self.pipe, "device") else "cuda"
            generator = torch.Generator(device=device).manual_seed(seed)

        # Use strategy to build kwargs
        pipe_kwargs = self.strategy.build_pipeline_kwargs(
            images=images,
            instruction=instruction,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            **kwargs,
        )

        # Run editing
        output = self.pipe(**pipe_kwargs)

        # Extract images
        if hasattr(output, "images"):
            return output.images
        return output

    def batch_edit(
        self,
        batch_images: List[Union[Image.Image, List[Image.Image]]],
        batch_instructions: List[str],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: Optional[float] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit multiple image-instruction pairs.

        Args:
            batch_images: List of input image(s) for each edit request
            batch_instructions: List of editing instructions
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            true_cfg_scale: True CFG scale
            negative_prompt: Negative prompt(s)
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        if len(batch_images) != len(batch_instructions):
            raise ValueError("batch_images and batch_instructions must have same length")

        if len(batch_images) == 0:
            return []

        # Single item - use regular edit
        if len(batch_images) == 1:
            return self.edit(
                images=batch_images[0],
                instruction=batch_instructions[0],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                **kwargs,
            )

        # Check if batch is supported
        if not self.strategy.supports_batch:
            print_rank0("Pipeline does not support batch. Falling back to sequential.")
            return self._sequential_edit(
                batch_images,
                batch_instructions,
                num_inference_steps,
                guidance_scale,
                true_cfg_scale,
                negative_prompt,
                seed,
                **kwargs,
            )

        # True batch inference
        return self._batch_edit_impl(
            batch_images,
            batch_instructions,
            num_inference_steps,
            guidance_scale,
            true_cfg_scale,
            negative_prompt,
            seed,
            **kwargs,
        )

    def _sequential_edit(
        self,
        batch_images: List[Union[Image.Image, List[Image.Image]]],
        batch_instructions: List[str],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        true_cfg_scale: Optional[float],
        negative_prompt: Optional[Union[str, List[str]]],
        seed: Optional[int],
        **kwargs,
    ) -> List[Image.Image]:
        """Process edits sequentially."""
        all_images = []
        for img, instruction in zip(batch_images, batch_instructions):
            edited = self.edit(
                images=img,
                instruction=instruction,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
                seed=seed,
                **kwargs,
            )
            all_images.extend(edited)
        return all_images

    def _batch_edit_impl(
        self,
        batch_images: List[Union[Image.Image, List[Image.Image]]],
        batch_instructions: List[str],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
        true_cfg_scale: Optional[float],
        negative_prompt: Optional[Union[str, List[str]]],
        seed: Optional[int],
        **kwargs,
    ) -> List[Image.Image]:
        """Batch inference implementation."""
        config = self.strategy.config
        batch_size = self.backend_config.batch_size

        # Prepare generator
        generator = None
        if seed is not None:
            device = self.pipe.device if hasattr(self.pipe, "device") else "cuda"
            generator = torch.Generator(device=device).manual_seed(seed)

        # Normalize negative prompts
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompts = [negative_prompt] * len(batch_instructions)
            else:
                negative_prompts = list(negative_prompt)
        else:
            negative_prompts = None

        # Normalize images
        normalized_images = []
        for img in batch_images:
            if isinstance(img, Image.Image):
                normalized_images.append(img)
            else:
                normalized_images.append(img[0] if img else None)

        all_images = []
        num_batches = (len(batch_instructions) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(batch_instructions))

            batch_prompts = batch_instructions[start_idx:end_idx]
            batch_imgs = normalized_images[start_idx:end_idx]

            pipe_kwargs = {
                "prompt": batch_prompts,
                "image": batch_imgs,
                "num_inference_steps": num_inference_steps or config.default_steps,
                "guidance_scale": guidance_scale or config.default_guidance,
                **kwargs,
            }

            if config.use_true_cfg:
                pipe_kwargs["true_cfg_scale"] = true_cfg_scale or config.default_true_cfg

            if config.use_negative_prompt and negative_prompts:
                pipe_kwargs["negative_prompt"] = negative_prompts[start_idx:end_idx]

            if generator is not None:
                pipe_kwargs["generator"] = generator

            if num_batches > 1:
                print_rank0(f"Processing batch {batch_idx + 1}/{num_batches}")

            output = self.pipe(**pipe_kwargs)

            if hasattr(output, "images"):
                all_images.extend(output.images)
            else:
                all_images.extend(output)

        return all_images

    def edit_batch(
        self,
        batch_data: List[Tuple[List[Image.Image], str, int]],
        **kwargs,
    ) -> List[Tuple[int, Optional[Image.Image]]]:
        """Edit a batch of images with error handling.

        Args:
            batch_data: List of (images, instruction, index) tuples
            **kwargs: Additional editing arguments

        Returns:
            List of (index, edited_image) tuples
        """
        if not batch_data:
            return []

        batch_size = self.backend_config.batch_size

        # Sequential processing for batch_size=1
        if batch_size <= 1:
            results = []
            for images, instruction, idx in batch_data:
                try:
                    edited = self.edit(images, instruction, **kwargs)
                    results.append((idx, edited[0] if edited else None))
                except Exception as e:
                    log_error(
                        EditingError(f"Failed to edit index {idx}", cause=e),
                        context=f"edit_batch[{idx}]",
                        include_traceback=True,
                    )
                    results.append((idx, None))
            return results

        # Batch inference
        batch_images = [item[0] for item in batch_data]
        batch_instructions = [item[1] for item in batch_data]
        indices = [item[2] for item in batch_data]

        try:
            edited_images = self.batch_edit(
                batch_images=batch_images,
                batch_instructions=batch_instructions,
                **kwargs,
            )
            return list(zip(indices, edited_images))

        except Exception as e:
            log_error(
                EditingError("Batch inference failed", cause=e),
                context="edit_batch",
                include_traceback=True,
            )
            print_rank0("Falling back to sequential processing")

            # Fallback
            results = []
            for images, instruction, idx in batch_data:
                try:
                    edited = self.edit(images, instruction, **kwargs)
                    results.append((idx, edited[0] if edited else None))
                except Exception as e2:
                    log_error(
                        EditingError(f"Failed to edit index {idx}", cause=e2),
                        context=f"edit_batch[{idx}]",
                        include_traceback=True,
                    )
                    results.append((idx, None))
            return results

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._strategy = None
        self._effective_model_type = None
