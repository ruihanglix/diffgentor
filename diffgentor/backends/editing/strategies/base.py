# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Base strategy class for model-specific editing behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass
class ModelConfig:
    """Configuration for a specific model type."""

    pipeline_class: str | None = None
    multi_image: bool = False
    batch_disabled: bool = False
    shared_image_batch: bool = False
    use_true_cfg: bool = False
    use_negative_prompt: bool = False
    default_steps: int = 28
    default_guidance: float = 3.5
    default_true_cfg: float = 4.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class ModelStrategy(ABC):
    """Abstract strategy for model-specific editing behavior.

    Each model type can have different:
    - Pipeline class
    - Input image handling (single vs multi-image)
    - Batch inference support
    - Default parameters
    - Special kwargs handling
    """

    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        """Get model configuration."""
        pass

    @property
    def pipeline_class_name(self) -> str | None:
        """Get pipeline class name."""
        return self.config.pipeline_class

    @property
    def supports_batch(self) -> bool:
        """Check if model supports batch inference."""
        return not self.config.batch_disabled and not self.config.shared_image_batch

    def prepare_images(
        self, images: Image.Image | list[Image.Image]
    ) -> Image.Image | list[Image.Image]:
        """Prepare images for the pipeline.

        Args:
            images: Input image(s)

        Returns:
            Processed image(s) for the pipeline
        """
        if isinstance(images, Image.Image):
            images = [images]

        if self.config.multi_image:
            return images
        else:
            return images[0] if images else None

    def build_pipeline_kwargs(
        self,
        images: Image.Image | list[Image.Image],
        instruction: str,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        true_cfg_scale: float | None = None,
        negative_prompt: str | None = None,
        generator: Any | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build kwargs for pipeline call.

        Args:
            images: Input image(s)
            instruction: Editing instruction
            num_inference_steps: Denoising steps
            guidance_scale: Guidance scale
            true_cfg_scale: True CFG scale
            negative_prompt: Negative prompt
            generator: Random generator
            **kwargs: Additional kwargs

        Returns:
            Dict of pipeline kwargs
        """
        config = self.config

        pipe_kwargs = {
            "prompt": instruction,
            "image": self.prepare_images(images),
            "num_inference_steps": num_inference_steps or config.default_steps,
            "guidance_scale": guidance_scale or config.default_guidance,
            **kwargs,
        }

        if config.use_true_cfg:
            pipe_kwargs["true_cfg_scale"] = true_cfg_scale or config.default_true_cfg

        if config.use_negative_prompt and negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt

        if generator is not None:
            pipe_kwargs["generator"] = generator

        return pipe_kwargs


class DefaultStrategy(ModelStrategy):
    """Default strategy for unknown model types."""

    def __init__(self):
        self._config = ModelConfig()

    @property
    def config(self) -> ModelConfig:
        return self._config
