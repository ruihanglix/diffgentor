# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Concrete model strategy implementations."""

from diffgentor.backends.editing.strategies.base import ModelConfig, ModelStrategy
from diffgentor.backends.editing.strategies.registry import register_strategy


@register_strategy("qwen")
class QwenStrategy(ModelStrategy):
    """Strategy for Qwen Image Edit Plus (multi-image)."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="QwenImageEditPlusPipeline",
            multi_image=True,
            batch_disabled=True,
            use_true_cfg=True,
            default_steps=40,
            default_guidance=1.0,
            default_true_cfg=4.0,
        )


@register_strategy("qwen_singleimg")
class QwenSingleImgStrategy(ModelStrategy):
    """Strategy for Qwen Image Edit (single-image)."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="QwenImageEditPipeline",
            multi_image=False,
            use_true_cfg=True,
            default_steps=50,
            default_guidance=1.0,
            default_true_cfg=4.0,
        )


@register_strategy("flux2")
class Flux2Strategy(ModelStrategy):
    """Strategy for Flux2 Pipeline."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="Flux2Pipeline",
            multi_image=True,
            shared_image_batch=True,
            use_true_cfg=False,
            default_steps=28,
            default_guidance=4.0,
        )


@register_strategy("flux2_klein")
class Flux2KleinStrategy(ModelStrategy):
    """Strategy for Flux2 Klein Pipeline."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="Flux2KleinPipeline",
            multi_image=True,
            shared_image_batch=True,
            use_true_cfg=False,
            default_steps=4,
            default_guidance=1.0,
        )


@register_strategy("flux1_kontext")
class Flux1KontextStrategy(ModelStrategy):
    """Strategy for Flux1 Kontext Pipeline."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="FluxKontextPipeline",
            multi_image=False,
            shared_image_batch=True,
            use_true_cfg=False,
            default_steps=30,
            default_guidance=2.5,
        )


@register_strategy("longcat")
class LongCatStrategy(ModelStrategy):
    """Strategy for LongCat Pipeline."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="LongCatImageEditPipeline",
            multi_image=False,
            batch_disabled=True,
            use_negative_prompt=True,
            default_steps=50,
            default_guidance=4.5,
        )


@register_strategy("glm_image")
class GlmImageStrategy(ModelStrategy):
    """Strategy for GLM Image Pipeline."""

    @property
    def config(self) -> ModelConfig:
        return ModelConfig(
            pipeline_class="GlmImagePipeline",
            multi_image=True,
            batch_disabled=True,
            default_steps=50,
            default_guidance=1.5,
        )
