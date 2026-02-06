# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen model package - Qwen2.5-VL + SD3.5 for unified image generation and editing.

This package provides the DeepGen model which combines:
- Qwen2.5-VL as the language/vision encoder
- SD3.5 Transformer as the diffusion backbone
- A connector module to bridge LLM hidden states to DiT

Example usage:
    from diffgentor.models.deepgen_v1 import DeepGenModel

    model = DeepGenModel(
        diffusion_path="/path/to/diffusion_model",
        qwen_path="/path/to/qwen2.5-vl",
    )
    model.load_checkpoint("/path/to/checkpoint.pt")

    # Text-to-image generation
    images = model.generate(
        prompt="A cat sitting on a windowsill",
        height=1024,
        width=1024,
    )

    # Image editing
    images = model.generate(
        prompt="Make the cat wear a hat",
        pixel_values_src=[[source_image_tensor]],
        height=1024,
        width=1024,
    )
"""

from .modeling import DeepGenModel, resize_image
from .connector import ConnectorConfig, ConnectorEncoder
from .transformer import SD3Transformer2DModel
from .pipeline import StableDiffusion3Pipeline

__all__ = [
    "DeepGenModel",
    "resize_image",
    "ConnectorConfig",
    "ConnectorEncoder",
    "SD3Transformer2DModel",
    "StableDiffusion3Pipeline",
]
