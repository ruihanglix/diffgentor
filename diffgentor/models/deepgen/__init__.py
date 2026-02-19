# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen model module.

This module contains the DeepGen model implementation for unified visual generation,
supporting both text-to-image generation and image editing.

The model architecture consists of:
- Qwen2.5-VL as the language/vision understanding module
- SD3.5 Transformer as the image generation module
- Connector module to bridge LLM and DiT

Based on the DeepGen-SFT project.
"""

from diffgentor.models.deepgen.model import DeepGenModel

__all__ = ["DeepGenModel"]
