"""
Prompt enhancement module for image editing and generation.
Supports various prompt enhancement strategies using LLM APIs.

Each enhancer type is implemented in a separate file for easy extension.
"""

from diffgentor.prompt_enhance.base import PromptEnhancer, encode_image_to_base64
from diffgentor.prompt_enhance.registry import get_prompt_enhancer, ENHANCER_REGISTRY
from diffgentor.prompt_enhance.flux2 import Flux2PromptEnhancer

__all__ = [
    "PromptEnhancer",
    "encode_image_to_base64",
    "get_prompt_enhancer",
    "ENHANCER_REGISTRY",
    "Flux2PromptEnhancer",
]
