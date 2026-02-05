# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Registry for prompt enhancers.
New enhancer types should be registered here.
"""

from typing import Dict, Optional, Type

from diffgentor.prompt_enhance.base import PromptEnhancer


# Registry of available enhancers (lazy loading)
# Format: {"enhancer_type": ("module_name", "ClassName")}
_ENHANCER_REGISTRY: Dict[str, tuple] = {
    "qwen_image_edit": ("diffgentor.prompt_enhance.qwen_image_edit", "QwenImageEditEnhancer"),
    "glm_image": ("diffgentor.prompt_enhance.glm_image", "GlmImageEnhancer"),
    "flux2": ("diffgentor.prompt_enhance.flux2", "Flux2PromptEnhancer"),
}

# Cache for loaded classes
_LOADED_CLASSES: Dict[str, Type[PromptEnhancer]] = {}


def _load_enhancer_class(enhancer_type: str) -> Type[PromptEnhancer]:
    """
    Lazily load an enhancer class.

    Args:
        enhancer_type: Type of enhancer

    Returns:
        Enhancer class
    """
    if enhancer_type in _LOADED_CLASSES:
        return _LOADED_CLASSES[enhancer_type]

    if enhancer_type not in _ENHANCER_REGISTRY:
        raise ValueError(
            f"Unknown enhancer type: {enhancer_type}. "
            f"Available types: {list(_ENHANCER_REGISTRY.keys())}"
        )

    module_name, class_name = _ENHANCER_REGISTRY[enhancer_type]

    import importlib
    module = importlib.import_module(module_name)
    enhancer_class = getattr(module, class_name)

    _LOADED_CLASSES[enhancer_type] = enhancer_class
    return enhancer_class


def get_prompt_enhancer(
    enhancer_type: str,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> PromptEnhancer:
    """
    Get a prompt enhancer by type.

    Args:
        enhancer_type: Type of enhancer (e.g., "qwen_image_edit", "glm_image")
        api_key: API key for the LLM service
        api_base: Base URL for the API
        model: Model name to use
        **kwargs: Additional arguments passed to the enhancer

    Returns:
        PromptEnhancer instance

    Example:
        >>> enhancer = get_prompt_enhancer(
        ...     "qwen_image_edit",
        ...     api_key="your-api-key",
        ...     api_base="https://api.example.com/v1",
        ...     model="gpt-4o"
        ... )
        >>> enhanced = enhancer.enhance("Add a cat", images=[img])
    """
    enhancer_class = _load_enhancer_class(enhancer_type)
    return enhancer_class(api_key=api_key, api_base=api_base, model=model, **kwargs)


def register_enhancer(enhancer_type: str, module_name: str, class_name: str) -> None:
    """
    Register a new enhancer type.

    Args:
        enhancer_type: Type name for the enhancer
        module_name: Full module path (e.g., "diffgentor.prompt_enhance.my_enhancer")
        class_name: Class name in the module

    Example:
        >>> register_enhancer(
        ...     "my_custom_enhancer",
        ...     "diffgentor.prompt_enhance.my_enhancer",
        ...     "MyCustomEnhancer"
        ... )
    """
    _ENHANCER_REGISTRY[enhancer_type] = (module_name, class_name)
    # Clear cache if already loaded
    if enhancer_type in _LOADED_CLASSES:
        del _LOADED_CLASSES[enhancer_type]


# Public registry for inspection
ENHANCER_REGISTRY = _ENHANCER_REGISTRY
