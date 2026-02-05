# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Registry for model strategies."""

from pathlib import Path
from typing import Dict, Optional, Type

from diffgentor.backends.editing.strategies.base import (
    DefaultStrategy,
    ModelStrategy,
)


# Strategy registry: model_type -> strategy class
MODEL_STRATEGIES: Dict[str, Type[ModelStrategy]] = {}


def register_strategy(model_type: str):
    """Decorator to register a model strategy.

    Args:
        model_type: Model type identifier

    Returns:
        Decorator function
    """

    def decorator(cls: Type[ModelStrategy]) -> Type[ModelStrategy]:
        MODEL_STRATEGIES[model_type] = cls
        return cls

    return decorator


def get_model_strategy(model_type: Optional[str]) -> ModelStrategy:
    """Get strategy for a model type.

    Args:
        model_type: Model type identifier

    Returns:
        ModelStrategy instance
    """
    if model_type and model_type in MODEL_STRATEGIES:
        return MODEL_STRATEGIES[model_type]()
    return DefaultStrategy()


def detect_editing_model_type(model_name: str) -> Optional[str]:
    """Detect editing model type from model name.

    Args:
        model_name: Model name or path

    Returns:
        Detected model type or None
    """
    model_name_lower = model_name.lower()
    model_basename = Path(model_name).name.lower()

    # Qwen patterns
    QWEN_EDIT_PREFIX = "qwen-image-edit-"
    QWEN_EDIT_EXACT = "qwen-image-edit"

    for name in [model_name_lower, model_basename]:
        if QWEN_EDIT_PREFIX in name:
            return "qwen"
        if QWEN_EDIT_EXACT in name:
            idx = name.find(QWEN_EDIT_EXACT)
            end_idx = idx + len(QWEN_EDIT_EXACT)
            if end_idx == len(name):
                return "qwen_singleimg"
            next_char = name[end_idx]
            if next_char == "-":
                return "qwen"
            elif not next_char.isalnum():
                return "qwen_singleimg"

    # Other patterns
    patterns = {
        "FLUX.2-dev": "flux2",
        "FLUX.2-klein": "flux2_klein",
        "FLUX.1-Kontext": "flux1_kontext",
        "LongCat": "longcat",
        "GLM-Image": "glm_image",
    }

    for pattern, model_type in patterns.items():
        if pattern.lower() in model_name_lower or pattern.lower() in model_basename:
            return model_type

    return None


# Import concrete strategies to trigger registration
def _register_builtin_strategies():
    """Register built-in strategies."""
    from diffgentor.backends.editing.strategies import implementations  # noqa: F401


# Auto-register on import
_register_builtin_strategies()
