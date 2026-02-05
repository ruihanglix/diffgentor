# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Registry for editing backends."""

from typing import Dict, Optional, Type

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig

# Registry for editing backends
EDITING_BACKEND_REGISTRY: Dict[str, Type[BaseEditingBackend]] = {}


def register_editing_backend(name: str):
    """Decorator to register an editing backend class.

    Args:
        name: Backend name

    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseEditingBackend]) -> Type[BaseEditingBackend]:
        EDITING_BACKEND_REGISTRY[name] = cls
        return cls
    return decorator


def get_editing_backend(
    backend_config: BackendConfig,
    optimization_config: Optional[OptimizationConfig] = None,
) -> BaseEditingBackend:
    """Get an editing backend instance by configuration.

    Args:
        backend_config: Backend configuration
        optimization_config: Optional optimization configuration

    Returns:
        Editing backend instance

    Raises:
        ValueError: If backend type is not supported
    """
    backend_name = backend_config.backend.lower()
    model_type = backend_config.model_type

    # Check if it's a diffusers-based editing model
    diffusers_models = {
        "qwen", "qwen_singleimg", "flux2", "flux2_klein",
        "flux1_kontext", "longcat", "glm_image",
    }

    # Check if it's a third-party model
    third_party_models = {
        "flux_kontext_official", "bagel", "step1x", "emu35", "dreamomni2", "hunyuan_image_3",
    }

    if backend_name == "diffusers" or model_type in diffusers_models:
        from diffgentor.backends.editing.diffusers_editing import DiffusersEditingBackend
        return DiffusersEditingBackend(backend_config, optimization_config)

    elif backend_name == "openai":
        from diffgentor.backends.editing.openai_editing import OpenAIEditingBackend
        return OpenAIEditingBackend(backend_config, optimization_config)

    elif backend_name in ("google_genai", "gemini"):
        from diffgentor.backends.editing.google_genai_editing import GoogleGenAIEditingBackend
        return GoogleGenAIEditingBackend(backend_config, optimization_config)

    elif model_type in third_party_models or backend_name in third_party_models:
        # Route to specific third-party backend
        effective_type = model_type or backend_name

        if effective_type == "flux_kontext_official":
            from diffgentor.backends.editing.flux_kontext import FluxKontextOfficialBackend
            return FluxKontextOfficialBackend(backend_config, optimization_config)
        elif effective_type == "bagel":
            from diffgentor.backends.editing.bagel import BagelBackend
            return BagelBackend(backend_config, optimization_config)
        elif effective_type == "step1x":
            from diffgentor.backends.editing.step1x import Step1XBackend
            return Step1XBackend(backend_config, optimization_config)
        elif effective_type == "emu35":
            from diffgentor.backends.editing.emu35 import Emu35Backend
            return Emu35Backend(backend_config, optimization_config)
        elif effective_type == "dreamomni2":
            from diffgentor.backends.editing.dreamomni2 import DreamOmni2Backend
            return DreamOmni2Backend(backend_config, optimization_config)
        elif effective_type == "hunyuan_image_3":
            from diffgentor.backends.editing.hunyuan_image_3 import HunyuanImage3Backend
            return HunyuanImage3Backend(backend_config, optimization_config)

    # Check registry for custom backends
    if backend_name in EDITING_BACKEND_REGISTRY:
        return EDITING_BACKEND_REGISTRY[backend_name](backend_config, optimization_config)

    raise ValueError(
        f"Unknown editing backend: {backend_name} (model_type={model_type}). "
        f"Available: diffusers, openai, google_genai, flux_kontext_official, bagel, step1x, emu35, dreamomni2, hunyuan_image_3"
    )
