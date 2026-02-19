# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Backend registry for diffgentor."""

from typing import Callable, Dict, Optional, Type

from diffgentor.backends.base import BaseBackend
from diffgentor.config import BackendConfig, OptimizationConfig

# Registry for T2I backends
BACKEND_REGISTRY: Dict[str, Type[BaseBackend]] = {}


def register_backend(name: str) -> Callable:
    """Decorator to register a backend class.

    Args:
        name: Backend name

    Returns:
        Decorator function
    """

    def decorator(cls: Type[BaseBackend]) -> Type[BaseBackend]:
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(
    backend_config: BackendConfig,
    optimization_config: Optional[OptimizationConfig] = None,
) -> BaseBackend:
    """Get a backend instance by configuration.

    Args:
        backend_config: Backend configuration
        optimization_config: Optional optimization configuration

    Returns:
        Backend instance

    Raises:
        ValueError: If backend type is not supported
    """
    backend_name = backend_config.backend.lower()

    # Lazy import backends to avoid circular imports
    if backend_name == "diffusers":
        from diffgentor.backends.t2i.diffusers_backend import DiffusersBackend

        return DiffusersBackend(backend_config, optimization_config)
    elif backend_name == "xdit":
        from diffgentor.backends.t2i.xdit_backend import XDiTBackend

        return XDiTBackend(backend_config, optimization_config)
    elif backend_name == "openai":
        from diffgentor.backends.t2i.openai_backend import OpenAIBackend

        return OpenAIBackend(backend_config, optimization_config)
    elif backend_name in ("google_genai", "gemini"):
        from diffgentor.backends.t2i.google_genai_backend import GoogleGenAIBackend

        return GoogleGenAIBackend(backend_config, optimization_config)
    elif backend_name == "deepgen":
        from diffgentor.backends.t2i.deepgen import DeepGenT2IBackend

        return DeepGenT2IBackend(backend_config, optimization_config)
    else:
        if backend_name in BACKEND_REGISTRY:
            return BACKEND_REGISTRY[backend_name](backend_config, optimization_config)
        raise ValueError(f"Unknown backend: {backend_name}. Available: diffusers, xdit, openai, google_genai, deepgen")
