"""
diffgentor - A unified visual generation data synthesis factory.

Supports multiple backends (diffusers, xDiT, OpenAI API) with various optimization methods.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("diffgentor")
except PackageNotFoundError:
    # Fallback for running from source without installing
    __version__ = "0.1.0"

from diffgentor.config import (
    BackendConfig,
    OptimizationConfig,
    T2IConfig,
    EditingConfig,
)

__all__ = [
    "__version__",
    "BackendConfig",
    "OptimizationConfig",
    "T2IConfig",
    "EditingConfig",
]
