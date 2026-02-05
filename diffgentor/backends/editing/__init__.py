"""Editing backend implementations."""

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.backends.editing.registry import get_editing_backend, EDITING_BACKEND_REGISTRY

__all__ = [
    "BaseEditingBackend",
    "get_editing_backend",
    "EDITING_BACKEND_REGISTRY",
]
