# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Base optimizer class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from diffgentor.config import OptimizationConfig
from diffgentor.utils.exceptions import OptimizationError, log_error


class Optimizer(ABC):
    """Abstract base class for pipeline optimizers.

    Each optimizer handles a specific type of optimization (e.g., memory,
    attention, compilation, caching).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Optimizer name for logging."""
        pass

    @abstractmethod
    def should_apply(self, config: OptimizationConfig) -> bool:
        """Check if this optimizer should be applied.

        Args:
            config: Optimization configuration

        Returns:
            True if optimizer should be applied
        """
        pass

    @abstractmethod
    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        """Apply optimization to pipeline.

        Args:
            pipe: Pipeline instance
            config: Optimization configuration

        Returns:
            Optimized pipeline
        """
        pass

    def safe_apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        """Safely apply optimization with error handling.

        Args:
            pipe: Pipeline instance
            config: Optimization configuration

        Returns:
            Pipeline (optimized or unchanged on error)
        """
        try:
            return self.apply(pipe, config)
        except Exception as e:
            log_error(
                OptimizationError(f"Failed to apply {self.name}", cause=e),
                context=f"Optimizer.{self.name}",
                include_traceback=True,
            )
            return pipe


# Registry for optimizers
_OPTIMIZER_REGISTRY: list[type[Optimizer]] = []


def register_optimizer(cls: type[Optimizer]) -> type[Optimizer]:
    """Decorator to register an optimizer class.

    Args:
        cls: Optimizer class to register

    Returns:
        The registered class
    """
    _OPTIMIZER_REGISTRY.append(cls)
    return cls


def get_registered_optimizers() -> list[type[Optimizer]]:
    """Get all registered optimizer classes.

    Returns:
        List of optimizer classes
    """
    return _OPTIMIZER_REGISTRY.copy()
