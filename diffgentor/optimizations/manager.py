# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Optimization manager for applying pipeline optimizations."""

from typing import Any

from diffgentor.config import OptimizationConfig
from diffgentor.optimizations.base import get_registered_optimizers

# Import optimizers to trigger registration
from diffgentor.optimizations import optimizers as _  # noqa: F401


class OptimizationManager:
    """Manager for applying various optimizations to diffusers pipelines.

    This class uses the optimizer registry to apply all configured
    optimizations in a modular, extensible way.
    """

    def __init__(self, config: OptimizationConfig):
        """Initialize optimization manager.

        Args:
            config: Optimization configuration
        """
        self.config = config

    def apply_all(self, pipe: Any) -> Any:
        """Apply all configured optimizations to pipeline.

        Args:
            pipe: Diffusers pipeline instance

        Returns:
            Optimized pipeline
        """
        if pipe is None:
            return pipe

        optimizer_classes = get_registered_optimizers()

        for optimizer_cls in optimizer_classes:
            optimizer = optimizer_cls()
            if optimizer.should_apply(self.config):
                pipe = optimizer.safe_apply(pipe, self.config)

        return pipe


def parse_optimization_string(opt_string: str) -> OptimizationConfig:
    """Parse optimization string to config.

    Supports comma-separated optimization names like:
    "torch_compile,vae_slicing,flash_attention"

    Args:
        opt_string: Comma-separated optimization names

    Returns:
        OptimizationConfig with enabled optimizations
    """
    config = OptimizationConfig()

    if not opt_string:
        return config

    opts = [o.strip().lower() for o in opt_string.split(",")]

    for opt in opts:
        if opt in ("torch_compile", "compile"):
            config.enable_compile = True
        elif opt == "vae_slicing":
            config.enable_vae_slicing = True
        elif opt == "vae_tiling":
            config.enable_vae_tiling = True
        elif opt == "cpu_offload":
            config.enable_cpu_offload = True
        elif opt == "sequential_cpu_offload":
            config.enable_sequential_cpu_offload = True
        elif opt == "xformers":
            config.enable_xformers = True
        elif opt in ("flash_attention", "flash"):
            config.attention_backend = "flash"
        elif opt in ("sage_attention", "sage"):
            config.attention_backend = "sage"
        elif opt == "fuse_qkv":
            config.enable_fuse_qkv = True
        elif opt == "group_offload":
            config.enable_group_offloading = True
        elif opt == "layerwise_cast":
            config.enable_layerwise_casting = True
        elif opt == "deep_cache":
            config.cache_type = "deep_cache"
        elif opt == "first_block_cache":
            config.cache_type = "first_block_cache"
        elif opt in ("pab_cache", "pab"):
            config.cache_type = "pab"
        elif opt == "faster_cache":
            config.cache_type = "faster_cache"
        elif opt == "cache_dit":
            config.cache_type = "cache_dit"
        elif opt == "tf32":
            config.enable_tf32 = True

    return config
