# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Concrete optimizer implementations."""

from __future__ import annotations

from typing import Any

import torch

from diffgentor.config import OptimizationConfig
from diffgentor.optimizations.base import Optimizer, register_optimizer
from diffgentor.utils.logging import print_rank0


@register_optimizer
class TF32Optimizer(Optimizer):
    """Enable TF32 for Ampere+ GPUs."""

    @property
    def name(self) -> str:
        return "TF32"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_tf32

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print_rank0("Enabled TF32")
        return pipe


@register_optimizer
class VAESlicingOptimizer(Optimizer):
    """Enable VAE slicing for memory efficiency."""

    @property
    def name(self) -> str:
        return "VAE Slicing"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_vae_slicing

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
            print_rank0("Enabled VAE slicing")
        return pipe


@register_optimizer
class VAETilingOptimizer(Optimizer):
    """Enable VAE tiling for large images."""

    @property
    def name(self) -> str:
        return "VAE Tiling"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_vae_tiling

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
            print_rank0("Enabled VAE tiling")
        return pipe


@register_optimizer
class SequentialCPUOffloadOptimizer(Optimizer):
    """Enable sequential CPU offload."""

    @property
    def name(self) -> str:
        return "Sequential CPU Offload"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_sequential_cpu_offload

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
            print_rank0("Enabled sequential CPU offload")
        return pipe


@register_optimizer
class ModelCPUOffloadOptimizer(Optimizer):
    """Enable model CPU offload."""

    @property
    def name(self) -> str:
        return "Model CPU Offload"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_cpu_offload and not config.enable_sequential_cpu_offload

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            print_rank0("Enabled model CPU offload")
        return pipe


@register_optimizer
class XFormersOptimizer(Optimizer):
    """Enable xFormers memory efficient attention."""

    @property
    def name(self) -> str:
        return "xFormers"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_xformers

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
            print_rank0("Enabled xFormers memory efficient attention")
        return pipe


@register_optimizer
class FuseQKVOptimizer(Optimizer):
    """Fuse QKV projections."""

    @property
    def name(self) -> str:
        return "Fuse QKV"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_fuse_qkv

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "fuse_qkv_projections"):
            pipe.fuse_qkv_projections()
            print_rank0("Fused QKV projections")
        return pipe


@register_optimizer
class GroupOffloadingOptimizer(Optimizer):
    """Enable group offloading."""

    @property
    def name(self) -> str:
        return "Group Offloading"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_group_offloading

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        if hasattr(pipe, "enable_group_offload"):
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type=config.group_offload_type,
                use_stream=True,
            )
            print_rank0(f"Enabled group offloading (type={config.group_offload_type})")
        return pipe


@register_optimizer
class LayerwiseCastingOptimizer(Optimizer):
    """Enable layerwise dtype casting."""

    @property
    def name(self) -> str:
        return "Layerwise Casting"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_layerwise_casting

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        dtype_map = {
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        storage_dtype = dtype_map.get(config.storage_dtype)
        compute_dtype = dtype_map.get(config.compute_dtype)

        if storage_dtype is None or compute_dtype is None:
            print_rank0("Invalid dtype for layerwise casting")
            return pipe

        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "enable_layerwise_casting"):
            component.enable_layerwise_casting(
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
            )
            print_rank0(f"Enabled layerwise casting (storage={config.storage_dtype})")

        return pipe


@register_optimizer
class AttentionBackendOptimizer(Optimizer):
    """Configure attention backend."""

    @property
    def name(self) -> str:
        return "Attention Backend"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return bool(config.attention_backend)

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "set_attention_backend"):
            component.set_attention_backend(config.attention_backend)
            print_rank0(f"Set attention backend to: {config.attention_backend}")
        return pipe


@register_optimizer
class CompileOptimizer(Optimizer):
    """Apply torch.compile optimization."""

    @property
    def name(self) -> str:
        return "torch.compile"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_compile

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        # Configure inductor
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        for component_name in config.compile_components:
            component = getattr(pipe, component_name, None)
            if component is None:
                continue

            component.to(memory_format=torch.channels_last)
            compiled = torch.compile(
                component,
                mode=config.compile_mode,
                fullgraph=config.compile_fullgraph,
            )
            setattr(pipe, component_name, compiled)
            print_rank0(f"Compiled {component_name} with mode={config.compile_mode}")

        return pipe


@register_optimizer
class CacheOptimizer(Optimizer):
    """Apply cache acceleration."""

    @property
    def name(self) -> str:
        return "Cache"

    def should_apply(self, config: OptimizationConfig) -> bool:
        return bool(config.cache_type)

    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        cache_type = config.cache_type.lower()
        cache_config = config.cache_config

        cache_handlers = {
            "deep_cache": self._apply_deep_cache,
            "first_block_cache": self._apply_first_block_cache,
            "pab": self._apply_pab_cache,
            "faster_cache": self._apply_faster_cache,
            "taylor_seer": self._apply_taylor_seer_cache,
            "cache_dit": self._apply_cache_dit,
        }

        handler = cache_handlers.get(cache_type)
        if handler:
            return handler(pipe, cache_config)
        else:
            print_rank0(f"Unknown cache type: {cache_type}")
            return pipe

    def _apply_deep_cache(self, pipe: Any, cache_config: Dict) -> Any:
        from DeepCache import DeepCacheSDHelper

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=cache_config.get("cache_interval", 3),
            cache_branch_id=cache_config.get("cache_branch_id", 0),
        )
        helper.enable()
        print_rank0("Enabled DeepCache")
        return pipe

    def _apply_first_block_cache(self, pipe: Any, cache_config: Dict) -> Any:
        from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache

        fbc_config = FirstBlockCacheConfig(threshold=cache_config.get("threshold", 0.2))
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component:
            apply_first_block_cache(component, fbc_config)
            print_rank0("Enabled FirstBlockCache")
        return pipe

    def _apply_pab_cache(self, pipe: Any, cache_config: Dict) -> Any:
        from diffusers import PyramidAttentionBroadcastConfig

        pab_config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=cache_config.get("spatial_skip_range", 2),
            spatial_attention_timestep_skip_range=cache_config.get("timestep_skip_range", (100, 800)),
            current_timestep_callback=lambda: pipe.current_timestep,
        )
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "enable_cache"):
            component.enable_cache(pab_config)
            print_rank0("Enabled PAB cache")
        return pipe

    def _apply_faster_cache(self, pipe: Any, cache_config: Dict) -> Any:
        from diffusers import FasterCacheConfig

        fc_config = FasterCacheConfig(
            spatial_attention_block_skip_range=cache_config.get("spatial_skip_range", 2),
            spatial_attention_timestep_skip_range=cache_config.get("timestep_skip_range", (-1, 681)),
            current_timestep_callback=lambda: pipe.current_timestep,
            attention_weight_callback=lambda _: cache_config.get("attention_weight", 0.3),
            unconditional_batch_skip_range=cache_config.get("unconditional_batch_skip_range", 5),
        )
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "enable_cache"):
            component.enable_cache(fc_config)
            print_rank0("Enabled FasterCache")
        return pipe

    def _apply_taylor_seer_cache(self, pipe: Any, cache_config: Dict) -> Any:
        from diffusers import TaylorSeerCacheConfig

        ts_config = TaylorSeerCacheConfig(
            cache_interval=cache_config.get("cache_interval", 5),
            max_order=cache_config.get("max_order", 1),
            disable_cache_before_step=cache_config.get("disable_cache_before_step", 10),
        )
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "enable_cache"):
            component.enable_cache(ts_config)
            print_rank0("Enabled TaylorSeer cache")
        return pipe

    def _apply_cache_dit(self, pipe: Any, cache_config: Dict) -> Any:
        import cache_dit

        cache_dit.enable_cache(pipe)
        print_rank0("Enabled CacheDiT")
        return pipe
