# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""LoRA adapter management mixin for diffusers-based backends.

Provides dynamic loading, activation, fusion, and unloading of LoRA adapters
on any diffusers ``DiffusionPipeline`` that supports the PEFT LoRA interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("diffgentor.backends.lora")


@dataclass
class _LoraAdapterState:
    """Tracked state for a loaded LoRA adapter."""

    nickname: str
    path: str
    strength: float = 1.0
    fused: bool = False


class DiffusersLoRAMixin:
    """Mixin that adds LoRA management to any backend with a ``self.pipe`` diffusers pipeline.

    The mixin keeps an internal registry of loaded adapters and delegates to
    the diffusers ``load_lora_weights`` / ``set_adapters`` / ``fuse_lora`` /
    ``unfuse_lora`` / ``delete_adapters`` APIs.
    """

    def _init_lora_state(self) -> None:
        """Initialize LoRA tracking structures. Call from ``__init__`` or ``load_model``."""
        self._lora_adapters: Dict[str, _LoraAdapterState] = {}
        self._lora_fused: bool = False

    # -- public API (mirrors BasePipelineBackend signatures) ----------------

    @property
    def supports_lora(self) -> bool:  # type: ignore[override]
        if self.pipe is None:
            return False
        return hasattr(self.pipe, "load_lora_weights")

    def load_lora(self, lora_path: str, adapter_name: str, strength: float = 1.0) -> None:
        """Load a single LoRA adapter and make it active.

        Convenience wrapper around ``load_loras`` for the single-adapter case.
        """
        self.load_loras([lora_path], [adapter_name], [strength])

    def load_loras(
        self,
        lora_paths: List[str],
        adapter_names: List[str],
        strengths: List[float],
        targets: Optional[List[str]] = None,
    ) -> None:
        """Load one or more LoRA adapters and activate them all simultaneously.

        Adapters that are already loaded (same nickname + path) are reused from
        cache without re-downloading.  All specified adapters are activated
        together via ``set_adapters``.

        Args:
            targets: Per-adapter target component.  Accepted values depend on
                the pipeline; the default ``"all"`` applies to every component
                that has LoRA support.  For diffusers pipelines with a single
                transformer/unet this is always ``"all"``.
        """
        self._ensure_lora_state()
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Pipeline not loaded")

        n = len(lora_paths)
        if len(adapter_names) != n or len(strengths) != n:
            raise ValueError("lora_paths, adapter_names, and strengths must have the same length")
        if targets is not None and len(targets) != n:
            raise ValueError("targets must have the same length as lora_paths")

        if self._lora_fused:
            logger.info("Unfusing current LoRA before loading new adapters")
            self.unfuse_lora()

        for i, (lora_path, adapter_name, strength) in enumerate(zip(lora_paths, adapter_names, strengths)):
            if adapter_name in self._lora_adapters:
                existing = self._lora_adapters[adapter_name]
                if existing.path == lora_path:
                    logger.info("Adapter '%s' already loaded from '%s', reusing from cache", adapter_name, lora_path)
                    existing.strength = strength
                    continue
                else:
                    logger.info("Adapter '%s' loaded from different path, replacing", adapter_name)
                    pipe.delete_adapters([adapter_name])
                    del self._lora_adapters[adapter_name]

            target = (targets[i] if targets else "all") or "all"
            load_kwargs: Dict[str, Any] = {"adapter_name": adapter_name}
            if target != "all":
                load_kwargs["components"] = [target]

            logger.info(
                "Loading LoRA adapter '%s' from '%s' (strength=%.2f, target=%s)",
                adapter_name, lora_path, strength, target,
            )
            pipe.load_lora_weights(lora_path, **load_kwargs)
            self._lora_adapters[adapter_name] = _LoraAdapterState(
                nickname=adapter_name, path=lora_path, strength=strength,
            )

        self._activate_adapters(list(adapter_names), list(strengths))
        names_str = ", ".join(adapter_names)
        logger.info("LoRA adapter(s) loaded and activated: %s", names_str)

    def set_active_loras(self, adapter_names: List[str], strengths: List[float]) -> None:
        self._ensure_lora_state()
        if self._lora_fused:
            raise RuntimeError("Cannot change active adapters while LoRA is fused. Call unfuse_lora() first.")
        for name in adapter_names:
            if name not in self._lora_adapters:
                raise ValueError(f"Adapter '{name}' is not loaded")
        self._activate_adapters(adapter_names, strengths)

    def fuse_lora(self, adapter_names: Optional[List[str]] = None, lora_scale: float = 1.0) -> None:
        self._ensure_lora_state()
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Pipeline not loaded")
        if self._lora_fused:
            logger.warning("LoRA already fused, unfusing first")
            self.unfuse_lora()

        kwargs: Dict[str, Any] = {"lora_scale": lora_scale}
        if adapter_names is not None:
            kwargs["adapter_names"] = adapter_names

        pipe.fuse_lora(**kwargs)
        self._lora_fused = True
        for state in self._lora_adapters.values():
            if adapter_names is None or state.nickname in adapter_names:
                state.fused = True
        logger.info("LoRA weights fused (scale=%.2f)", lora_scale)

    def unfuse_lora(self) -> None:
        self._ensure_lora_state()
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Pipeline not loaded")
        if not self._lora_fused:
            logger.info("No LoRA is currently fused, nothing to do")
            return
        pipe.unfuse_lora()
        self._lora_fused = False
        for state in self._lora_adapters.values():
            state.fused = False
        logger.info("LoRA weights unfused")

    def unload_lora(self, adapter_name: Optional[str] = None) -> None:
        self._ensure_lora_state()
        pipe = self.pipe
        if pipe is None:
            raise RuntimeError("Pipeline not loaded")

        if self._lora_fused:
            self.unfuse_lora()

        if adapter_name is not None:
            if adapter_name not in self._lora_adapters:
                raise ValueError(f"Adapter '{adapter_name}' is not loaded")
            pipe.delete_adapters([adapter_name])
            del self._lora_adapters[adapter_name]
            logger.info("Unloaded adapter '%s'", adapter_name)
        else:
            if self._lora_adapters:
                pipe.unload_lora_weights()
                self._lora_adapters.clear()
                logger.info("All LoRA adapters unloaded")

    def list_loras(self) -> Dict[str, Any]:
        self._ensure_lora_state()
        loaded = [
            {
                "nickname": s.nickname,
                "path": s.path,
                "strength": s.strength,
                "fused": s.fused,
            }
            for s in self._lora_adapters.values()
        ]

        active: Optional[List[str]] = None
        if self.pipe is not None and hasattr(self.pipe, "get_active_adapters"):
            try:
                active = self.pipe.get_active_adapters()
            except Exception:
                active = None

        return {
            "loaded_adapters": loaded,
            "active": active,
            "fused": self._lora_fused,
        }

    # -- internal helpers ---------------------------------------------------

    def _ensure_lora_state(self) -> None:
        if not hasattr(self, "_lora_adapters"):
            self._init_lora_state()

    def _activate_adapters(self, names: List[str], strengths: List[float]) -> None:
        pipe = self.pipe
        if pipe is None:
            return
        pipe.set_adapters(names, adapter_weights=strengths)
        for name, strength in zip(names, strengths):
            if name in self._lora_adapters:
                self._lora_adapters[name].strength = strength
