# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""FastAPI application factory and model lifecycle for diffgentor serve mode."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Optional

from diffgentor.backends.base import BaseBackend, BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig

logger = logging.getLogger("diffgentor.serve")


def _build_backend_config(args: Namespace) -> BackendConfig:
    return BackendConfig(
        backend=args.backend,
        model_name=args.model_name,
        model_type=getattr(args, "model_type", None),
        device=getattr(args, "device", "cuda"),
        seed=getattr(args, "seed", None),
    )


def _build_optimization_config(args: Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        torch_dtype=getattr(args, "torch_dtype", "bfloat16"),
        enable_vae_slicing=getattr(args, "enable_vae_slicing", False),
        enable_vae_tiling=getattr(args, "enable_vae_tiling", False),
        enable_cpu_offload=getattr(args, "enable_cpu_offload", False),
        enable_compile=getattr(args, "enable_compile", False),
        attention_backend=getattr(args, "attention_backend", None),
        cache_type=getattr(args, "cache_type", None),
    )


def _load_t2i_backend(
    backend_config: BackendConfig,
    optimization_config: OptimizationConfig,
) -> BaseBackend:
    from diffgentor.backends.registry import get_backend

    backend = get_backend(backend_config, optimization_config)
    backend.load_model()
    return backend


def _load_editing_backend(
    backend_config: BackendConfig,
    optimization_config: OptimizationConfig,
) -> BaseEditingBackend:
    from diffgentor.backends.editing.registry import get_editing_backend

    backend = get_editing_backend(backend_config, optimization_config)
    backend.load_model()
    return backend


def _detect_num_gpus(args: Namespace) -> int:
    """Resolve the effective number of GPUs to use."""
    import os

    num_gpus = getattr(args, "num_gpus", None)
    if num_gpus is not None:
        return num_gpus

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return len([g for g in cuda_visible.split(",") if g.strip()])

    try:
        import torch
        count = torch.cuda.device_count()
        return max(count, 1)
    except (ImportError, Exception):
        return 1


def run_serve(args: Namespace) -> int:
    """Start the OpenAI-compatible API server.

    Args:
        args: Parsed CLI arguments (from the ``serve`` subcommand).

    Returns:
        Exit code (0 on clean shutdown).
    """
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print(
            "ERROR: serve mode requires fastapi and uvicorn. "
            "Install them with:  pip install diffgentor[serve]"
        )
        return 1

    mode: str = getattr(args, "mode", "t2i")
    host: str = getattr(args, "host", "0.0.0.0")
    port: int = getattr(args, "port", 8000)
    max_concurrent: int = getattr(args, "max_concurrent", 1)

    backend_config = _build_backend_config(args)
    opt_config = _build_optimization_config(args)
    num_gpus = _detect_num_gpus(args)

    # ------------------------------------------------------------------
    # Try to create a multi-GPU worker pool
    # ------------------------------------------------------------------
    from diffgentor.serve.worker_pool import (
        InProcessPool,
        SubprocessPool,
        WorkerPool,
        create_worker_pool,
    )

    pool: Optional[WorkerPool] = None
    t2i_backend: Optional[BaseBackend] = None
    editing_backend: Optional[BaseEditingBackend] = None

    if num_gpus > 1:
        logger.info("Attempting multi-GPU pool with %d GPUs ...", num_gpus)
        pool = create_worker_pool(mode, backend_config, opt_config, num_gpus)

    if pool is not None:
        logger.info("Multi-GPU worker pool created (%d workers)", pool._num_workers)
    else:
        # Fallback: single-backend path
        if mode == "t2i":
            logger.info("Loading T2I backend: %s / %s", backend_config.backend, backend_config.model_name)
            t2i_backend = _load_t2i_backend(backend_config, opt_config)
            logger.info("T2I backend loaded successfully")
        elif mode == "edit":
            logger.info("Loading editing backend: %s / %s", backend_config.backend, backend_config.model_name)
            editing_backend = _load_editing_backend(backend_config, opt_config)
            logger.info("Editing backend loaded successfully")
        else:
            print(f"ERROR: unknown mode '{mode}'. Use 't2i' or 'edit'.")
            return 1

    # ------------------------------------------------------------------
    # Create FastAPI app & wire routes
    # ------------------------------------------------------------------
    from diffgentor.serve.routes import configure, router

    configure(
        t2i_backend=t2i_backend,
        editing_backend=editing_backend,
        model_name=backend_config.model_name,
        max_concurrent=max_concurrent,
        worker_pool=pool,
        mode=mode,
    )

    app = fastapi.FastAPI(
        title="diffgentor",
        description="OpenAI-compatible image generation / editing API powered by diffgentor",
    )
    app.include_router(router)

    # Start async worker tasks after the event loop is available
    @app.on_event("startup")
    async def _start_pool_workers() -> None:
        if pool is not None:
            if isinstance(pool, InProcessPool):
                pool.start_workers()
            elif isinstance(pool, SubprocessPool):
                pool.start_reader()

    @app.on_event("shutdown")
    async def _shutdown_pool() -> None:
        if pool is not None:
            await pool.shutdown()

    # ------------------------------------------------------------------
    # Launch uvicorn
    # ------------------------------------------------------------------
    if pool is not None:
        print(f"Starting server on {host}:{port}  (mode={mode}, model={backend_config.model_name}, "
              f"workers={pool._num_workers})")
    else:
        print(f"Starting server on {host}:{port}  (mode={mode}, model={backend_config.model_name})")
    print(f"  POST /v1/images/generations  {'[enabled]' if (t2i_backend or (pool and mode == 't2i')) else '[disabled]'}")
    print(f"  POST /v1/images/edits        {'[enabled]' if (editing_backend or (pool and mode == 'edit')) else '[disabled]'}")
    print("  GET  /v1/models              [enabled]")
    print("  POST /v1/set_lora            [enabled]")
    print("  POST /v1/merge_lora_weights  [enabled]")
    print("  POST /v1/unmerge_lora_weights [enabled]")
    print("  GET  /v1/list_loras          [enabled]")
    print("  GET  /health                 [enabled]")

    uvicorn.run(app, host=host, port=port, log_level="info")

    # Cleanup on shutdown (single-backend path)
    if t2i_backend is not None:
        t2i_backend.cleanup()
    if editing_backend is not None:
        editing_backend.cleanup()

    return 0
