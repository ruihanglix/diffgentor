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

    # ------------------------------------------------------------------
    # Load backend(s)
    # ------------------------------------------------------------------
    t2i_backend: Optional[BaseBackend] = None
    editing_backend: Optional[BaseEditingBackend] = None

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
    )

    app = fastapi.FastAPI(
        title="diffgentor",
        description="OpenAI-compatible image generation / editing API powered by diffgentor",
    )
    app.include_router(router)

    # ------------------------------------------------------------------
    # Launch uvicorn
    # ------------------------------------------------------------------
    print(f"Starting server on {host}:{port}  (mode={mode}, model={backend_config.model_name})")
    print(f"  POST /v1/images/generations  {'[enabled]' if t2i_backend else '[disabled]'}")
    print(f"  POST /v1/images/edits        {'[enabled]' if editing_backend else '[disabled]'}")
    print(f"  GET  /v1/models              [enabled]")

    uvicorn.run(app, host=host, port=port, log_level="info")

    # Cleanup on shutdown
    if t2i_backend is not None:
        t2i_backend.cleanup()
    if editing_backend is not None:
        editing_backend.cleanup()

    return 0
