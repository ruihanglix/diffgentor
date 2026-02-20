# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""OpenAI-compatible API route handlers for diffgentor serve mode."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from diffgentor.serve.schemas import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ImageData,
    ImageGenerateRequest,
    ImagesResponse,
    ModelListResponse,
    ModelObject,
)

if TYPE_CHECKING:
    from diffgentor.serve.worker_pool import WorkerPool

logger = logging.getLogger("diffgentor.serve")

router = APIRouter()

# These are set by app.py at startup
_t2i_backend = None
_editing_backend = None
_model_name: str = ""
_gpu_semaphore: Optional[asyncio.Semaphore] = None
_worker_pool: Optional["WorkerPool"] = None
_serve_mode: Optional[str] = None


def configure(
    *,
    t2i_backend=None,
    editing_backend=None,
    model_name: str = "",
    max_concurrent: int = 1,
    worker_pool: Optional["WorkerPool"] = None,
    mode: str = "t2i",
) -> None:
    """Inject backend instances and settings into the router.

    Called once by app.py after the model has been loaded.
    """
    global _t2i_backend, _editing_backend, _model_name, _gpu_semaphore, _worker_pool, _serve_mode
    _t2i_backend = t2i_backend
    _editing_backend = editing_backend
    _model_name = model_name
    _gpu_semaphore = asyncio.Semaphore(max_concurrent)
    _worker_pool = worker_pool
    _serve_mode = mode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_size(size: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    """Parse an OpenAI size string like '1024x1024' into (width, height)."""
    if not size or size == "auto":
        return None, None
    try:
        w_str, h_str = size.lower().split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return None, None


def _pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string."""
    fmt_upper = fmt.upper()
    if fmt_upper in ("JPG", "JPEG"):
        fmt_upper = "JPEG"
    buf = io.BytesIO()
    image.save(buf, format=fmt_upper)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _read_upload_as_pil(upload: UploadFile) -> Image.Image:
    """Read an UploadFile into a PIL Image (RGB)."""
    data = await upload.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _build_inference_kwargs(
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: Optional[str] = None,
) -> dict:
    """Build common inference kwargs including env-var overrides."""
    kwargs: dict = {}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if quality is not None:
        kwargs["quality"] = quality

    from diffgentor.utils.env import get_env_float, get_env_int

    steps = get_env_int("SERVE_NUM_INFERENCE_STEPS", 0)
    if steps > 0:
        kwargs["num_inference_steps"] = steps
    cfg = get_env_float("SERVE_GUIDANCE_SCALE", 0.0)
    if cfg > 0:
        kwargs["guidance_scale"] = cfg

    return kwargs


# ---------------------------------------------------------------------------
# POST /v1/images/generations
# ---------------------------------------------------------------------------


@router.post("/v1/images/generations", response_model=ImagesResponse)
async def generate_images(req: ImageGenerateRequest):
    if _t2i_backend is None and not (_worker_pool and _serve_mode == "t2i"):
        raise HTTPException(
            status_code=400,
            detail="This server is not configured for image generation (T2I). "
            "Start the server with --mode t2i to enable this endpoint.",
        )

    width, height = _parse_size(req.size)
    n = max(req.n or 1, 1)
    output_format = (req.output_format or "png").lower()

    kwargs = _build_inference_kwargs(width, height, req.quality)

    try:
        all_images: List[Image.Image] = []
        for _ in range(n):
            if _worker_pool is not None:
                images = await _worker_pool.submit_generate(req.prompt, **kwargs)
            else:
                async with _gpu_semaphore:
                    images = await asyncio.to_thread(
                        _t2i_backend.generate,
                        req.prompt,
                        **kwargs,
                    )
            all_images.extend(images)

        data = [ImageData(b64_json=_pil_to_b64(img, output_format)) for img in all_images]
        return ImagesResponse(created=int(time.time()), data=data)

    except Exception as e:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /v1/images/edits
# ---------------------------------------------------------------------------


@router.post("/v1/images/edits", response_model=ImagesResponse)
async def edit_images(
    image: List[UploadFile] = File(...),
    prompt: str = Form(...),
    model: Optional[str] = Form(None),
    n: Optional[int] = Form(1),
    size: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    response_format: Optional[str] = Form("b64_json"),
    output_format: Optional[str] = Form("png"),
    mask: Optional[UploadFile] = File(None),
):
    if _editing_backend is None and not (_worker_pool and _serve_mode == "edit"):
        raise HTTPException(
            status_code=400,
            detail="This server is not configured for image editing. "
            "Start the server with --mode edit to enable this endpoint.",
        )

    out_fmt = (output_format or "png").lower()
    n = max(n or 1, 1)

    pil_images: List[Image.Image] = []
    for upload in image:
        pil_images.append(await _read_upload_as_pil(upload))

    width, height = _parse_size(size)
    kwargs = _build_inference_kwargs(width, height, quality)

    try:
        all_images: List[Image.Image] = []
        input_imgs = pil_images if len(pil_images) > 1 else pil_images[0]
        for _ in range(n):
            if _worker_pool is not None:
                edited = await _worker_pool.submit_edit(input_imgs, prompt, **kwargs)
            else:
                async with _gpu_semaphore:
                    edited = await asyncio.to_thread(
                        _editing_backend.edit,
                        input_imgs,
                        prompt,
                        **kwargs,
                    )
            all_images.extend(edited)

        data = [ImageData(b64_json=_pil_to_b64(img, out_fmt)) for img in all_images]
        return ImagesResponse(created=int(time.time()), data=data)

    except Exception as e:
        logger.exception("Image editing failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    return ModelListResponse(
        data=[
            ModelObject(id=_model_name, created=0, owned_by="diffgentor"),
        ]
    )


@router.get("/v1/models/{model_id}", response_model=ModelObject)
async def retrieve_model(model_id: str):
    if model_id != _model_name:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelObject(id=_model_name, created=0, owned_by="diffgentor")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health():
    backend_ready = (
        _t2i_backend is not None
        or _editing_backend is not None
        or _worker_pool is not None
    )
    return HealthResponse(
        status="ok",
        mode=_serve_mode,
        model=_model_name or None,
        backend_ready=backend_ready,
    )
