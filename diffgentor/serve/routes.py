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
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from diffgentor.serve.schemas import (
    ErrorDetail,
    ErrorResponse,
    ImageData,
    ImageGenerateRequest,
    ImagesResponse,
    ModelListResponse,
    ModelObject,
)

logger = logging.getLogger("diffgentor.serve")

router = APIRouter()

# These are set by app.py at startup
_t2i_backend = None
_editing_backend = None
_model_name: str = ""
_gpu_semaphore: Optional[asyncio.Semaphore] = None


def configure(
    *,
    t2i_backend=None,
    editing_backend=None,
    model_name: str = "",
    max_concurrent: int = 1,
) -> None:
    """Inject backend instances and settings into the router.

    Called once by app.py after the model has been loaded.
    """
    global _t2i_backend, _editing_backend, _model_name, _gpu_semaphore
    _t2i_backend = t2i_backend
    _editing_backend = editing_backend
    _model_name = model_name
    _gpu_semaphore = asyncio.Semaphore(max_concurrent)


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


# ---------------------------------------------------------------------------
# POST /v1/images/generations
# ---------------------------------------------------------------------------


@router.post("/v1/images/generations", response_model=ImagesResponse)
async def generate_images(req: ImageGenerateRequest):
    if _t2i_backend is None:
        raise HTTPException(
            status_code=400,
            detail="This server is not configured for image generation (T2I). "
            "Start the server with --mode t2i to enable this endpoint.",
        )

    width, height = _parse_size(req.size)
    n = max(req.n or 1, 1)
    output_format = (req.output_format or "png").lower()

    kwargs: dict = {}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if req.quality is not None:
        kwargs["quality"] = req.quality

    from diffgentor.utils.env import get_env_int, get_env_float

    steps = get_env_int("SERVE_NUM_INFERENCE_STEPS", 0)
    if steps > 0:
        kwargs["num_inference_steps"] = steps
    cfg = get_env_float("SERVE_GUIDANCE_SCALE", 0.0)
    if cfg > 0:
        kwargs["guidance_scale"] = cfg

    try:
        all_images: List[Image.Image] = []
        for _ in range(n):
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
    if _editing_backend is None:
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

    kwargs: dict = {}
    if quality is not None:
        kwargs["quality"] = quality

    width, height = _parse_size(size)
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height

    from diffgentor.utils.env import get_env_int, get_env_float

    steps = get_env_int("SERVE_NUM_INFERENCE_STEPS", 0)
    if steps > 0:
        kwargs["num_inference_steps"] = steps
    cfg = get_env_float("SERVE_GUIDANCE_SCALE", 0.0)
    if cfg > 0:
        kwargs["guidance_scale"] = cfg

    try:
        all_images: List[Image.Image] = []
        input_imgs = pil_images if len(pil_images) > 1 else pil_images[0]
        for _ in range(n):
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
