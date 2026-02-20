# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""OpenAI-compatible API route handlers for diffgentor serve mode.

Supports both ``multipart/form-data`` (binary uploads) and
``application/json`` (image URLs / file IDs) for the edit endpoint,
matching the OpenAI Images API specification.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import TYPE_CHECKING, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image

from diffgentor.serve.schemas import (
    HealthResponse,
    ImageData,
    ImageEditJsonRequest,
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


async def _download_image_url(url: str) -> Image.Image:
    """Download an image from a URL, data-URL, or raw base64 string.

    Supported formats:
    - ``https://...`` or ``http://...`` — fetched via HTTP
    - ``data:image/png;base64,iVBOR...`` — base64 data-URL (RFC 2397)
    - raw base64 string (no prefix) — decoded directly
    """
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    if url.startswith(("http://", "https://")):
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # Treat as raw base64-encoded image bytes
    try:
        return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image_url: not a valid URL, data-URL, or base64 string")


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


def _form_str(value: object) -> Optional[str]:
    """Extract a string from a form field value, returning None for empty."""
    if value is None or isinstance(value, UploadFile):
        return None
    s = str(value)
    return s if s else None


def _form_int(value: object, default: int = 1) -> int:
    """Extract an int from a form field value."""
    if value is None:
        return default
    try:
        return int(str(value))
    except (ValueError, TypeError):
        return default


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
        return ImagesResponse(created=int(time.time()), data=data, output_format=output_format)

    except Exception as e:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /v1/images/edits
# ---------------------------------------------------------------------------


async def _parse_edit_multipart(request: Request) -> dict:
    """Parse a multipart/form-data edit request.

    Handles both ``image`` (single file) and ``image[]`` (multiple files)
    field names as used by the OpenAI API / curl clients.
    """
    form = await request.form()

    image_uploads: List[UploadFile] = []
    for key, value in form.multi_items():
        if key in ("image", "image[]") and isinstance(value, UploadFile):
            image_uploads.append(value)

    prompt = _form_str(form.get("prompt"))
    if not prompt:
        raise HTTPException(status_code=422, detail="'prompt' field is required")

    pil_images: List[Image.Image] = []
    for upload in image_uploads:
        pil_images.append(await _read_upload_as_pil(upload))

    mask_upload = form.get("mask")
    mask_image = None
    if isinstance(mask_upload, UploadFile):
        mask_image = await _read_upload_as_pil(mask_upload)

    return {
        "prompt": prompt,
        "pil_images": pil_images,
        "mask": mask_image,
        "model": _form_str(form.get("model")),
        "n": _form_int(form.get("n"), 1),
        "size": _form_str(form.get("size")),
        "quality": _form_str(form.get("quality")),
        "output_format": _form_str(form.get("output_format")) or "png",
        "output_compression": _form_int(form.get("output_compression"), 100) if form.get("output_compression") else None,
        "background": _form_str(form.get("background")),
        "input_fidelity": _form_str(form.get("input_fidelity")),
    }


async def _parse_edit_json(request: Request) -> dict:
    """Parse a JSON edit request with image references (URLs / file IDs)."""
    body = await request.json()
    req = ImageEditJsonRequest(**body)

    if not req.prompt:
        raise HTTPException(status_code=422, detail="'prompt' field is required")

    pil_images: List[Image.Image] = []
    if req.images:
        for ref in req.images:
            if ref.image_url:
                pil_images.append(await _download_image_url(ref.image_url))
            elif ref.file_id:
                raise HTTPException(
                    status_code=400,
                    detail="file_id references are not supported by this server. Use image_url instead.",
                )

    mask_image = None
    if req.mask and req.mask.image_url:
        mask_image = await _download_image_url(req.mask.image_url)

    return {
        "prompt": req.prompt,
        "pil_images": pil_images,
        "mask": mask_image,
        "model": req.model,
        "n": max(req.n or 1, 1),
        "size": req.size,
        "quality": req.quality,
        "output_format": req.output_format or "png",
        "output_compression": req.output_compression,
        "background": req.background,
        "input_fidelity": req.input_fidelity,
    }


@router.post("/v1/images/edits", response_model=ImagesResponse)
async def edit_images(request: Request):
    """Handle image edit requests.

    Accepts both ``multipart/form-data`` (binary uploads via ``image`` /
    ``image[]``) and ``application/json`` (image references via ``images``
    array with ``image_url`` or ``file_id``).
    """
    if _editing_backend is None and not (_worker_pool and _serve_mode == "edit"):
        raise HTTPException(
            status_code=400,
            detail="This server is not configured for image editing. "
            "Start the server with --mode edit to enable this endpoint.",
        )

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        params = await _parse_edit_json(request)
    else:
        params = await _parse_edit_multipart(request)

    pil_images: List[Image.Image] = params["pil_images"]
    prompt: str = params["prompt"]
    n: int = max(params["n"], 1)
    out_fmt: str = params["output_format"].lower()

    if not pil_images:
        raise HTTPException(status_code=422, detail="At least one image is required")

    width, height = _parse_size(params["size"])
    kwargs = _build_inference_kwargs(width, height, params["quality"])

    try:
        all_images: List[Image.Image] = []
        input_imgs: Union[Image.Image, List[Image.Image]] = pil_images if len(pil_images) > 1 else pil_images[0]
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
        return ImagesResponse(created=int(time.time()), data=data, output_format=out_fmt)

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
