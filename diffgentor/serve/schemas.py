# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Pydantic models matching the OpenAI Images API request/response types.

Reference: https://platform.openai.com/docs/api-reference/images
"""

from typing import List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ImageData(BaseModel):
    """Single image entry inside an ImagesResponse."""

    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    url: Optional[str] = None


class ImageGenUsage(BaseModel):
    """Token usage for GPT image models (optional for local backends)."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ImagesResponse(BaseModel):
    """Response format returned by /images/generations and /images/edits."""

    created: int
    data: List[ImageData] = Field(default_factory=list)
    background: Optional[str] = None
    output_format: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    usage: Optional[ImageGenUsage] = None


# ---------------------------------------------------------------------------
# Request models — /v1/images/generations (JSON body)
# ---------------------------------------------------------------------------


class ImageGenerateRequest(BaseModel):
    """JSON body for POST /v1/images/generations."""

    prompt: str
    model: Optional[str] = None
    n: Optional[int] = 1
    size: Optional[str] = None
    quality: Optional[str] = None
    response_format: Optional[str] = "b64_json"
    output_format: Optional[str] = "png"
    output_compression: Optional[int] = None
    background: Optional[str] = None
    moderation: Optional[str] = None
    style: Optional[str] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Request models — /v1/images/edits (JSON body)
# ---------------------------------------------------------------------------


class ImageRefParam(BaseModel):
    """Reference an input image by URL or file ID (JSON edit requests)."""

    image_url: Optional[str] = None
    file_id: Optional[str] = None


class ImageEditJsonRequest(BaseModel):
    """JSON body for POST /v1/images/edits.

    Used when the client sends ``application/json`` instead of
    ``multipart/form-data``.  Images are referenced via ``images`` (array of
    ``ImageRefParam``) rather than binary uploads.
    """

    prompt: str
    images: Optional[List[ImageRefParam]] = None
    mask: Optional[ImageRefParam] = None
    model: Optional[str] = None
    n: Optional[int] = 1
    size: Optional[str] = None
    quality: Optional[str] = None
    response_format: Optional[str] = "b64_json"
    output_format: Optional[str] = "png"
    output_compression: Optional[int] = None
    background: Optional[str] = None
    input_fidelity: Optional[str] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


class ModelObject(BaseModel):
    """Single model entry for GET /v1/models."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "diffgentor"


class ModelListResponse(BaseModel):
    """Response for GET /v1/models."""

    object: str = "list"
    data: List[ModelObject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health / Error
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    mode: Optional[str] = None
    model: Optional[str] = None
    backend_ready: bool = False


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


# ---------------------------------------------------------------------------
# LoRA management
# ---------------------------------------------------------------------------


class SetLoraRequest(BaseModel):
    """Request body for POST /v1/set_lora.

    Follows the SGLang convention.  Supports both single and multiple LoRA
    adapters.  When lists are provided for ``lora_nickname``, ``lora_path``,
    ``target``, and ``strength``, all adapters are loaded and activated
    simultaneously.  Scalar values are broadcast to all adapters.
    """

    lora_nickname: Union[str, List[str]]
    lora_path: Union[str, List[str], None] = None
    target: Union[str, List[str], None] = None
    strength: Union[float, List[float]] = 1.0


class MergeLoraRequest(BaseModel):
    """Request body for POST /v1/merge_lora_weights."""

    target: Optional[str] = None
    strength: float = 1.0


class LoadedAdapterInfo(BaseModel):
    """Entry in the ``loaded_adapters`` list of ``ListLorasResponse``."""

    nickname: str
    path: str


class ActiveLoraInfo(BaseModel):
    """Per-adapter detail inside the ``active`` dict of ``ListLorasResponse``."""

    nickname: str
    path: str
    merged: bool = False
    strength: float = 1.0


class ListLorasResponse(BaseModel):
    """Response for GET /v1/list_loras.

    Matches the SGLang response format::

        {
          "loaded_adapters": [{"nickname": "...", "path": "..."}],
          "active": {
            "all": [{"nickname": "...", "path": "...", "merged": true, "strength": 1.0}]
          }
        }
    """

    loaded_adapters: List[LoadedAdapterInfo] = Field(default_factory=list)
    active: dict = Field(default_factory=dict)


class LoraActionResponse(BaseModel):
    """Generic response for LoRA mutation endpoints."""

    status: str = "ok"
    message: str = ""
