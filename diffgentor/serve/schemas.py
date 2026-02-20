# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Pydantic models matching the OpenAI Images API request/response types."""

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ImageData(BaseModel):
    """Single image entry inside an ImagesResponse."""

    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None
    url: Optional[str] = None


class ImagesResponse(BaseModel):
    """Response format returned by /images/generations and /images/edits."""

    created: int
    data: List[ImageData] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Request models
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
    background: Optional[str] = None
    style: Optional[str] = None


# ImageEditRequest is not a Pydantic model because the OpenAI client sends
# multipart/form-data.  We parse it directly from FastAPI Form/File params
# in the route handler.


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
# Error response
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
