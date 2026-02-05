# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Google GenAI editing backend using Gemini native image models.

Features:
- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
- Multimodal input support (instruction + multiple images)
"""

import base64
import os
import re
from typing import Any, List, Optional, Tuple, Union

from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.api_pool import (
    APIClientPool,
    PoolConfig,
    EndpointConfig,
    parse_pool_config_from_env,
    apply_pool_kwargs,
    bytes_to_image,
)

# Regex for extracting data URL images from text responses
_DATA_URL_IMAGE_RE = re.compile(
    r"data:image/(?P<fmt>png|jpe?g|webp);base64,(?P<b64>[A-Za-z0-9+/=\s]+)",
    re.IGNORECASE,
)


def _normalize_base_url_and_version(
    base_url: Optional[str],
    api_version: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize base_url/api_version for google-genai HttpOptions."""
    if not base_url:
        return None, api_version

    b = base_url.strip().rstrip("/")
    for suffix in ("/v1beta", "/v1alpha", "/v1"):
        if b.endswith(suffix):
            inferred = suffix[1:]
            b = b[: -len(suffix)]
            if api_version is None:
                api_version = inferred
            break
    return b, api_version


def _try_extract_data_url_image_bytes(text: str) -> Optional[bytes]:
    """Extract images embedded in text as data URLs."""
    if not text:
        return None
    m = _DATA_URL_IMAGE_RE.search(text)
    if not m:
        return None
    b64 = m.group("b64")
    b64 = "".join(b64.split())
    try:
        return base64.b64decode(b64, validate=False)
    except Exception:
        return None


def _extract_image_bytes_from_response(response) -> Optional[bytes]:
    """Extract image bytes from google-genai response."""
    try:
        collected_texts: List[str] = []

        if getattr(response, "candidates", None):
            for cand in response.candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) if content else None
                if not parts:
                    continue
                for part in parts:
                    if getattr(part, "text", None):
                        collected_texts.append(str(part.text))
                    inline = getattr(part, "inline_data", None)
                    if not inline:
                        continue
                    data = getattr(inline, "data", None)
                    if data is None:
                        continue
                    if isinstance(data, str):
                        return base64.b64decode(data)
                    return data

        if getattr(response, "parts", None):
            for part in response.parts:
                if getattr(part, "text", None):
                    collected_texts.append(str(part.text))
                inline = getattr(part, "inline_data", None)
                if not inline:
                    continue
                data = getattr(inline, "data", None)
                if data is None:
                    continue
                if isinstance(data, str):
                    return base64.b64decode(data)
                return data

        for txt in collected_texts:
            b = _try_extract_data_url_image_bytes(txt)
            if b:
                return b
    except Exception:
        return None
    return None


class GoogleGenAIEditingClientPool(APIClientPool):
    """Google GenAI API client pool for image editing."""

    def __init__(self, config: PoolConfig, model: str, default_aspect_ratio: Optional[str] = None):
        super().__init__(config)
        self.model = model
        self.default_aspect_ratio = default_aspect_ratio

    def _create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        api_version: Optional[str],
        timeout: float,
    ):
        from google import genai
        from google.genai import types

        # Normalize URL/version
        base_url, api_version = _normalize_base_url_and_version(base_url, api_version)

        # Create HTTP options if needed
        http_options = None
        if base_url or api_version:
            http_options = types.HttpOptions(base_url=base_url, api_version=api_version)

        return genai.Client(api_key=api_key, http_options=http_options)

    def _execute_request(
        self,
        client,
        instruction: str,
        images: List[Image.Image],
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Execute image editing request with multimodal input."""
        from google.genai import types

        # Build config
        config_kwargs = {"response_modalities": ["IMAGE"]}
        ar = aspect_ratio or self.default_aspect_ratio
        if ar:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=ar)
        config = types.GenerateContentConfig(**config_kwargs)

        # Build contents: instruction first, then images
        # The google-genai SDK accepts PIL Images directly
        contents: List[Any] = [instruction] + list(images)

        # Call API
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Extract image
        img_bytes = _extract_image_bytes_from_response(response)
        if img_bytes:
            return bytes_to_image(img_bytes)
        return None


class GoogleGenAIEditingBackend(BaseEditingBackend):
    """Google GenAI backend for image editing using Gemini native image models.

    Supports:
    - Multiple API endpoints with load balancing
    - Multiple API keys per endpoint
    - Configurable timeout (default: 5 minutes)
    - Configurable retry with exponential backoff (default: no retry)
    - Multimodal input: [instruction, image1, image2, ...]

    Supported models:
    - gemini-2.5-flash-image (Nano Banana)
    - gemini-3-pro-image-preview (Nano Banana Pro)

    Environment variables:
        # Single endpoint
        GEMINI_API_KEY: API key
        DG_GEMINI_BASE_URL: Base URL (optional)
        DG_GEMINI_API_VERSION: API version (v1, v1beta, v1alpha)
        DG_GEMINI_RATE_LIMIT: Rate limit per minute
        DG_GEMINI_ASPECT_RATIO: Default aspect ratio

        # Multiple endpoints
        DG_GEMINI_ENDPOINTS: Comma-separated base URLs
        DG_GEMINI_API_KEYS: Comma-separated API keys
        DG_GEMINI_RATE_LIMITS: Comma-separated rate limits
        DG_GEMINI_WEIGHTS: Comma-separated weights

        # Pool settings
        DG_GEMINI_TIMEOUT: Timeout in seconds (default: 300)
        DG_GEMINI_MAX_RETRIES: Max retry attempts (default: 0)
        DG_GEMINI_RETRY_DELAY: Initial retry delay (default: 1.0)
        DG_GEMINI_MAX_WORKERS: Max concurrent workers (default: 4)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize Google GenAI editing backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self.pool: Optional[GoogleGenAIEditingClientPool] = None
        self._model = None

    def load_model(self, **kwargs) -> None:
        """Initialize Google GenAI client pool.

        Args:
            **kwargs: Additional arguments
                - api_key: API key override
                - base_url: Base URL override
                - api_version: API version override
                - timeout: Timeout override
                - max_retries: Max retries override
                - max_workers: Max workers override
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai is required for Google GenAI backend. "
                "Install with: pip install google-genai"
            )

        self._model = self.model_name or "gemini-2.5-flash-image"

        # Parse pool config from environment
        pool_config = parse_pool_config_from_env("GEMINI", "GEMINI_API_KEY")

        # Apply CLI kwargs overrides (timeout, max_retries, retry_delay, max_workers)
        pool_config = apply_pool_kwargs(pool_config, **kwargs)

        # Handle single key/url override from kwargs
        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        api_version = kwargs.get("api_version")

        if not pool_config.endpoints and api_key:
            pool_config.endpoints = [EndpointConfig(
                base_url=base_url,
                api_keys=[api_key],
                api_version=api_version,
            )]

        if not pool_config.endpoints:
            # Try default env var
            default_key = os.environ.get("GEMINI_API_KEY")
            if default_key:
                pool_config.endpoints = [EndpointConfig(api_keys=[default_key])]

        if not pool_config.endpoints or not any(ep.api_keys for ep in pool_config.endpoints):
            raise ValueError(
                "No Gemini API key configured. Set GEMINI_API_KEY or DG_GEMINI_API_KEYS."
            )

        # Get default aspect ratio
        default_aspect_ratio = (
            kwargs.get("aspect_ratio")
            or os.environ.get("DG_GEMINI_ASPECT_RATIO")
        )

        # Create pool
        self.pool = GoogleGenAIEditingClientPool(pool_config, self._model, default_aspect_ratio)

        self._initialized = True

        # Log configuration
        num_endpoints = len(pool_config.endpoints)
        total_keys = sum(len(ep.api_keys) for ep in pool_config.endpoints)
        print_rank0(f"Google GenAI editing pool initialized: model={self._model}, "
                    f"endpoints={num_endpoints}, keys={total_keys}, "
                    f"timeout={pool_config.timeout}s, max_retries={pool_config.max_retries}")

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,  # Not used
        guidance_scale: Optional[float] = None,  # Not used
        seed: Optional[int] = None,  # Not supported
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using Google GenAI.

        The instruction and images are sent as multimodal content:
        [instruction, image1, image2, ...]

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Not used by Gemini
            guidance_scale: Not used by Gemini
            seed: Not supported by Gemini
            aspect_ratio: Aspect ratio string (e.g., "9:16")
            **kwargs: Additional arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize images to list
        if isinstance(images, Image.Image):
            images = [images]

        # Convert images to RGB
        images = [img.convert("RGB") for img in images]

        # Execute with retry
        try:
            result = self.pool.execute_with_retry(instruction, images, aspect_ratio, **kwargs)
            if result is not None:
                return [result]
            return []
        except Exception as e:
            print_rank0(f"Gemini editing failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def edit_batch(
        self,
        batch_data: List[Tuple[List[Image.Image], str, int]],
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[int, Optional[Image.Image]]]:
        """Edit a batch of images in parallel.

        Args:
            batch_data: List of (images, instruction, index) tuples
            max_workers: Override max concurrent workers
            **kwargs: Additional editing arguments

        Returns:
            List of (index, edited_image) tuples
        """
        # Build batch items
        batch_items = []
        indices = []
        for images, instruction, idx in batch_data:
            # Normalize and convert images
            if isinstance(images, Image.Image):
                images = [images]
            images = [img.convert("RGB") for img in images]

            aspect_ratio = kwargs.get("aspect_ratio")
            batch_items.append(((instruction, images, aspect_ratio), {}))
            indices.append(idx)

        # Execute batch
        raw_results = self.pool.execute_batch(batch_items, max_workers=max_workers)

        # Process results
        results = []
        for batch_idx, result, error in raw_results:
            idx = indices[batch_idx]
            if error:
                print_rank0(f"Failed to edit index {idx}: {error}")
                results.append((idx, None))
            elif result:
                results.append((idx, result))
            else:
                results.append((idx, None))

        # Sort by index
        results.sort(key=lambda x: x[0])
        return results

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self.pool = None
        self._model = None
