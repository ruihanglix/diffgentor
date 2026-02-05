# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Google GenAI backend for T2I generation using Gemini native image models.

Features:
- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
- Rate limiting per endpoint
"""

import base64
import io
import os
import re
from typing import List, Optional, Tuple, Union

from PIL import Image

from diffgentor.backends.base import BaseBackend
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
    """Normalize base_url/api_version for google-genai HttpOptions.

    Some proxies expose versioned base URLs like 'http://host/v1beta'.
    The google-genai SDK typically appends api_version itself.
    """
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

        # Check candidates
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

        # Check parts directly
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

        # Text-only fallback
        for txt in collected_texts:
            b = _try_extract_data_url_image_bytes(txt)
            if b:
                return b
    except Exception:
        return None
    return None


class GoogleGenAIClientPool(APIClientPool):
    """Google GenAI API client pool for T2I generation."""

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
        prompt: str,
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> Optional[Image.Image]:
        """Execute image generation request."""
        from google.genai import types

        # Build config
        config_kwargs = {"response_modalities": ["IMAGE"]}
        ar = aspect_ratio or self.default_aspect_ratio
        if ar:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=ar)
        config = types.GenerateContentConfig(**config_kwargs)

        # Call API
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        # Extract image
        img_bytes = _extract_image_bytes_from_response(response)
        if img_bytes:
            return bytes_to_image(img_bytes)
        return None


class GoogleGenAIBackend(BaseBackend):
    """Google GenAI backend using Gemini native image models.

    Supports:
    - Multiple API endpoints with load balancing
    - Multiple API keys per endpoint
    - Configurable timeout (default: 5 minutes)
    - Configurable retry with exponential backoff (default: no retry)

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
        """Initialize Google GenAI backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self.pool: Optional[GoogleGenAIClientPool] = None
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
        self.pool = GoogleGenAIClientPool(pool_config, self._model, default_aspect_ratio)

        self._initialized = True

        # Log configuration
        num_endpoints = len(pool_config.endpoints)
        total_keys = sum(len(ep.api_keys) for ep in pool_config.endpoints)
        print_rank0(f"Google GenAI pool initialized: model={self._model}, "
                    f"endpoints={num_endpoints}, keys={total_keys}, "
                    f"timeout={pool_config.timeout}s, max_retries={pool_config.max_retries}")

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,  # Not used
        guidance_scale: Optional[float] = None,  # Not used
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,  # Not directly supported
        max_workers: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images using Google GenAI.

        Args:
            prompt: Single prompt or list of prompts
            negative_prompt: Not supported by Gemini
            height: Image height (converted to aspect_ratio)
            width: Image width (converted to aspect_ratio)
            num_inference_steps: Not used by Gemini
            guidance_scale: Not used by Gemini
            num_images_per_prompt: Number of images per prompt
            seed: Not directly supported
            max_workers: Override max concurrent workers
            aspect_ratio: Aspect ratio string (e.g., "9:16", "16:9", "1:1")
            **kwargs: Additional arguments

        Returns:
            List of generated PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize prompts to list
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        # Determine aspect ratio
        if aspect_ratio is None and width and height:
            aspect_ratio = self._size_to_aspect_ratio(width, height)

        # Build batch items
        batch_items = []
        for p in prompts:
            for _ in range(num_images_per_prompt):
                batch_items.append(((p,), {"aspect_ratio": aspect_ratio}))

        # Execute batch
        results = self.pool.execute_batch(batch_items, max_workers=max_workers)

        # Collect images
        all_images = []
        for idx, result, error in results:
            if error:
                print_rank0(f"Failed to generate image {idx}: {error}")
            elif result:
                all_images.append(result)

        return all_images

    def _size_to_aspect_ratio(self, width: int, height: int) -> str:
        """Convert width/height to aspect ratio string.

        Args:
            width: Image width
            height: Image height

        Returns:
            Aspect ratio string like "16:9"
        """
        from math import gcd

        g = gcd(width, height)
        w_ratio = width // g
        h_ratio = height // g

        # Simplify common ratios
        common_ratios = {
            (16, 9): "16:9",
            (9, 16): "9:16",
            (4, 3): "4:3",
            (3, 4): "3:4",
            (1, 1): "1:1",
            (3, 2): "3:2",
            (2, 3): "2:3",
        }

        if (w_ratio, h_ratio) in common_ratios:
            return common_ratios[(w_ratio, h_ratio)]

        return f"{w_ratio}:{h_ratio}"

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self.pool = None
        self._model = None
