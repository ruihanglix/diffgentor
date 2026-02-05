# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""OpenAI backend implementation for T2I generation via API.

Features:
- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
"""

import os
from typing import List, Optional, Union

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
    b64_to_image,
)


class OpenAIClientPool(APIClientPool):
    """OpenAI API client pool for T2I generation."""

    def __init__(self, config: PoolConfig, model: str):
        super().__init__(config)
        self.model = model

    def _create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        api_version: Optional[str],
        timeout: float,
    ):
        from openai import OpenAI

        client_kwargs = {"timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        return OpenAI(**client_kwargs)

    def _execute_request(
        self,
        client,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        **kwargs,
    ) -> List[Image.Image]:
        """Execute image generation request."""
        api_kwargs = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
        }

        # Add optional parameters
        if "quality" in kwargs:
            api_kwargs["quality"] = kwargs["quality"]

        response = client.images.generate(**api_kwargs)

        # Decode images
        images = []
        for data in response.data:
            if hasattr(data, "b64_json") and data.b64_json:
                images.append(b64_to_image(data.b64_json))

        return images


class OpenAIBackend(BaseBackend):
    """OpenAI backend using images.generate API.

    Supports:
    - Multiple API endpoints with load balancing
    - Multiple API keys per endpoint
    - Configurable timeout (default: 5 minutes)
    - Configurable retry with exponential backoff (default: no retry)

    Environment variables:
        # Single endpoint
        OPENAI_API_KEY: API key
        OPENAI_API_BASE: Base URL (optional)
        DG_OPENAI_RATE_LIMIT: Rate limit per minute

        # Multiple endpoints
        DG_OPENAI_ENDPOINTS: Comma-separated base URLs
        DG_OPENAI_API_KEYS: Comma-separated API keys
        DG_OPENAI_RATE_LIMITS: Comma-separated rate limits
        DG_OPENAI_WEIGHTS: Comma-separated weights

        # Pool settings
        DG_OPENAI_TIMEOUT: Timeout in seconds (default: 300)
        DG_OPENAI_MAX_RETRIES: Max retry attempts (default: 0)
        DG_OPENAI_RETRY_DELAY: Initial retry delay (default: 1.0)
        DG_OPENAI_MAX_WORKERS: Max concurrent workers (default: 4)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize OpenAI backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration (mostly unused for API)
        """
        super().__init__(backend_config, optimization_config)
        self.pool: Optional[OpenAIClientPool] = None
        self._model = None

    def load_model(self, **kwargs) -> None:
        """Initialize OpenAI client pool.

        Args:
            **kwargs: Additional arguments
                - api_key: API key override
                - api_base/base_url: Base URL override
                - timeout: Timeout override
                - max_retries: Max retries override
                - max_workers: Max workers override
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI backend. "
                "Install with: pip install openai"
            )

        self._model = self.model_name or "gpt-image-1"

        # Parse pool config from environment
        pool_config = parse_pool_config_from_env("OPENAI", "OPENAI_API_KEY")

        # Apply CLI kwargs overrides (timeout, max_retries, retry_delay, max_workers)
        pool_config = apply_pool_kwargs(pool_config, **kwargs)

        # Handle single key/url override from kwargs or config
        api_key = kwargs.get("api_key") or self.backend_config.openai_api_key
        api_base = kwargs.get("api_base") or kwargs.get("base_url") or self.backend_config.openai_api_base

        if not pool_config.endpoints and (api_key or api_base):
            pool_config.endpoints = [EndpointConfig(
                base_url=api_base,
                api_keys=[api_key] if api_key else [],
            )]

        if not pool_config.endpoints:
            # Try default env var
            default_key = os.environ.get("OPENAI_API_KEY")
            if default_key:
                pool_config.endpoints = [EndpointConfig(api_keys=[default_key])]

        if not pool_config.endpoints or not any(ep.api_keys for ep in pool_config.endpoints):
            raise ValueError(
                "No OpenAI API key configured. Set OPENAI_API_KEY or DG_OPENAI_API_KEYS."
            )

        # Create pool
        self.pool = OpenAIClientPool(pool_config, self._model)

        self._initialized = True

        # Log configuration
        num_endpoints = len(pool_config.endpoints)
        total_keys = sum(len(ep.api_keys) for ep in pool_config.endpoints)
        print_rank0(f"OpenAI client pool initialized: model={self._model}, "
                    f"endpoints={num_endpoints}, keys={total_keys}, "
                    f"timeout={pool_config.timeout}s, max_retries={pool_config.max_retries}")

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,  # Not used by OpenAI API
        guidance_scale: Optional[float] = None,  # Not used by OpenAI API
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,  # Not directly supported
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images using OpenAI API.

        Args:
            prompt: Single prompt or list of prompts
            negative_prompt: Not supported by OpenAI API
            height: Image height (will use closest supported size)
            width: Image width (will use closest supported size)
            num_inference_steps: Not used by OpenAI API
            guidance_scale: Not used by OpenAI API
            num_images_per_prompt: Number of images per prompt
            seed: Not directly supported by OpenAI API
            max_workers: Override max concurrent workers
            **kwargs: Additional arguments passed to API

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

        # Determine size
        size = self._get_size_string(width, height)

        # Build batch items
        batch_items = []
        for p in prompts:
            for _ in range(num_images_per_prompt):
                batch_items.append((p, 1, size))

        # Execute batch
        results = self.pool.execute_batch(
            [(item, kwargs) for item in batch_items],
            max_workers=max_workers,
        )

        # Collect images
        all_images = []
        for idx, result, error in results:
            if error:
                print_rank0(f"Failed to generate image {idx}: {error}")
            elif result:
                all_images.extend(result)

        return all_images

    def _get_size_string(
        self,
        width: Optional[int],
        height: Optional[int],
    ) -> str:
        """Convert width/height to OpenAI size string.

        Args:
            width: Requested width
            height: Requested height

        Returns:
            Size string like "1024x1024"
        """
        # Supported sizes for different models may vary
        # Common sizes: 256x256, 512x512, 1024x1024, 1792x1024, 1024x1792
        supported_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (1792, 1024),
            (1024, 1792),
        ]

        if width is None and height is None:
            return "1024x1024"

        target_w = width or 1024
        target_h = height or 1024

        # Find closest supported size
        best_size = min(
            supported_sizes,
            key=lambda s: abs(s[0] - target_w) + abs(s[1] - target_h),
        )

        return f"{best_size[0]}x{best_size[1]}"

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self.pool = None
        self._model = None
