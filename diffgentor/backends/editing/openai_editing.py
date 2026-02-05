# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""OpenAI editing backend using images.edit API.

Features:
- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
"""

import io
import os
from typing import List, Optional, Tuple, Union

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
    b64_to_image,
    image_to_bytes,
)


class OpenAIEditingClientPool(APIClientPool):
    """OpenAI API client pool for image editing."""

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
        image_data: Union[bytes, List[bytes]],
        prompt: str,
        size: str = "1024x1024",
        **kwargs,
    ) -> List[Image.Image]:
        """Execute image editing request.

        Args:
            client: OpenAI client
            image_data: Single image bytes or list of image bytes for multi-image input
            prompt: Editing instruction
            size: Output size

        Returns:
            List of edited images
        """
        # Convert bytes to file-like objects
        if isinstance(image_data, list):
            # Multi-image mode (gpt-image-1.5+)
            image_files = [io.BytesIO(img_bytes) for img_bytes in image_data]
            response = client.images.edit(
                model=self.model,
                image=image_files,
                prompt=prompt,
                size=size,
                response_format="b64_json",
            )
        else:
            # Single image mode
            response = client.images.edit(
                model=self.model,
                image=image_data,
                prompt=prompt,
                size=size,
                response_format="b64_json",
            )

        # Decode results
        images = []
        for data in response.data:
            if hasattr(data, "b64_json") and data.b64_json:
                images.append(b64_to_image(data.b64_json))

        return images


class OpenAIEditingBackend(BaseEditingBackend):
    """OpenAI backend using images.edit API for image editing.

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
        """Initialize OpenAI editing backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self.pool: Optional[OpenAIEditingClientPool] = None
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
        self.pool = OpenAIEditingClientPool(pool_config, self._model)

        self._initialized = True

        # Log configuration
        num_endpoints = len(pool_config.endpoints)
        total_keys = sum(len(ep.api_keys) for ep in pool_config.endpoints)
        print_rank0(f"OpenAI editing pool initialized: model={self._model}, "
                    f"endpoints={num_endpoints}, keys={total_keys}, "
                    f"timeout={pool_config.timeout}s, max_retries={pool_config.max_retries}")

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        size: str = "1024x1024",
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using OpenAI API.

        Args:
            images: Input image(s) to edit. Supports multiple images for gpt-image-1.5+
            instruction: Editing instruction
            size: Output image size
            **kwargs: Additional API arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize to list
        if isinstance(images, Image.Image):
            images = [images]

        if not images:
            raise ValueError("No input image provided")

        # Convert images to bytes
        if len(images) == 1:
            # Single image - pass as bytes directly
            img_data = image_to_bytes(images[0])
        else:
            # Multiple images - pass as list of bytes
            img_data = [image_to_bytes(img) for img in images]

        # Execute with retry
        try:
            result = self.pool.execute_with_retry(img_data, instruction, size, **kwargs)
            return result
        except Exception as e:
            print_rank0(f"Edit failed: {e}")
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
            if not images:
                continue

            # Support multi-image input
            if len(images) == 1:
                img_data = image_to_bytes(images[0])
            else:
                img_data = [image_to_bytes(img) for img in images]

            size = kwargs.get("size", "1024x1024")
            batch_items.append(((img_data, instruction, size), {}))
            indices.append(idx)

        # Execute batch
        raw_results = self.pool.execute_batch(batch_items, max_workers=max_workers)

        # Process results
        results = []
        for i, (batch_idx, result, error) in enumerate(raw_results):
            idx = indices[batch_idx]
            if error:
                print_rank0(f"Failed to edit index {idx}: {error}")
                results.append((idx, None))
            elif result:
                results.append((idx, result[0] if result else None))
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
