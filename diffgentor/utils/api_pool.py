# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""API client pool with multi-endpoint, multi-key, timeout and retry support.

Features:
- Multiple API endpoints with load balancing
- Multiple API keys per endpoint (or shared across endpoints)
- Thread-safe round-robin key/endpoint selection
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Rate limiting per endpoint
"""

import base64
import io
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, Tuple, TypeVar

from PIL import Image

T = TypeVar("T")


@dataclass
class EndpointConfig:
    """Configuration for a single API endpoint."""

    base_url: Optional[str] = None  # None means default/official endpoint
    api_keys: List[str] = field(default_factory=list)
    rate_limit: int = 0  # requests per minute, 0 = no limit
    weight: int = 1  # load balancing weight
    api_version: Optional[str] = None  # for Google GenAI

    def __post_init__(self):
        if not self.api_keys:
            self.api_keys = []


@dataclass
class PoolConfig:
    """Configuration for API client pool."""

    endpoints: List[EndpointConfig] = field(default_factory=list)
    timeout: float = 300.0  # 5 minutes default
    max_retries: int = 0  # no retry by default
    retry_delay: float = 1.0  # initial retry delay in seconds
    retry_backoff: float = 2.0  # exponential backoff multiplier
    retry_max_delay: float = 60.0  # max retry delay
    max_global_workers: int = 16  # total concurrent workers across all processes
    num_processes: int = 4  # number of worker processes

    @property
    def workers_per_process(self) -> int:
        """Calculate threads per process."""
        return max(1, self.max_global_workers // self.num_processes)


class RateLimiter:
    """Thread-safe rate limiter using sliding window."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_times: deque = deque()
        self.lock = threading.Lock()
        self._enabled = max_requests > 0

    def acquire(self) -> float:
        """Block until a request slot is available. Returns total wait time."""
        if not self._enabled:
            return 0.0

        total_wait = 0.0
        while True:
            with self.lock:
                now = time.time()
                # Remove expired entries
                while self.request_times and now - self.request_times[0] >= self.window_seconds:
                    self.request_times.popleft()
                # Check if slot available
                if len(self.request_times) < self.max_requests:
                    self.request_times.append(now)
                    return total_wait
                # Calculate wait time
                wait_time = self.request_times[0] + self.window_seconds - now + 0.05
            if wait_time > 0:
                time.sleep(wait_time)
                total_wait += wait_time

    def get_available_slots(self) -> int:
        """Get number of available request slots."""
        if not self._enabled:
            return float('inf')
        with self.lock:
            now = time.time()
            while self.request_times and now - self.request_times[0] >= self.window_seconds:
                self.request_times.popleft()
            return self.max_requests - len(self.request_times)


class EndpointState:
    """State for a single endpoint including key rotation and rate limiting."""

    def __init__(self, config: EndpointConfig):
        self.config = config
        self.api_keys = list(config.api_keys) if config.api_keys else []
        self.key_index = 0
        self.key_lock = threading.Lock()
        self.rate_limiter = RateLimiter(config.rate_limit) if config.rate_limit > 0 else None
        self.error_count = 0
        self.last_error_time = 0.0

    def get_next_key(self) -> Optional[str]:
        """Get next API key in round-robin fashion."""
        if not self.api_keys:
            return None
        with self.key_lock:
            key = self.api_keys[self.key_index]
            self.key_index = (self.key_index + 1) % len(self.api_keys)
            return key

    def acquire_rate_limit(self) -> float:
        """Acquire rate limit slot. Returns wait time."""
        if self.rate_limiter:
            return self.rate_limiter.acquire()
        return 0.0

    def record_error(self):
        """Record an error for this endpoint."""
        self.error_count += 1
        self.last_error_time = time.time()

    def record_success(self):
        """Record a success, reducing error weight."""
        if self.error_count > 0:
            self.error_count = max(0, self.error_count - 1)


class APIClientPool(ABC, Generic[T]):
    """Abstract base class for API client pools.

    Subclasses must implement:
    - _create_client(base_url, api_key, api_version, timeout) -> client
    - _execute_request(client, *args, **kwargs) -> result
    """

    def __init__(self, config: PoolConfig):
        self.config = config
        self.endpoints: List[EndpointState] = []
        self.endpoint_index = 0
        self.endpoint_lock = threading.Lock()
        self._initialized = False

        # Initialize endpoints
        for ep_config in config.endpoints:
            self.endpoints.append(EndpointState(ep_config))

    @abstractmethod
    def _create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
        api_version: Optional[str],
        timeout: float,
    ) -> T:
        """Create an API client instance."""
        pass

    @abstractmethod
    def _execute_request(
        self,
        client: T,
        *args,
        **kwargs,
    ) -> Any:
        """Execute a request using the client."""
        pass

    def _select_endpoint(self) -> EndpointState:
        """Select next endpoint using weighted round-robin."""
        if not self.endpoints:
            raise RuntimeError("No endpoints configured")

        with self.endpoint_lock:
            # Simple weighted selection based on error count
            weights = []
            for ep in self.endpoints:
                # Reduce weight for endpoints with recent errors
                w = ep.config.weight
                if ep.error_count > 0:
                    # Exponential decay based on error count
                    w = max(1, w // (2 ** min(ep.error_count, 4)))
                weights.append(w)

            # Weighted random selection
            total = sum(weights)
            r = random.randint(0, total - 1)
            cumsum = 0
            for i, w in enumerate(weights):
                cumsum += w
                if r < cumsum:
                    return self.endpoints[i]

            return self.endpoints[0]

    def execute_with_retry(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """Execute request with retry and endpoint failover."""
        last_error = None
        tried_endpoints = set()

        for attempt in range(max(1, self.config.max_retries + 1)):
            # Select endpoint
            endpoint = self._select_endpoint()
            tried_endpoints.add(id(endpoint))

            # Get API key
            api_key = endpoint.get_next_key()

            # Acquire rate limit
            endpoint.acquire_rate_limit()

            # Create client
            client = self._create_client(
                base_url=endpoint.config.base_url,
                api_key=api_key,
                api_version=endpoint.config.api_version,
                timeout=self.config.timeout,
            )

            try:
                result = self._execute_request(client, *args, **kwargs)
                endpoint.record_success()
                return result
            except Exception as e:
                last_error = e
                endpoint.record_error()

                # Check if we should retry
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_delay * (self.config.retry_backoff ** attempt),
                        self.config.retry_max_delay,
                    )
                    time.sleep(delay)
                    continue

        raise last_error if last_error else RuntimeError("Request failed")

    def execute_batch(
        self,
        items: List[Tuple[Any, ...]],
        max_workers: Optional[int] = None,
    ) -> List[Tuple[int, Any, Optional[Exception]]]:
        """Execute batch of requests with concurrent processing.

        Args:
            items: List of (args_tuple,) or (args_tuple, kwargs_dict) for each request
            max_workers: Override workers per process

        Returns:
            List of (index, result, error) tuples
        """
        workers = max_workers or self.config.workers_per_process

        results = []

        # Use thread pool for API requests (I/O bound)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}

            for idx, item in enumerate(items):
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                    args, kwargs = item
                else:
                    args = item if isinstance(item, tuple) else (item,)
                    kwargs = {}

                future = executor.submit(self.execute_with_retry, *args, **kwargs)
                futures[future] = idx

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result, None))
                except Exception as e:
                    results.append((idx, None, e))

        # Sort by index
        results.sort(key=lambda x: x[0])
        return results


def parse_endpoints_from_env(
    prefix: str,
    default_api_key_var: Optional[str] = None,
) -> List[EndpointConfig]:
    """Parse endpoint configurations from environment variables.

    Format:
        {PREFIX}_ENDPOINTS = url1,url2,url3  (comma-separated base URLs, empty for default)
        {PREFIX}_API_KEYS = key1,key2,key3   (comma-separated API keys)
        {PREFIX}_RATE_LIMITS = 60,30,60      (optional, per-endpoint rate limits)
        {PREFIX}_WEIGHTS = 1,2,1             (optional, load balancing weights)

    Or single endpoint:
        {PREFIX}_BASE_URL = url
        {PREFIX}_API_KEY = key
        {PREFIX}_RATE_LIMIT = 60

    Args:
        prefix: Environment variable prefix (e.g., "OPENAI", "GEMINI")
        default_api_key_var: Fallback API key variable name

    Returns:
        List of EndpointConfig
    """
    endpoints = []

    # Check for multi-endpoint config
    endpoints_str = os.environ.get(f"DG_{prefix}_ENDPOINTS") or os.environ.get(f"{prefix}_ENDPOINTS")
    api_keys_str = os.environ.get(f"DG_{prefix}_API_KEYS") or os.environ.get(f"{prefix}_API_KEYS")

    if endpoints_str or api_keys_str:
        # Parse multiple endpoints
        base_urls = [u.strip() if u.strip() else None for u in (endpoints_str or "").split(",")]
        api_keys = [k.strip() for k in (api_keys_str or "").split(",") if k.strip()]

        # Parse optional rate limits and weights
        rate_limits_str = os.environ.get(f"DG_{prefix}_RATE_LIMITS") or os.environ.get(f"{prefix}_RATE_LIMITS")
        weights_str = os.environ.get(f"DG_{prefix}_WEIGHTS") or os.environ.get(f"{prefix}_WEIGHTS")

        rate_limits = [int(r.strip()) for r in rate_limits_str.split(",")] if rate_limits_str else []
        weights = [int(w.strip()) for w in weights_str.split(",")] if weights_str else []

        # If no base URLs specified but keys exist, create one endpoint with all keys
        if not base_urls and api_keys:
            base_urls = [None]

        # Distribute keys to endpoints
        for i, base_url in enumerate(base_urls):
            ep_config = EndpointConfig(
                base_url=base_url,
                api_keys=api_keys,  # All keys available to all endpoints by default
                rate_limit=rate_limits[i] if i < len(rate_limits) else 0,
                weight=weights[i] if i < len(weights) else 1,
            )
            endpoints.append(ep_config)

    else:
        # Single endpoint config
        base_url = (
            os.environ.get(f"DG_{prefix}_BASE_URL")
            or os.environ.get(f"{prefix}_BASE_URL")
            or os.environ.get(f"{prefix}_API_BASE")
        )

        api_key = (
            os.environ.get(f"DG_{prefix}_API_KEY")
            or os.environ.get(f"{prefix}_API_KEY")
        )

        # Fallback to default key variable
        if not api_key and default_api_key_var:
            api_key = os.environ.get(default_api_key_var)

        rate_limit_str = os.environ.get(f"DG_{prefix}_RATE_LIMIT") or os.environ.get(f"{prefix}_RATE_LIMIT")
        rate_limit = int(rate_limit_str) if rate_limit_str else 0

        if api_key or base_url:
            endpoints.append(EndpointConfig(
                base_url=base_url,
                api_keys=[api_key] if api_key else [],
                rate_limit=rate_limit,
            ))

    return endpoints


def parse_pool_config_from_env(prefix: str, default_api_key_var: Optional[str] = None) -> PoolConfig:
    """Parse pool configuration from environment variables.

    Only parses endpoint configuration. Pool settings (timeout, max_retries, etc.)
    are set via CLI args and applied via apply_pool_kwargs().
    """
    endpoints = parse_endpoints_from_env(prefix, default_api_key_var)
    return PoolConfig(endpoints=endpoints)


def apply_pool_kwargs(config: PoolConfig, **kwargs) -> PoolConfig:
    """Apply kwargs overrides to pool config.

    CLI args take precedence over defaults.

    Args:
        config: Base pool configuration
        **kwargs: Overrides (timeout, max_retries, retry_delay, max_global_workers, num_processes)

    Returns:
        Updated PoolConfig
    """
    if kwargs.get("timeout") is not None:
        config.timeout = float(kwargs["timeout"])
    if kwargs.get("max_retries") is not None:
        config.max_retries = int(kwargs["max_retries"])
    if kwargs.get("retry_delay") is not None:
        config.retry_delay = float(kwargs["retry_delay"])
    if kwargs.get("max_global_workers") is not None:
        config.max_global_workers = int(kwargs["max_global_workers"])
    if kwargs.get("num_processes") is not None:
        config.num_processes = int(kwargs["num_processes"])
    return config


# Utility functions for image handling

def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.read()


def bytes_to_image(data: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(io.BytesIO(data))


def b64_to_image(b64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    image_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_bytes))
