# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Unified exception handling for diffgentor.

This module provides a consistent error handling strategy with full stack traces
for easier debugging.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from enum import Enum
from typing import TypeVar

from diffgentor.utils.logging import print_rank0


class ErrorCode(Enum):
    """Error codes for categorizing exceptions."""

    # General errors
    UNKNOWN = "E0000"
    CONFIG_ERROR = "E0001"
    INITIALIZATION_ERROR = "E0002"

    # Backend errors
    BACKEND_NOT_FOUND = "E1001"
    MODEL_LOAD_ERROR = "E1002"
    GENERATION_ERROR = "E1003"
    EDITING_ERROR = "E1004"

    # API errors
    API_CONNECTION_ERROR = "E2001"
    API_TIMEOUT_ERROR = "E2002"
    API_RATE_LIMIT_ERROR = "E2003"
    API_AUTH_ERROR = "E2004"

    # Data errors
    DATA_LOAD_ERROR = "E3001"
    IMAGE_LOAD_ERROR = "E3002"
    IMAGE_SAVE_ERROR = "E3003"

    # Optimization errors
    OPTIMIZATION_ERROR = "E4001"


class DiffgentorError(Exception):
    """Base exception for all diffgentor errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        cause: Exception | None = None,
    ):
        self.message = message
        self.code = code
        self.cause = cause
        self._traceback = traceback.format_exc() if cause else None
        super().__init__(message)

    def __str__(self) -> str:
        parts = [f"[{self.code.value}] {self.message}"]
        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")
        return "\n".join(parts)

    def format_full(self) -> str:
        """Format error with full traceback."""
        parts = [str(self)]
        if self._traceback and "NoneType: None" not in self._traceback:
            parts.append("\nFull traceback:")
            parts.append(self._traceback)
        return "\n".join(parts)


class BackendError(DiffgentorError):
    """Backend-related errors."""

    pass


class ModelLoadError(BackendError):
    """Error loading model."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.MODEL_LOAD_ERROR, cause)


class GenerationError(BackendError):
    """Error during image generation."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.GENERATION_ERROR, cause)


class EditingError(BackendError):
    """Error during image editing."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.EDITING_ERROR, cause)


class APIError(DiffgentorError):
    """API-related errors."""

    pass


class APIConnectionError(APIError):
    """API connection error."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.API_CONNECTION_ERROR, cause)


class APITimeoutError(APIError):
    """API timeout error."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.API_TIMEOUT_ERROR, cause)


class DataError(DiffgentorError):
    """Data-related errors."""

    pass


class ImageLoadError(DataError):
    """Error loading image."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.IMAGE_LOAD_ERROR, cause)


class OptimizationError(DiffgentorError):
    """Optimization-related errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message, ErrorCode.OPTIMIZATION_ERROR, cause)


T = TypeVar("T")


def log_error(
    error: Exception,
    context: str | None = None,
    include_traceback: bool = True,
) -> None:
    """Log error with optional context and traceback.

    Args:
        error: The exception to log
        context: Optional context information
        include_traceback: Whether to include full traceback
    """
    if context:
        print_rank0(f"Error in {context}:")

    if isinstance(error, DiffgentorError):
        if include_traceback:
            print_rank0(error.format_full())
        else:
            print_rank0(str(error))
    else:
        print_rank0(f"{type(error).__name__}: {error}")
        if include_traceback:
            print_rank0("Traceback:")
            print_rank0(traceback.format_exc())


def safe_execute(
    func: Callable[..., T],
    *args,
    default: T | None = None,
    error_context: str | None = None,
    reraise: bool = False,
    **kwargs,
) -> T | None:
    """Safely execute a function with error logging.

    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        error_context: Context for error logging
        reraise: Whether to reraise the exception
        **kwargs: Keyword arguments

    Returns:
        Function result or default on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(e, context=error_context, include_traceback=True)
        if reraise:
            raise
        return default
