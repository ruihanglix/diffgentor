# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Base class and utilities for prompt enhancement.
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PIL import Image

from diffgentor.utils.env import get_env_str, get_env_bool


def _is_debug_enabled() -> bool:
    """Check if debug mode is enabled via environment variable."""
    return get_env_bool("PROMPT_ENHANCER_DEBUG", False)


def _get_debug_output_dir() -> Path:
    """Get the debug output directory from environment variable."""
    base_dir = Path(get_env_str("PROMPT_ENHANCER_DEBUG_DIR", "./debug_output"))
    return base_dir / "prompt_enhancer"


def _is_global_rank_zero() -> bool:
    """Check if this is global rank 0 (for distributed training)."""
    import os
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    return str(rank) == "0"


def encode_image_to_base64(pil_image: Image.Image, max_size: int = 2000) -> str:
    """
    Encode a PIL Image to base64 string.

    Args:
        pil_image: PIL Image object
        max_size: Maximum dimension (width or height) before resizing

    Returns:
        Base64 encoded string of the image
    """
    width, height = pil_image.size
    if height > max_size or width > max_size:
        resize_ratio = max_size / max(height, width)
        resize_height = int(height * resize_ratio)
        resize_width = int(width * resize_ratio)
        pil_image = pil_image.resize((resize_width, resize_height))
        print(f"[Warning] Image resized to {resize_width}x{resize_height} for API call")

    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class PromptEnhancer:
    """
    Base class for prompt enhancement.
    Uses OpenAI-compatible API for LLM calls.
    
    Environment variables:
        DG_PROMPT_ENHANCER_API_KEY: API key for the LLM service
        DG_PROMPT_ENHANCER_API_BASE: Base URL for the API
        DG_PROMPT_ENHANCER_MODEL: Model name to use (default: gpt-4o)
        DG_PROMPT_ENHANCER_DEBUG: Enable debug mode (default: false)
        DG_PROMPT_ENHANCER_DEBUG_DIR: Debug output directory (default: ./debug_output)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
    ):
        """
        Initialize the prompt enhancer.

        Args:
            api_key: API key for the LLM service (default: from DG_PROMPT_ENHANCER_API_KEY env var)
            api_base: Base URL for the API (default: from DG_PROMPT_ENHANCER_API_BASE env var)
            model: Model name to use (default: from DG_PROMPT_ENHANCER_MODEL env var)
            max_retries: Maximum number of retries for API calls
        """
        self.api_key = api_key or get_env_str("PROMPT_ENHANCER_API_KEY")
        self.api_base = api_base or get_env_str("PROMPT_ENHANCER_API_BASE")
        self.model = model or get_env_str("PROMPT_ENHANCER_MODEL", "gpt-4o")
        self.max_retries = max_retries

        # Debug mode settings
        self.debug_enabled = _is_debug_enabled() and _is_global_rank_zero()
        self.debug_output_dir = _get_debug_output_dir() if self.debug_enabled else None
        self._debug_counter = 0

        if self.debug_enabled:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Debug] Prompt enhancer debug mode enabled, output dir: {self.debug_output_dir}")

        if not self.api_key:
            raise ValueError(
                "API key not provided. Set DG_PROMPT_ENHANCER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None,
            )
        except ImportError:
            raise ImportError("openai package is required. Install it with: pip install openai")

    def _save_debug_info(
        self,
        original_prompt: str,
        enhanced_prompt: str,
        raw_response: str,
        messages: list,
        images: Optional[List[Image.Image]] = None,
        error: Optional[str] = None,
    ):
        """
        Save debug information to a log file.

        Args:
            original_prompt: The original prompt before enhancement
            enhanced_prompt: The enhanced prompt (or original if failed)
            raw_response: Raw response from LLM API
            messages: Messages sent to LLM API
            images: Input images (optional)
            error: Error message if any
        """
        if not self.debug_enabled or not self.debug_output_dir:
            return

        self._debug_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build log content
        log_lines = [
            "=" * 80,
            f"PROMPT ENHANCER DEBUG LOG",
            "=" * 80,
            "",
            f"[Meta Info]",
            f"Timestamp: {timestamp}",
            f"Counter: {self._debug_counter}",
            f"Model: {self.model}",
            f"API Base: {self.api_base}",
            f"Num Images: {len(images) if images else 0}",
            f"Error: {error if error else 'None'}",
            "",
            "-" * 80,
            "[Original Prompt]",
            "-" * 80,
            original_prompt,
            "",
            "-" * 80,
            "[Enhanced Prompt]",
            "-" * 80,
            enhanced_prompt,
            "",
            "-" * 80,
            "[Raw LLM Response]",
            "-" * 80,
            raw_response or "<empty>",
            "",
            "=" * 80,
        ]

        log_content = "\n".join(log_lines)

        # Save to {idx}.log file
        log_file = self.debug_output_dir / f"{self._debug_counter}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)

        print(f"[Debug] Saved debug info to: {log_file}")

    def enhance(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        """
        Enhance a prompt. Override in subclasses.

        Args:
            prompt: Original prompt to enhance
            images: Optional list of input images for context

        Returns:
            Enhanced prompt string
        """
        raise NotImplementedError("Subclasses must implement enhance method")
