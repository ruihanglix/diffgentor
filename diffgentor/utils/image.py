# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Image utilities for diffgentor."""

import hashlib
import io
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import requests
from PIL import Image


def download_image(
    url_or_path: str,
    cache_dir: Optional[str] = None,
    timeout: int = 30,
) -> Optional[Image.Image]:
    """Download or load an image from URL or local path.

    Args:
        url_or_path: URL or local file path
        cache_dir: Directory to cache downloaded images
        timeout: Request timeout in seconds

    Returns:
        PIL Image or None if failed
    """
    # Check if it's a URL
    if url_or_path.startswith(("http://", "https://")):
        return _download_from_url(url_or_path, cache_dir, timeout)
    else:
        return _load_from_path(url_or_path)


def _download_from_url(
    url: str,
    cache_dir: Optional[str] = None,
    timeout: int = 30,
) -> Optional[Image.Image]:
    """Download image from URL."""
    try:
        # Check cache first
        if cache_dir:
            cache_path = _get_cache_path(url, cache_dir)
            if cache_path.exists():
                return Image.open(cache_path).convert("RGB")

        # Download
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        # Load image
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Cache if directory specified
        if cache_dir:
            cache_path = _get_cache_path(url, cache_dir)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_path)

        return image

    except Exception as e:
        print(f"Failed to download image from {url}: {e}")
        return None


def _load_from_path(path: str) -> Optional[Image.Image]:
    """Load image from local path."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Failed to load image from {path}: {e}")
        return None


def _get_cache_path(url: str, cache_dir: str) -> Path:
    """Get cache path for URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = Path(urlparse(url).path).suffix or ".jpg"
    return Path(cache_dir) / f"{url_hash}{ext}"


def save_image(
    image: Image.Image,
    output_dir: Union[str, Path],
    filename: str,
    format: str = "PNG",
) -> str:
    """Save image to file.

    Args:
        image: PIL Image to save
        output_dir: Output directory
        filename: Filename (with or without extension)
        format: Image format

    Returns:
        Full path to saved image
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add extension if not present
    if not Path(filename).suffix:
        filename = f"{filename}.{format.lower()}"

    output_path = output_dir / filename
    image.save(output_path, format=format)

    return str(output_path)


def parse_input_images(input_str: str) -> List[str]:
    """Parse input images string to list of paths/URLs.

    Supports formats:
    - JSON array: '["path1", "path2"]'
    - Comma-separated: 'path1,path2'
    - Single path: 'path1'

    Args:
        input_str: Input string

    Returns:
        List of image paths/URLs
    """
    import json

    input_str = input_str.strip()

    if not input_str or input_str == "[]":
        return []

    # Try JSON array first
    try:
        result = json.loads(input_str)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        pass

    # Try comma-separated
    if "," in input_str:
        return [x.strip() for x in input_str.split(",") if x.strip()]

    # Single path
    return [input_str]


def get_output_filename(index: int, sub_index: int | None = None) -> str:
    """Generate output image filename.

    Args:
        index: Image index (0-999999)
        sub_index: Sub-index for multiple images per prompt (0-99), None for single image

    Returns:
        Filename like "000000.png" or "000000_00.png"
    """
    if sub_index is None:
        return f"{index:06d}.png"
    return f"{index:06d}_{sub_index:02d}.png"


# Supported image formats for custom output names
SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg"}
DEFAULT_IMAGE_FORMAT = ".png"


def normalize_custom_output_path(custom_name: str, sub_index: int | None = None) -> str:
    """Normalize a custom output path from user-specified column value.

    This function handles:
    - Paths with directories (e.g., "a/b/c" -> "a/b/c.png")
    - Paths with extensions (e.g., "a/b/c.png" -> "a/b/c.png", "a/b/c.jpg" -> "a/b/c.jpg")
    - Unsupported extensions default to .png (e.g., "a/b/c.webp" -> "a/b/c.webp.png")
    - Sub-index for multiple images per prompt (e.g., "a/b/c" with sub_index=1 -> "a/b/c_01.png")

    Args:
        custom_name: Custom filename/path from the data column
        sub_index: Sub-index for multiple images per prompt (0-99), None for single image

    Returns:
        Normalized path string (relative path with proper extension)
    """
    if not custom_name or not custom_name.strip():
        raise ValueError("Custom output name cannot be empty")

    custom_name = custom_name.strip()

    # Parse the path
    path = Path(custom_name)
    suffix = path.suffix.lower()

    # Determine the base name and extension
    if suffix in SUPPORTED_IMAGE_FORMATS:
        # Has a supported extension
        base_path = path.with_suffix("")
        extension = suffix
    else:
        # No extension or unsupported extension - use the whole name as base
        base_path = path
        extension = DEFAULT_IMAGE_FORMAT

    # Add sub_index if needed
    if sub_index is not None:
        base_str = str(base_path)
        base_path = Path(f"{base_str}_{sub_index:02d}")

    # Reconstruct the full path with extension
    return str(base_path) + extension


def get_custom_output_path(
    output_dir: Union[str, Path],
    custom_name: str,
    sub_index: int | None = None,
) -> Path:
    """Get the full output path for a custom-named file.

    This function:
    - Normalizes the custom name
    - Creates parent directories if needed
    - Returns the full absolute path

    Args:
        output_dir: Base output directory
        custom_name: Custom filename/path from the data column
        sub_index: Sub-index for multiple images per prompt

    Returns:
        Full Path object for the output file
    """
    output_dir = Path(output_dir)
    normalized = normalize_custom_output_path(custom_name, sub_index)
    full_path = output_dir / normalized

    # Ensure parent directory exists
    full_path.parent.mkdir(parents=True, exist_ok=True)

    return full_path


def save_image_to_custom_path(
    image: Image.Image,
    output_dir: Union[str, Path],
    custom_name: str,
    sub_index: int | None = None,
) -> str:
    """Save an image to a custom-named path.

    Args:
        image: PIL Image to save
        output_dir: Base output directory
        custom_name: Custom filename/path from the data column
        sub_index: Sub-index for multiple images per prompt

    Returns:
        Relative path string (from output_dir) of the saved file
    """
    output_path = get_custom_output_path(output_dir, custom_name, sub_index)

    # Determine format from extension
    suffix = output_path.suffix.lower()
    if suffix == ".jpg" or suffix == ".jpeg":
        image.save(output_path, format="JPEG", quality=95)
    else:
        image.save(output_path, format="PNG")

    # Return relative path from output_dir
    return str(output_path.relative_to(output_dir))
