# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Environment variable utilities for model-specific parameters.

Model-specific parameters should be passed via DG_ prefixed environment variables
instead of CLI arguments. This keeps the CLI clean and allows flexible configuration.

Naming convention: DG_{MODEL}_{PARAM}
Examples:
    DG_STEP1X_VERSION=v1.1
    DG_STEP1X_SIZE_LEVEL=512
    DG_BAGEL_CFG_TEXT_SCALE=3.0
    DG_EMU35_VQ_PATH=/path/to/vq
    DG_DREAMOMNI2_VLM_PATH=/path/to/vlm

API Backend Configuration (OpenAI, Google GenAI):
    # Single endpoint
    {PREFIX}_API_KEY: API key
    DG_{PREFIX}_BASE_URL: Base URL override
    DG_{PREFIX}_RATE_LIMIT: Rate limit per minute

    # Multiple endpoints
    DG_{PREFIX}_ENDPOINTS: Comma-separated base URLs (empty for default)
    DG_{PREFIX}_API_KEYS: Comma-separated API keys
    DG_{PREFIX}_RATE_LIMITS: Comma-separated rate limits per endpoint
    DG_{PREFIX}_WEIGHTS: Comma-separated load balancing weights

    # Pool settings
    DG_{PREFIX}_TIMEOUT: Timeout in seconds (default: 300)
    DG_{PREFIX}_MAX_RETRIES: Max retry attempts (default: 0)
    DG_{PREFIX}_RETRY_DELAY: Initial retry delay in seconds (default: 1.0)
    DG_{PREFIX}_MAX_WORKERS: Max concurrent workers (default: 4)

Where {PREFIX} is OPENAI or GEMINI.
"""

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, TypeVar, Union

T = TypeVar("T")


def get_env(
    name: str,
    default: T = None,
    type_cast: type = str,
) -> Union[T, Any]:
    """Get environment variable with optional type casting.

    Args:
        name: Environment variable name (without DG_ prefix)
        default: Default value if not set
        type_cast: Type to cast the value to

    Returns:
        Environment variable value or default
    """
    full_name = f"DG_{name}"
    value = os.environ.get(full_name)

    if value is None:
        return default

    try:
        if type_cast == bool:
            return value.lower() in ("true", "1", "yes", "on")
        elif type_cast == tuple:
            parts = value.split(",")
            return tuple(float(p.strip()) for p in parts)
        elif type_cast == list:
            return [p.strip() for p in value.split(",")]
        else:
            return type_cast(value)
    except (ValueError, TypeError):
        return default


def get_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get string environment variable."""
    return get_env(name, default, str)


def get_env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Get integer environment variable."""
    return get_env(name, default, int)


def get_env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Get float environment variable."""
    return get_env(name, default, float)


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return get_env(name, default, bool)


def get_env_tuple(
    name: str,
    default: Optional[Tuple[float, ...]] = None,
) -> Optional[Tuple[float, ...]]:
    """Get tuple environment variable (comma-separated floats)."""
    return get_env(name, default, tuple)


def get_env_list(
    name: str,
    default: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Get list environment variable (comma-separated strings)."""
    return get_env(name, default, list)


# =============================================================================
# Model-specific environment configurations using dataclasses
# =============================================================================


@dataclass
class ModelEnvConfig:
    """Base class for model environment configurations."""

    _prefix: str = field(default="", repr=False)

    @classmethod
    def load(cls) -> "ModelEnvConfig":
        """Load configuration from environment variables."""
        raise NotImplementedError


@dataclass
class Step1XEnv(ModelEnvConfig):
    """Step1X model environment variables.

    Environment variables:
        DG_STEP1X_VERSION: Model version (default: v1.1)
        DG_STEP1X_SIZE_LEVEL: Size level for image processing (default: 512)
    """

    _prefix: str = field(default="STEP1X", repr=False)
    version: str = "v1.1"
    size_level: int = 512

    @classmethod
    def load(cls) -> "Step1XEnv":
        return cls(
            version=get_env_str("STEP1X_VERSION", "v1.1"),
            size_level=get_env_int("STEP1X_SIZE_LEVEL", 512),
        )


@dataclass
class BagelEnv(ModelEnvConfig):
    """BAGEL model environment variables.

    Environment variables:
        DG_BAGEL_CFG_TEXT_SCALE: Text CFG scale (default: 3.0)
        DG_BAGEL_CFG_IMG_SCALE: Image CFG scale (default: 1.5)
        DG_BAGEL_CFG_INTERVAL: CFG interval as comma-separated floats (default: 0.4,1.0)
        DG_BAGEL_TIMESTEP_SHIFT: Timestep shift value (default: 3.0)
    """

    _prefix: str = field(default="BAGEL", repr=False)
    cfg_text_scale: float = 3.0
    cfg_img_scale: float = 1.5
    cfg_interval: Tuple[float, float] = (0.4, 1.0)
    timestep_shift: float = 3.0

    @classmethod
    def load(cls) -> "BagelEnv":
        return cls(
            cfg_text_scale=get_env_float("BAGEL_CFG_TEXT_SCALE", 3.0),
            cfg_img_scale=get_env_float("BAGEL_CFG_IMG_SCALE", 1.5),
            cfg_interval=get_env_tuple("BAGEL_CFG_INTERVAL", (0.4, 1.0)),
            timestep_shift=get_env_float("BAGEL_TIMESTEP_SHIFT", 3.0),
        )


@dataclass
class Emu35Env(ModelEnvConfig):
    """Emu3.5 model environment variables.

    Environment variables:
        DG_EMU35_VQ_PATH: Path to VisionTokenizer model
        DG_EMU35_TOKENIZER_PATH: Path to tokenizer (optional)
        DG_EMU35_CFG: Classifier-free guidance scale (default: 3.0)
        DG_EMU35_MAX_NEW_TOKENS: Maximum new tokens to generate (default: 5120)
        DG_EMU35_IMAGE_AREA: Image area for resizing (default: 1048576)
        DG_EMU35_VQ_DEVICE: Device for VQ model (default: cuda:0)
        DG_EMU35_GPUS_PER_MODEL: Number of GPUs per model instance (default: 0, use all visible)
    """

    _prefix: str = field(default="EMU35", repr=False)
    vq_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    cfg: float = 3.0
    max_new_tokens: int = 5120
    image_area: int = 1048576
    vq_device: str = "cuda:0"
    _gpus_per_model: int = field(default=0, repr=False)

    @classmethod
    def load(cls) -> "Emu35Env":
        return cls(
            vq_path=get_env_str("EMU35_VQ_PATH"),
            tokenizer_path=get_env_str("EMU35_TOKENIZER_PATH"),
            cfg=get_env_float("EMU35_CFG", 3.0),
            max_new_tokens=get_env_int("EMU35_MAX_NEW_TOKENS", 5120),
            image_area=get_env_int("EMU35_IMAGE_AREA", 1048576),
            vq_device=get_env_str("EMU35_VQ_DEVICE", "cuda:0"),
            _gpus_per_model=get_env_int("EMU35_GPUS_PER_MODEL", 0),
        )

    @staticmethod
    def gpus_per_model() -> int:
        """Get number of GPUs per model instance from environment.

        Used by Launcher to determine launch strategy.

        Returns:
            Number of GPUs per model, 0 means use all visible GPUs
        """
        return get_env_int("EMU35_GPUS_PER_MODEL", 0)


@dataclass
class DreamOmni2Env(ModelEnvConfig):
    """DreamOmni2 model environment variables.

    Environment variables:
        DG_DREAMOMNI2_VLM_PATH: Path to VLM model (Qwen2.5-VL)
        DG_DREAMOMNI2_LORA_PATH: Path to LoRA weights
        DG_DREAMOMNI2_TASK_TYPE: Task type - "generation" or "editing" (default: generation)
        DG_DREAMOMNI2_OUTPUT_HEIGHT: Output image height (default: 1024)
        DG_DREAMOMNI2_OUTPUT_WIDTH: Output image width (default: 1024)
    """

    _prefix: str = field(default="DREAMOMNI2", repr=False)
    vlm_path: Optional[str] = None
    lora_path: Optional[str] = None
    task_type: str = "generation"
    output_height: int = 1024
    output_width: int = 1024

    @classmethod
    def load(cls) -> "DreamOmni2Env":
        return cls(
            vlm_path=get_env_str("DREAMOMNI2_VLM_PATH"),
            lora_path=get_env_str("DREAMOMNI2_LORA_PATH"),
            task_type=get_env_str("DREAMOMNI2_TASK_TYPE", "generation"),
            output_height=get_env_int("DREAMOMNI2_OUTPUT_HEIGHT", 1024),
            output_width=get_env_int("DREAMOMNI2_OUTPUT_WIDTH", 1024),
        )


@dataclass
class FluxKontextEnv(ModelEnvConfig):
    """Flux Kontext Official model environment variables.

    Environment variables:
        DG_FLUX_KONTEXT_OFFLOAD: Enable model offloading (default: false)
        DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH: Max sequence length (default: 512)
    """

    _prefix: str = field(default="FLUX_KONTEXT", repr=False)
    offload: bool = False
    max_sequence_length: int = 512

    @classmethod
    def load(cls) -> "FluxKontextEnv":
        return cls(
            offload=get_env_bool("FLUX_KONTEXT_OFFLOAD", False),
            max_sequence_length=get_env_int("FLUX_KONTEXT_MAX_SEQUENCE_LENGTH", 512),
        )


@dataclass
class OpenAIEnv(ModelEnvConfig):
    """OpenAI API environment variables.

    Single endpoint:
        OPENAI_API_KEY: API key
        OPENAI_API_BASE: Base URL (optional)
        DG_OPENAI_RATE_LIMIT: Rate limit per minute

    Multiple endpoints:
        DG_OPENAI_ENDPOINTS: Comma-separated base URLs (empty for default)
        DG_OPENAI_API_KEYS: Comma-separated API keys
        DG_OPENAI_RATE_LIMITS: Comma-separated rate limits
        DG_OPENAI_WEIGHTS: Comma-separated load balancing weights

    Pool settings:
        DG_OPENAI_TIMEOUT: Timeout in seconds (default: 300)
        DG_OPENAI_MAX_RETRIES: Max retry attempts (default: 0)
        DG_OPENAI_RETRY_DELAY: Initial retry delay (default: 1.0)
        DG_OPENAI_MAX_WORKERS: Max concurrent workers (default: 4)
    """

    _prefix: str = field(default="OPENAI", repr=False)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 300.0
    max_retries: int = 0
    max_workers: int = 4

    @classmethod
    def load(cls) -> "OpenAIEnv":
        return cls(
            api_key=get_env_str("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=get_env_str("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
            timeout=get_env_float("OPENAI_TIMEOUT", 300.0),
            max_retries=get_env_int("OPENAI_MAX_RETRIES", 0),
            max_workers=get_env_int("OPENAI_MAX_WORKERS", 4),
        )


@dataclass
class GoogleGenAIEnv(ModelEnvConfig):
    """Google GenAI (Gemini) API environment variables.

    Single endpoint:
        GEMINI_API_KEY: API key
        DG_GEMINI_BASE_URL: Base URL override for proxy
        DG_GEMINI_API_VERSION: API version (v1, v1beta, v1alpha)
        DG_GEMINI_RATE_LIMIT: Rate limit per minute
        DG_GEMINI_ASPECT_RATIO: Default aspect ratio

    Multiple endpoints:
        DG_GEMINI_ENDPOINTS: Comma-separated base URLs (empty for default)
        DG_GEMINI_API_KEYS: Comma-separated API keys
        DG_GEMINI_RATE_LIMITS: Comma-separated rate limits
        DG_GEMINI_WEIGHTS: Comma-separated load balancing weights

    Pool settings:
        DG_GEMINI_TIMEOUT: Timeout in seconds (default: 300)
        DG_GEMINI_MAX_RETRIES: Max retry attempts (default: 0)
        DG_GEMINI_RETRY_DELAY: Initial retry delay (default: 1.0)
        DG_GEMINI_MAX_WORKERS: Max concurrent workers (default: 4)
    """

    _prefix: str = field(default="GEMINI", repr=False)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    rate_limit: int = 0
    aspect_ratio: Optional[str] = None
    timeout: float = 300.0
    max_retries: int = 0
    max_workers: int = 4

    @classmethod
    def load(cls) -> "GoogleGenAIEnv":
        return cls(
            api_key=get_env_str("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY"),
            base_url=get_env_str("GEMINI_BASE_URL") or os.environ.get("GOOGLE_GENAI_BASE_URL"),
            api_version=get_env_str("GEMINI_API_VERSION") or os.environ.get("GOOGLE_GENAI_API_VERSION"),
            rate_limit=get_env_int("GEMINI_RATE_LIMIT", 0),
            aspect_ratio=get_env_str("GEMINI_ASPECT_RATIO"),
            timeout=get_env_float("GEMINI_TIMEOUT", 300.0),
            max_retries=get_env_int("GEMINI_MAX_RETRIES", 0),
            max_workers=get_env_int("GEMINI_MAX_WORKERS", 4),
        )


@dataclass
class XDiTEnv(ModelEnvConfig):
    """xDiT parallelism environment variables.

    Environment variables:
        DG_XDIT_DATA_PARALLEL_DEGREE: Data parallelism degree (default: 1)
        DG_XDIT_ULYSSES_DEGREE: Ulysses sequence parallelism degree (default: 1)
        DG_XDIT_RING_DEGREE: Ring sequence parallelism degree (default: 1)
        DG_XDIT_PIPEFUSION_DEGREE: PipeFusion parallelism degree (default: 1)
        DG_XDIT_USE_CFG_PARALLEL: Enable CFG parallelism (default: false)
    """

    _prefix: str = field(default="XDIT", repr=False)
    data_parallel_degree: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    pipefusion_degree: int = 1
    use_cfg_parallel: bool = False

    @classmethod
    def load(cls) -> "XDiTEnv":
        return cls(
            data_parallel_degree=get_env_int("XDIT_DATA_PARALLEL_DEGREE", 1),
            ulysses_degree=get_env_int("XDIT_ULYSSES_DEGREE", 1),
            ring_degree=get_env_int("XDIT_RING_DEGREE", 1),
            pipefusion_degree=get_env_int("XDIT_PIPEFUSION_DEGREE", 1),
            use_cfg_parallel=get_env_bool("XDIT_USE_CFG_PARALLEL", False),
        )


@dataclass
class HunyuanImage3Env(ModelEnvConfig):
    """HunyuanImage-3.0-Instruct model environment variables.

    Supports HunyuanImage-3.0-Instruct and HunyuanImage-3.0-Instruct-Distil models.
    Model weights can be downloaded with:
        hf download tencent/HunyuanImage-3.0-Instruct-Distil --local-dir ./HunyuanImage-3-Instruct-Distil

    Environment variables:
        DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL: Number of GPUs per model instance (default: 0, use all visible)
        DG_HUNYUAN_IMAGE_3_ATTN_IMPL: Attention implementation (default: sdpa)
        DG_HUNYUAN_IMAGE_3_MOE_IMPL: MoE implementation, "eager" or "flashinfer" (default: eager)
        DG_HUNYUAN_IMAGE_3_MOE_DROP_TOKENS: Enable MoE token dropping (default: true)
        DG_HUNYUAN_IMAGE_3_USE_SYSTEM_PROMPT: System prompt type (default: en_unified)
            Options: None, dynamic, en_vanilla, en_recaption, en_think_recaption, en_unified, custom
        DG_HUNYUAN_IMAGE_3_BOT_TASK: Task type (default: think_recaption)
            Options: image (direct), auto (text), recaption (rewrite->image), think_recaption (think->rewrite->image)
        DG_HUNYUAN_IMAGE_3_INFER_ALIGN_IMAGE_SIZE: Align output size to input size (default: true)
        DG_HUNYUAN_IMAGE_3_MAX_NEW_TOKENS: Maximum new tokens for text generation (default: 2048)
        DG_HUNYUAN_IMAGE_3_USE_TAYLOR_CACHE: Use Taylor Cache when sampling (default: false)
    """

    _prefix: str = field(default="HUNYUAN_IMAGE_3", repr=False)
    _gpus_per_model: int = field(default=0, repr=False)
    attn_impl: str = "sdpa"
    moe_impl: str = "eager"
    moe_drop_tokens: bool = True
    use_system_prompt: str = "en_unified"
    bot_task: str = "think_recaption"
    infer_align_image_size: bool = True
    max_new_tokens: int = 2048
    use_taylor_cache: bool = False

    @classmethod
    def load(cls) -> "HunyuanImage3Env":
        return cls(
            _gpus_per_model=get_env_int("HUNYUAN_IMAGE_3_GPUS_PER_MODEL", 0),
            attn_impl=get_env_str("HUNYUAN_IMAGE_3_ATTN_IMPL", "sdpa"),
            moe_impl=get_env_str("HUNYUAN_IMAGE_3_MOE_IMPL", "eager"),
            moe_drop_tokens=get_env_bool("HUNYUAN_IMAGE_3_MOE_DROP_TOKENS", True),
            use_system_prompt=get_env_str("HUNYUAN_IMAGE_3_USE_SYSTEM_PROMPT", "en_unified"),
            bot_task=get_env_str("HUNYUAN_IMAGE_3_BOT_TASK", "think_recaption"),
            infer_align_image_size=get_env_bool("HUNYUAN_IMAGE_3_INFER_ALIGN_IMAGE_SIZE", True),
            max_new_tokens=get_env_int("HUNYUAN_IMAGE_3_MAX_NEW_TOKENS", 2048),
            use_taylor_cache=get_env_bool("HUNYUAN_IMAGE_3_USE_TAYLOR_CACHE", False),
        )

    @staticmethod
    def gpus_per_model() -> int:
        """Get number of GPUs per model instance from environment.

        Used by Launcher to determine launch strategy.

        Returns:
            Number of GPUs per model, 0 means use all visible GPUs
        """
        return get_env_int("HUNYUAN_IMAGE_3_GPUS_PER_MODEL", 0)
