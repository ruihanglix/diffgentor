# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Configuration classes for diffgentor."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_default_num_gpus() -> int:
    """Get the default number of GPUs to use.

    Respects CUDA_VISIBLE_DEVICES if set, otherwise uses all available GPUs.
    Falls back to 1 if torch is not available or no GPUs are detected.
    """
    # Check CUDA_VISIBLE_DEVICES first
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        if cuda_visible == "":
            return 1  # Empty string means no GPUs visible
        return len(cuda_visible.split(","))

    # Try to detect available GPUs via torch
    try:
        import torch

        gpu_count = torch.cuda.device_count()
        return gpu_count if gpu_count > 0 else 1
    except (ImportError, Exception):
        return 1


class Backend(str, Enum):
    """Supported backend types."""

    DIFFUSERS = "diffusers"
    XDIT = "xdit"
    OPENAI = "openai"


class AttentionBackend(str, Enum):
    """Supported attention backends."""

    DEFAULT = "default"
    FLASH = "flash"
    FLASH_3 = "flash_3"
    SAGE = "sage"
    XFORMERS = "xformers"


class CacheType(str, Enum):
    """Supported cache types for inference acceleration."""

    NONE = "none"
    PAB = "pab"  # Pyramid Attention Broadcast
    FASTER_CACHE = "faster_cache"
    FIRST_BLOCK_CACHE = "first_block_cache"
    TAYLOR_SEER = "taylor_seer"
    DEEP_CACHE = "deep_cache"
    CACHE_DIT = "cache_dit"


class QuantizationType(str, Enum):
    """Supported quantization types."""

    NONE = "none"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"


@dataclass
class OptimizationConfig:
    """Configuration for pipeline optimizations.

    Attributes:
        torch_dtype: Data type for model weights (float16, bfloat16, float32)
        enable_vae_slicing: Enable VAE slicing for memory efficiency in batch processing
        enable_vae_tiling: Enable VAE tiling for large resolution images
        enable_cpu_offload: Enable model CPU offload to reduce VRAM usage
        enable_sequential_cpu_offload: Enable sequential CPU offload (slower but less VRAM)
        enable_xformers: Enable xFormers memory efficient attention
        attention_backend: Attention backend to use (flash, sage, xformers)
        enable_compile: Enable torch.compile for speedup
        compile_mode: torch.compile mode (default, reduce-overhead, max-autotune, etc.)
        compile_fullgraph: Use fullgraph mode for torch.compile
        cache_type: Type of cache acceleration to use
        cache_config: Additional cache configuration parameters
        enable_group_offloading: Enable group offloading for memory optimization
        group_offload_type: Type of group offloading (leaf_level, block_level)
        enable_layerwise_casting: Enable layerwise dtype casting
        storage_dtype: Storage dtype for layerwise casting
        compute_dtype: Compute dtype for layerwise casting
        quantization: Quantization type to apply
        quantization_components: Model components to quantize
        enable_fuse_qkv: Fuse QKV projections for speedup
        enable_tf32: Enable TensorFloat-32 for speedup on Ampere+ GPUs
    """

    # Basic optimizations
    torch_dtype: str = "bfloat16"
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_xformers: bool = False

    # Attention backend
    attention_backend: Optional[str] = None

    # Compilation
    enable_compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_fullgraph: bool = False
    compile_components: List[str] = field(default_factory=lambda: ["transformer"])

    # Cache acceleration
    cache_type: Optional[str] = None
    cache_config: Dict[str, Any] = field(default_factory=dict)

    # Memory optimization
    enable_group_offloading: bool = False
    group_offload_type: str = "leaf_level"
    enable_layerwise_casting: bool = False
    storage_dtype: str = "float8_e4m3fn"
    compute_dtype: str = "bfloat16"

    # Quantization
    quantization: Optional[str] = None
    quantization_components: List[str] = field(default_factory=lambda: ["transformer"])

    # Other optimizations
    enable_fuse_qkv: bool = False
    enable_tf32: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "torch_dtype": self.torch_dtype,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_vae_tiling": self.enable_vae_tiling,
            "enable_cpu_offload": self.enable_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
            "enable_xformers": self.enable_xformers,
            "attention_backend": self.attention_backend,
            "enable_compile": self.enable_compile,
            "compile_mode": self.compile_mode,
            "compile_fullgraph": self.compile_fullgraph,
            "compile_components": self.compile_components,
            "cache_type": self.cache_type,
            "cache_config": self.cache_config,
            "enable_group_offloading": self.enable_group_offloading,
            "group_offload_type": self.group_offload_type,
            "enable_layerwise_casting": self.enable_layerwise_casting,
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "quantization": self.quantization,
            "quantization_components": self.quantization_components,
            "enable_fuse_qkv": self.enable_fuse_qkv,
            "enable_tf32": self.enable_tf32,
        }


@dataclass
class BackendConfig:
    """Configuration for backend selection and initialization.

    Attributes:
        backend: Backend type (diffusers, xdit, openai)
        model_name: Model name or path (HuggingFace ID or local path)
        model_type: Optional model type for explicit pipeline selection
        device: Device to use (cuda, cpu, or specific cuda:N)
        num_gpus: Number of GPUs to use
        seed: Random seed for reproducibility
        batch_size: Batch size for batch inference (multiple images per forward pass)

    Note:
        xDiT specific parameters (data_parallel_degree, ulysses_degree, ring_degree,
        pipefusion_degree, use_cfg_parallel) are configured via DG_XDIT_* environment
        variables. See diffgentor.utils.env.XDiTEnv for details.
    """

    backend: str = "diffusers"
    model_name: str = ""
    model_type: Optional[str] = None
    device: str = "cuda"
    num_gpus: int = field(default_factory=_get_default_num_gpus)
    seed: Optional[int] = None

    # Batch inference
    batch_size: int = 1

    # OpenAI specific
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None

    @property
    def data_parallel_degree(self) -> int:
        """Get data parallel degree from environment variable."""
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_DATA_PARALLEL_DEGREE", 1)

    @property
    def ulysses_degree(self) -> int:
        """Get Ulysses sequence parallel degree from environment variable."""
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_ULYSSES_DEGREE", 1)

    @property
    def ring_degree(self) -> int:
        """Get Ring sequence parallel degree from environment variable."""
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_RING_DEGREE", 1)

    @property
    def pipefusion_degree(self) -> int:
        """Get PipeFusion parallel degree from environment variable."""
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_PIPEFUSION_DEGREE", 1)

    @property
    def use_cfg_parallel(self) -> bool:
        """Get CFG parallel flag from environment variable."""
        from diffgentor.utils.env import get_env_bool
        return get_env_bool("XDIT_USE_CFG_PARALLEL", False)


@dataclass
class T2IConfig:
    """Configuration for Text-to-Image generation.

    Attributes:
        backend_config: Backend configuration
        optimization_config: Optimization configuration
        prompt: Single prompt or path to prompts file
        prompts_file: Path to file containing prompts (JSONL/CSV)
        output_dir: Directory to save generated images
        output_name_column: Column name to use for output filename (supports paths like "a/b/c.png")
        num_images_per_prompt: Number of images to generate per prompt
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
        negative_prompt: Negative prompt for generation
        batch_size: Batch size for generation
        max_retries: Maximum retries for failed generations
        resume: Resume from previous progress
    """

    backend_config: BackendConfig = field(default_factory=BackendConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Input/Output
    prompt: Optional[str] = None
    prompts_file: Optional[str] = None
    output_dir: str = "./output"
    output_name_column: Optional[str] = None

    # Generation parameters
    num_images_per_prompt: int = 1
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[str] = None
    batch_size: int = 1

    # Execution
    max_retries: int = 3
    resume: bool = True

    # Multi-node settings
    node_rank: int = 0
    num_nodes: int = 1


@dataclass
class EditingConfig:
    """Configuration for Image Editing.

    Attributes:
        backend_config: Backend configuration
        optimization_config: Optimization configuration
        input_data: Path to input data (CSV or Parquet)
        output_dir: Directory to save edited images
        output_csv: Path to output CSV file
        output_name_column: Column name to use for output filename (supports paths like "a/b/c.png")
        instruction_key: Column name for instruction in input data
        image_cache_dir: Directory to cache downloaded images
        batch_size: Batch size for processing
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        true_cfg_scale: True CFG scale (for Qwen models)
        negative_prompt: Negative prompt
        max_retries: Maximum retries for failed edits
        filter_rows: Filter expression for row selection
        resume: Resume from previous progress
        prompt_enhance_type: Type of prompt enhancement (qwen_image_edit, glm_image, etc.)
        prompt_enhance_api_key: API key for prompt enhancement
        prompt_enhance_api_base: API base URL for prompt enhancement
        prompt_enhance_model: Model name for prompt enhancement
    """

    backend_config: BackendConfig = field(default_factory=BackendConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Input/Output
    input_data: str = ""
    output_dir: str = "./output"
    output_csv: str = "./output/results.csv"
    output_name_column: Optional[str] = None
    instruction_key: str = "instruction"
    image_cache_dir: Optional[str] = None

    # Generation parameters
    batch_size: int = 1
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = None
    negative_prompt: str = " "

    # Execution
    max_retries: int = 3
    filter_rows: Optional[str] = None
    resume: bool = True

    # Multi-node settings
    node_rank: int = 0
    num_nodes: int = 1

    # Model-specific parameters (stored as dict for flexibility)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Prompt enhancement
    prompt_enhance_type: Optional[str] = None
    prompt_enhance_api_key: Optional[str] = None
    prompt_enhance_api_base: Optional[str] = None
    prompt_enhance_model: Optional[str] = None

    @property
    def input_csv_dir(self) -> str:
        """Get the directory containing the input data file."""
        return str(Path(self.input_data).parent.resolve())


# Mapping from model type hints to pipeline classes
MODEL_TYPE_HINTS = {
    # T2I models
    "flux": "FluxPipeline",
    "flux.1": "FluxPipeline",
    "flux.2": "FluxPipeline",
    "sd3": "StableDiffusion3Pipeline",
    "sdxl": "StableDiffusionXLPipeline",
    "sd": "StableDiffusionPipeline",
    "hunyuan": "HunyuanDiTPipeline",
    "pixart": "PixArtAlphaPipeline",
    "cogview": "CogView3PlusPipeline",
    # Editing models (diffusers-based)
    "qwen": "QwenImageEditPlusPipeline",
    "qwen_singleimg": "QwenImageEditPipeline",
    "flux2_edit": "Flux2Pipeline",
    "flux2_klein": "Flux2KleinPipeline",
    "flux1_kontext": "FluxKontextPipeline",
    "longcat": "LongCatImageEditPipeline",
    "glm_image": "GlmImagePipeline",
}

# Model name patterns for auto-detection
MODEL_NAME_PATTERNS = {
    "FLUX.1": "flux",
    "FLUX.2": "flux",
    "FLUX": "flux",
    "stable-diffusion-3": "sd3",
    "stable-diffusion-xl": "sdxl",
    "stable-diffusion": "sd",
    "HunyuanDiT": "hunyuan",
    "PixArt": "pixart",
    "CogView": "cogview",
    "Qwen-Image-Edit-Plus": "qwen",
    "Qwen-Image-Edit-2509": "qwen",
    "Qwen-Image-Edit-2511": "qwen",
    "Qwen-Image-Edit": "qwen_singleimg",
    "FLUX.2-dev": "flux2_edit",
    "FLUX.2-klein": "flux2_klein",
    "FLUX.1-Kontext": "flux1_kontext",
    "LongCat": "longcat",
    "GLM-Image": "glm_image",
}


def detect_model_type(model_name: str) -> Optional[str]:
    """Detect model type from model name.

    Args:
        model_name: Model name or path

    Returns:
        Detected model type or None if not detected
    """
    model_name_lower = model_name.lower()
    model_basename = Path(model_name).name

    # Check exact patterns first
    for pattern, model_type in MODEL_NAME_PATTERNS.items():
        if pattern.lower() in model_name_lower or pattern.lower() in model_basename.lower():
            return model_type

    return None
