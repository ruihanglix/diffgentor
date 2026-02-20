# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Diffusers backend implementation for diffgentor."""

from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseBackend
from diffgentor.backends.lora_mixin import DiffusersLoRAMixin
from diffgentor.config import (
    BackendConfig,
    OptimizationConfig,
    detect_model_type,
    MODEL_TYPE_HINTS,
)
from diffgentor.utils.logging import print_rank0


# Model configurations for different T2I models
T2I_MODEL_CONFIGS = {
    "flux": {
        "default_steps": 28,
        "default_guidance": 3.5,
    },
    "sd3": {
        "default_steps": 28,
        "default_guidance": 7.0,
    },
    "sdxl": {
        "default_steps": 30,
        "default_guidance": 7.5,
    },
    "sd": {
        "default_steps": 50,
        "default_guidance": 7.5,
    },
    "hunyuan": {
        "default_steps": 50,
        "default_guidance": 5.0,
    },
    "pixart": {
        "default_steps": 20,
        "default_guidance": 4.5,
    },
    "cogview": {
        "default_steps": 50,
        "default_guidance": 7.5,
    },
}


class DiffusersBackend(DiffusersLoRAMixin, BaseBackend):
    """Diffusers backend using HuggingFace diffusers library.

    Supports automatic model type detection via DiffusionPipeline
    or explicit pipeline selection via model_type parameter.
    Also supports dynamic LoRA adapter management via the ``DiffusersLoRAMixin``.
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize diffusers backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._pipeline_class = None
        self._model_config = None
        self._init_lora_state()

    def load_model(self, **kwargs) -> None:
        """Load diffusers pipeline.

        If model_type is specified, uses the corresponding pipeline class.
        Otherwise, uses DiffusionPipeline.from_pretrained for auto-detection.

        Args:
            **kwargs: Additional arguments passed to from_pretrained
        """
        from diffusers import DiffusionPipeline

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.optimization_config.torch_dtype, torch.bfloat16)

        # Merge kwargs with defaults
        load_kwargs = {
            "torch_dtype": torch_dtype,
            **kwargs,
        }

        model_type = self.model_type
        if model_type is None:
            # Try to auto-detect from model name
            model_type = detect_model_type(self.model_name)

        # Store model config for default parameters
        if model_type and model_type in T2I_MODEL_CONFIGS:
            self._model_config = T2I_MODEL_CONFIGS[model_type]
        else:
            # Use flux defaults as fallback
            self._model_config = {
                "default_steps": 28,
                "default_guidance": 3.5,
            }

        if model_type and model_type in MODEL_TYPE_HINTS:
            # Use specific pipeline class
            pipeline_class_name = MODEL_TYPE_HINTS[model_type]
            print_rank0(f"Loading pipeline: {pipeline_class_name} for model type: {model_type}")

            try:
                # Try to import the specific pipeline class
                pipeline_class = self._get_pipeline_class(pipeline_class_name)
                self.pipe = pipeline_class.from_pretrained(self.model_name, **load_kwargs)
                self._pipeline_class = pipeline_class_name
            except (ImportError, AttributeError) as e:
                print_rank0(f"Failed to load specific pipeline {pipeline_class_name}: {e}")
                print_rank0("Falling back to DiffusionPipeline auto-detection")
                self.pipe = DiffusionPipeline.from_pretrained(self.model_name, **load_kwargs)
        else:
            # Use auto-detection
            print_rank0(f"Auto-detecting pipeline for: {self.model_name}")
            self.pipe = DiffusionPipeline.from_pretrained(self.model_name, **load_kwargs)
            self._pipeline_class = type(self.pipe).__name__

        print_rank0(f"Loaded pipeline: {type(self.pipe).__name__}")

        # Move to device - handle distributed case with LOCAL_RANK
        device = self.device
        if device == "cuda" and torch.cuda.is_available():
            import os
            from diffgentor.utils.distributed import get_local_rank, is_distributed, get_world_size

            # Check if running in distributed mode:
            # 1. torch.distributed is initialized, OR
            # 2. Environment variables indicate multi-process launch (torchrun)
            env_world_size = int(os.environ.get("WORLD_SIZE", 1))
            env_local_rank = os.environ.get("LOCAL_RANK")

            if is_distributed() or env_world_size > 1 or env_local_rank is not None:
                local_rank = get_local_rank()
                device = f"cuda:{local_rank}"
                torch.cuda.set_device(local_rank)
                print_rank0(f"Distributed mode: using device {device} (world_size={get_world_size()})")

            self.pipe = self.pipe.to(device)
        elif device == "cuda" and not torch.cuda.is_available():
            print_rank0("WARNING: CUDA requested but not available, model will run on CPU")
        elif device != "cuda":
            self.pipe = self.pipe.to(device)

        # Apply optimizations
        self.apply_optimizations()

        self._initialized = True
        print_rank0("Pipeline initialization complete")

    def _get_pipeline_class(self, class_name: str):
        """Get pipeline class by name.

        Args:
            class_name: Pipeline class name

        Returns:
            Pipeline class
        """
        import diffusers

        # Try to get from diffusers
        if hasattr(diffusers, class_name):
            return getattr(diffusers, class_name)

        # Try some common aliases
        class_aliases = {
            "FluxPipeline": "FluxPipeline",
            "StableDiffusion3Pipeline": "StableDiffusion3Pipeline",
            "StableDiffusionXLPipeline": "StableDiffusionXLPipeline",
            "StableDiffusionPipeline": "StableDiffusionPipeline",
        }

        if class_name in class_aliases:
            actual_name = class_aliases[class_name]
            if hasattr(diffusers, actual_name):
                return getattr(diffusers, actual_name)

        raise AttributeError(f"Pipeline class {class_name} not found in diffusers")

    def apply_optimizations(self) -> None:
        """Apply optimization configuration to the pipeline."""
        if self.pipe is None:
            return

        config = self.optimization_config

        # Enable TF32 if requested
        if config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # VAE optimizations
        if config.enable_vae_slicing and hasattr(self.pipe, "vae"):
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()
                print_rank0("Enabled VAE slicing")

        if config.enable_vae_tiling and hasattr(self.pipe, "vae"):
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
                print_rank0("Enabled VAE tiling")

        # CPU offload
        if config.enable_sequential_cpu_offload:
            if hasattr(self.pipe, "enable_sequential_cpu_offload"):
                self.pipe.enable_sequential_cpu_offload()
                print_rank0("Enabled sequential CPU offload")
        elif config.enable_cpu_offload:
            if hasattr(self.pipe, "enable_model_cpu_offload"):
                self.pipe.enable_model_cpu_offload()
                print_rank0("Enabled model CPU offload")

        # xFormers
        if config.enable_xformers:
            try:
                if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print_rank0("Enabled xFormers memory efficient attention")
            except Exception as e:
                print_rank0(f"Failed to enable xFormers: {e}")

        # Attention backend
        if config.attention_backend:
            self._set_attention_backend(config.attention_backend)

        # Fuse QKV projections
        if config.enable_fuse_qkv:
            if hasattr(self.pipe, "fuse_qkv_projections"):
                self.pipe.fuse_qkv_projections()
                print_rank0("Fused QKV projections")

        # torch.compile
        if config.enable_compile:
            self._apply_torch_compile(config)

        # Group offloading
        if config.enable_group_offloading:
            self._apply_group_offloading(config)

        # Layerwise casting
        if config.enable_layerwise_casting:
            self._apply_layerwise_casting(config)

        # Cache acceleration
        if config.cache_type:
            self._apply_cache(config)

    def _set_attention_backend(self, backend: str) -> None:
        """Set attention backend.

        Args:
            backend: Attention backend name (flash, sage, xformers)
        """
        # Try to set on transformer
        if hasattr(self.pipe, "transformer"):
            if hasattr(self.pipe.transformer, "set_attention_backend"):
                try:
                    self.pipe.transformer.set_attention_backend(backend)
                    print_rank0(f"Set attention backend to: {backend}")
                except Exception as e:
                    print_rank0(f"Failed to set attention backend: {e}")
        # Try to set on unet
        elif hasattr(self.pipe, "unet"):
            if hasattr(self.pipe.unet, "set_attention_backend"):
                try:
                    self.pipe.unet.set_attention_backend(backend)
                    print_rank0(f"Set attention backend to: {backend}")
                except Exception as e:
                    print_rank0(f"Failed to set attention backend: {e}")

    def _apply_torch_compile(self, config: OptimizationConfig) -> None:
        """Apply torch.compile optimization.

        Args:
            config: Optimization configuration
        """
        # Configure inductor settings
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        for component_name in config.compile_components:
            component = getattr(self.pipe, component_name, None)
            if component is None:
                continue

            try:
                # Set channels last format
                component.to(memory_format=torch.channels_last)

                # Compile the component
                compiled = torch.compile(
                    component,
                    mode=config.compile_mode,
                    fullgraph=config.compile_fullgraph,
                )
                setattr(self.pipe, component_name, compiled)
                print_rank0(f"Compiled {component_name} with mode={config.compile_mode}")
            except Exception as e:
                print_rank0(f"Failed to compile {component_name}: {e}")

    def _apply_group_offloading(self, config: OptimizationConfig) -> None:
        """Apply group offloading optimization.

        Args:
            config: Optimization configuration
        """
        try:
            if hasattr(self.pipe, "enable_group_offload"):
                self.pipe.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type=config.group_offload_type,
                    use_stream=True,
                )
                print_rank0(f"Enabled group offloading (type={config.group_offload_type})")
        except Exception as e:
            print_rank0(f"Failed to enable group offloading: {e}")

    def _apply_layerwise_casting(self, config: OptimizationConfig) -> None:
        """Apply layerwise casting optimization.

        Args:
            config: Optimization configuration
        """
        dtype_map = {
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        storage_dtype = dtype_map.get(config.storage_dtype, torch.float8_e4m3fn)
        compute_dtype = dtype_map.get(config.compute_dtype, torch.bfloat16)

        # Apply to transformer or unet
        component = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        if component and hasattr(component, "enable_layerwise_casting"):
            try:
                component.enable_layerwise_casting(
                    storage_dtype=storage_dtype,
                    compute_dtype=compute_dtype,
                )
                print_rank0(f"Enabled layerwise casting (storage={config.storage_dtype}, compute={config.compute_dtype})")
            except Exception as e:
                print_rank0(f"Failed to enable layerwise casting: {e}")

    def _apply_cache(self, config: OptimizationConfig) -> None:
        """Apply cache acceleration.

        Args:
            config: Optimization configuration
        """
        cache_type = config.cache_type.lower()
        cache_config = config.cache_config

        if cache_type == "deep_cache":
            self._apply_deep_cache(cache_config)
        elif cache_type == "first_block_cache":
            self._apply_first_block_cache(cache_config)
        elif cache_type == "pab":
            self._apply_pab_cache(cache_config)
        elif cache_type == "faster_cache":
            self._apply_faster_cache(cache_config)
        elif cache_type == "cache_dit":
            self._apply_cache_dit(cache_config)
        else:
            print_rank0(f"Unknown cache type: {cache_type}")

    def _apply_deep_cache(self, cache_config: Dict[str, Any]) -> None:
        """Apply DeepCache optimization."""
        try:
            from DeepCache import DeepCacheSDHelper

            helper = DeepCacheSDHelper(pipe=self.pipe)
            helper.set_params(
                cache_interval=cache_config.get("cache_interval", 3),
                cache_branch_id=cache_config.get("cache_branch_id", 0),
            )
            helper.enable()
            print_rank0("Enabled DeepCache")
        except ImportError:
            print_rank0("DeepCache not installed. Install with: pip install DeepCache")
        except Exception as e:
            print_rank0(f"Failed to enable DeepCache: {e}")

    def _apply_first_block_cache(self, cache_config: Dict[str, Any]) -> None:
        """Apply FirstBlockCache optimization."""
        try:
            from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig

            fbc_config = FirstBlockCacheConfig(
                threshold=cache_config.get("threshold", 0.2),
            )
            component = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
            if component:
                apply_first_block_cache(component, fbc_config)
                print_rank0("Enabled FirstBlockCache")
        except ImportError:
            print_rank0("FirstBlockCache requires diffusers >= 0.31.0")
        except Exception as e:
            print_rank0(f"Failed to enable FirstBlockCache: {e}")

    def _apply_pab_cache(self, cache_config: Dict[str, Any]) -> None:
        """Apply Pyramid Attention Broadcast cache."""
        try:
            from diffusers import PyramidAttentionBroadcastConfig

            pab_config = PyramidAttentionBroadcastConfig(
                spatial_attention_block_skip_range=cache_config.get("spatial_skip_range", 2),
                spatial_attention_timestep_skip_range=cache_config.get("timestep_skip_range", (100, 800)),
                current_timestep_callback=lambda: self.pipe.current_timestep,
            )

            component = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
            if component and hasattr(component, "enable_cache"):
                component.enable_cache(pab_config)
                print_rank0("Enabled PAB cache")
        except ImportError:
            print_rank0("PAB cache requires diffusers >= 0.31.0")
        except Exception as e:
            print_rank0(f"Failed to enable PAB cache: {e}")

    def _apply_faster_cache(self, cache_config: Dict[str, Any]) -> None:
        """Apply FasterCache optimization."""
        try:
            from diffusers import FasterCacheConfig

            fc_config = FasterCacheConfig(
                spatial_attention_block_skip_range=cache_config.get("spatial_skip_range", 2),
                spatial_attention_timestep_skip_range=cache_config.get("timestep_skip_range", (-1, 681)),
                current_timestep_callback=lambda: self.pipe.current_timestep,
                attention_weight_callback=lambda _: cache_config.get("attention_weight", 0.3),
                unconditional_batch_skip_range=cache_config.get("unconditional_batch_skip_range", 5),
            )

            component = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
            if component and hasattr(component, "enable_cache"):
                component.enable_cache(fc_config)
                print_rank0("Enabled FasterCache")
        except ImportError:
            print_rank0("FasterCache requires diffusers >= 0.31.0")
        except Exception as e:
            print_rank0(f"Failed to enable FasterCache: {e}")

    def _apply_cache_dit(self, cache_config: Dict[str, Any]) -> None:
        """Apply CacheDiT optimization."""
        try:
            import cache_dit

            cache_dit.enable_cache(self.pipe)
            print_rank0("Enabled CacheDiT")
        except ImportError:
            print_rank0("CacheDiT not installed. Install with: pip install cache-dit")
        except Exception as e:
            print_rank0(f"Failed to enable CacheDiT: {e}")

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from prompts.

        Args:
            prompt: Single prompt or list of prompts
            negative_prompt: Optional negative prompt(s)
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps (model default if None)
            guidance_scale: Guidance scale (model default if None)
            num_images_per_prompt: Number of images per prompt
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            List of generated PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Get defaults from model config
        config = self._model_config or {}
        if num_inference_steps is None:
            num_inference_steps = config.get("default_steps", 28)
        if guidance_scale is None:
            guidance_scale = config.get("default_guidance", 3.5)

        # Prepare generator
        generator = None
        if seed is not None:
            device = self.pipe.device if hasattr(self.pipe, "device") else "cuda"
            generator = torch.Generator(device=device).manual_seed(seed)

        # Build pipeline kwargs
        pipe_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images_per_prompt,
            **kwargs,
        }

        if negative_prompt is not None:
            pipe_kwargs["negative_prompt"] = negative_prompt
        if height is not None:
            pipe_kwargs["height"] = height
        if width is not None:
            pipe_kwargs["width"] = width
        if generator is not None:
            pipe_kwargs["generator"] = generator

        # Run generation
        output = self.pipe(**pipe_kwargs)

        # Extract images
        if hasattr(output, "images"):
            return output.images
        else:
            return output

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self._pipeline_class = None
