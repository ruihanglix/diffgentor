# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""xDiT backend implementation using xfuser for multi-GPU parallel inference."""

import copy
from typing import Any, List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0


class XDiTBackend(BaseBackend):
    """xDiT backend using xfuser for multi-GPU parallel inference.

    Supports various parallelism strategies:
    - Data parallelism
    - Sequence parallelism (Ulysses, Ring)
    - Pipeline parallelism (PipeFusion)
    - CFG parallelism
    - Batch inference (multiple images per forward pass)

    Note: This backend requires torchrun for multi-GPU execution.
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize xDiT backend.

        Args:
            backend_config: Backend configuration with xDiT specific params
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self.engine_config = None
        self.input_config = None
        self._world_group = None
        self._runtime_state = None

    def load_model(self, **kwargs) -> None:
        """Load and wrap diffusers pipeline with xDiT parallel.

        Args:
            **kwargs: Additional arguments passed to pipeline loading
        """
        try:
            from xfuser import xFuserArgs, xDiTParallel
            from xfuser.core.distributed import (
                get_world_group,
                init_distributed_environment,
                get_runtime_state,
            )
        except ImportError:
            raise ImportError(
                "xfuser is required for xDiT backend. "
                "Install with: pip install xfuser"
            )

        # Initialize distributed environment if not already done
        init_distributed_environment(
            rank=int(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0),
            world_size=int(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1),
        )

        self._world_group = get_world_group()
        self._runtime_state = get_runtime_state()
        local_rank = self._world_group.local_rank

        # Create xFuser configuration
        engine_args = self._create_engine_args()
        self.engine_config, self.input_config = engine_args.create_config()

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.optimization_config.torch_dtype, torch.bfloat16)

        # Load base pipeline
        from diffusers import DiffusionPipeline

        print_rank0(f"Loading pipeline: {self.model_name}")
        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            **kwargs,
        ).to(f"cuda:{local_rank}")

        # Apply basic optimizations before wrapping
        self._apply_pre_wrap_optimizations(pipe)

        # Wrap with xDiT parallel
        print_rank0("Wrapping pipeline with xDiT parallel...")
        self.pipe = xDiTParallel(pipe, self.engine_config, self.input_config)

        self._initialized = True
        print_rank0(f"xDiT pipeline initialized on rank {local_rank}")

    def _create_engine_args(self):
        """Create xFuser engine arguments from config.

        Returns:
            xFuserArgs instance
        """
        from xfuser import xFuserArgs

        config = self.backend_config

        # Build args dict
        args_dict = {
            "model": self.model_name,
            "data_parallel_degree": config.data_parallel_degree,
            "ulysses_degree": config.ulysses_degree,
            "ring_degree": config.ring_degree,
            "pipefusion_parallel_degree": config.pipefusion_degree,
            "use_cfg_parallel": config.use_cfg_parallel,
        }

        # Create xFuserArgs
        return xFuserArgs(**args_dict)

    def _apply_pre_wrap_optimizations(self, pipe: Any) -> None:
        """Apply optimizations before xDiT wrapping.

        Some optimizations need to be applied before wrapping with xDiTParallel.

        Args:
            pipe: Pipeline instance
        """
        config = self.optimization_config

        # Enable TF32
        if config.enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # VAE optimizations
        if config.enable_vae_slicing and hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()

        if config.enable_vae_tiling and hasattr(pipe, "vae"):
            if hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

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
        """Generate images using xDiT parallel inference with batch support.

        Args:
            prompt: Single prompt or list of prompts
            negative_prompt: Optional negative prompt(s)
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps (default: 28)
            guidance_scale: Guidance scale (default: 3.5)
            num_images_per_prompt: Number of images per prompt
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            List of generated PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Apply defaults if not specified
        if num_inference_steps is None:
            num_inference_steps = 28
        if guidance_scale is None:
            guidance_scale = 3.5

        # Use input_config defaults if not specified
        if height is None and self.input_config:
            height = self.input_config.height
        if width is None and self.input_config:
            width = self.input_config.width

        # Prepare generator
        generator = None
        if seed is not None:
            local_rank = self._world_group.local_rank if self._world_group else 0
            generator = torch.Generator(device=f"cuda:{local_rank}").manual_seed(seed)

        # Normalize prompt to list
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        # Normalize negative_prompt to list
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompts = [negative_prompt] * len(prompts)
            else:
                negative_prompts = list(negative_prompt)
        else:
            negative_prompts = None

        # Get batch size from config
        batch_size = self.backend_config.batch_size

        # Set runtime state batch_size for xDiT
        if self._runtime_state is not None:
            try:
                from xfuser.core.distributed import get_pipeline_parallel_world_size
                self._runtime_state.set_input_parameters(
                    batch_size=min(batch_size, len(prompts)),
                    num_inference_steps=num_inference_steps,
                    split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
                )
            except Exception as e:
                print_rank0(f"Warning: Could not set runtime state parameters: {e}")

        # Build base pipeline kwargs
        base_pipe_kwargs = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }

        if height is not None:
            base_pipe_kwargs["height"] = height
        if width is not None:
            base_pipe_kwargs["width"] = width
        if generator is not None:
            base_pipe_kwargs["generator"] = generator

        # Generate images in batches
        all_images = []

        if batch_size >= len(prompts):
            # Single batch - process all prompts at once
            pipe_kwargs = copy.deepcopy(base_pipe_kwargs)
            pipe_kwargs["prompt"] = prompts
            if negative_prompts is not None:
                pipe_kwargs["negative_prompt"] = negative_prompts

            output = self.pipe(**pipe_kwargs)
            if hasattr(output, "images"):
                all_images.extend(output.images)
            else:
                all_images.extend(output)
        else:
            # Multiple batches - split prompts into batches
            num_batches = (len(prompts) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(prompts))

                batch_prompts = prompts[start_idx:end_idx]
                batch_negative_prompts = (
                    negative_prompts[start_idx:end_idx]
                    if negative_prompts is not None
                    else None
                )

                # Update runtime state for this batch size
                current_batch_size = len(batch_prompts)
                if self._runtime_state is not None:
                    try:
                        from xfuser.core.distributed import get_pipeline_parallel_world_size
                        self._runtime_state.set_input_parameters(
                            batch_size=current_batch_size,
                            num_inference_steps=num_inference_steps,
                            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
                        )
                    except Exception:
                        pass

                pipe_kwargs = copy.deepcopy(base_pipe_kwargs)
                pipe_kwargs["prompt"] = batch_prompts
                if batch_negative_prompts is not None:
                    pipe_kwargs["negative_prompt"] = batch_negative_prompts

                print_rank0(
                    f"Processing batch {batch_idx + 1}/{num_batches} "
                    f"(prompts {start_idx}-{end_idx - 1})"
                )

                output = self.pipe(**pipe_kwargs)
                if hasattr(output, "images"):
                    all_images.extend(output.images)
                else:
                    all_images.extend(output)

        return all_images

    def batch_generate(
        self,
        prompts: List[str],
        negative_prompts: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from a list of prompts using batch inference.

        This is a convenience method that calls generate() with proper batching.
        The batch_size is determined by backend_config.batch_size.

        Args:
            prompts: List of prompts
            negative_prompts: Optional list of negative prompts
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps (default: 28)
            guidance_scale: Guidance scale (default: 3.5)
            seed: Random seed
            **kwargs: Additional pipeline arguments

        Returns:
            List of generated PIL Images (one per prompt)
        """
        return self.generate(
            prompt=prompts,
            negative_prompt=negative_prompts,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs,
        )

    def save_images(self, output_dir: str, prefix: str = "xdit") -> List[str]:
        """Save generated images using xDiT's built-in save method.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of saved file paths
        """
        if hasattr(self.pipe, "save"):
            self.pipe.save(output_dir, prefix)
            return []  # xDiT handles saving internally
        return []

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        self.engine_config = None
        self.input_config = None
        self._world_group = None
        self._runtime_state = None


def get_xdit_launch_command(
    script_path: str,
    num_gpus: int,
    backend_config: BackendConfig,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Build torchrun command for xDiT execution.

    Note: xDiT parallelism parameters (data_parallel_degree, ulysses_degree,
    ring_degree, pipefusion_degree, use_cfg_parallel) are read from DG_XDIT_*
    environment variables.

    Args:
        script_path: Path to Python script to execute
        num_gpus: Number of GPUs to use
        backend_config: Backend configuration
        extra_args: Additional command line arguments

    Returns:
        List of command parts for subprocess
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        script_path,
        "--backend", "xdit",
        "--model_name", backend_config.model_name,
    ]

    if extra_args:
        cmd.extend(extra_args)

    return cmd
