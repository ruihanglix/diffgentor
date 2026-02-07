# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Launcher module for automatic multi-GPU startup."""

import os
import subprocess
import sys
from argparse import Namespace
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch


class LaunchStrategy(Enum):
    """Launch strategy types."""

    DIRECT = "direct"  # Direct Python execution
    TORCHRUN = "torchrun"  # torchrun for distributed data parallelism
    MULTIPROCESS = "multiprocess"  # Manual multiprocess for tensor parallelism


class Launcher:
    """Launcher for automatic multi-GPU startup.

    Automatically selects the appropriate launch strategy based on
    backend type and GPU configuration.
    """

    def __init__(self, args: Namespace):
        """Initialize launcher.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.num_gpus = self._detect_num_gpus(args.num_gpus)
        self.backend = args.backend
        self.command = args.command

    def _detect_num_gpus(self, num_gpus: Optional[int]) -> int:
        """Detect the number of GPUs to use.

        If num_gpus is specified, use it. Otherwise, auto-detect based on
        CUDA_VISIBLE_DEVICES or all available GPUs.

        Args:
            num_gpus: User-specified number of GPUs, or None for auto-detect

        Returns:
            Number of GPUs to use
        """
        if num_gpus is not None:
            return num_gpus

        # Check CUDA_VISIBLE_DEVICES first
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            # Count GPUs from CUDA_VISIBLE_DEVICES
            gpu_ids = [g.strip() for g in cuda_visible.split(",") if g.strip()]
            detected = len(gpu_ids)
        else:
            # Use all available GPUs
            detected = torch.cuda.device_count()

        # Default to 1 if no GPUs detected
        return max(detected, 1)

    def detect_strategy(self) -> LaunchStrategy:
        """Detect the best launch strategy based on configuration.

        Returns:
            Appropriate launch strategy
        """
        # xDiT always uses torchrun
        if self.backend == "xdit":
            return LaunchStrategy.TORCHRUN

        # OpenAI API doesn't need multi-GPU
        if self.backend == "openai":
            return LaunchStrategy.DIRECT

        # HunyuanImage-3.0 uses device_map="auto" for multi-GPU, not torchrun
        if self.backend == "hunyuan_image_3":
            from diffgentor.utils.env import HunyuanImage3Env
            gpus_per_model = HunyuanImage3Env.gpus_per_model()
            # If gpus_per_model is set and we have enough GPUs for multiple instances
            if gpus_per_model > 0 and self.num_gpus // gpus_per_model > 1:
                return LaunchStrategy.MULTIPROCESS
            # Default: single process with device_map="auto"
            return LaunchStrategy.DIRECT

        # DeepGen: each instance runs on its own GPU(s)
        if self.backend == "deepgen":
            from diffgentor.utils.env import DeepGenEnv
            gpus_per_model = DeepGenEnv.gpus_per_model()
            if self.num_gpus > 1:
                return LaunchStrategy.MULTIPROCESS
            return LaunchStrategy.DIRECT

        # Check for special models requiring tensor parallelism
        model_type = getattr(self.args, "model_type", None)
        if model_type == "emu35":
            from diffgentor.utils.env import Emu35Env

            gpus_per_model = Emu35Env.gpus_per_model()
            if gpus_per_model > 1:
                return LaunchStrategy.MULTIPROCESS

        if model_type == "hunyuan_image_3":
            from diffgentor.utils.env import HunyuanImage3Env

            gpus_per_model = HunyuanImage3Env.gpus_per_model()
            if gpus_per_model > 0 and self.num_gpus // gpus_per_model > 1:
                return LaunchStrategy.MULTIPROCESS
            return LaunchStrategy.DIRECT

        # Single GPU
        if self.num_gpus <= 1:
            return LaunchStrategy.DIRECT

        # Multi-GPU -> torchrun for data parallelism
        return LaunchStrategy.TORCHRUN

    def run(self) -> int:
        """Execute the command with appropriate launch strategy.

        Returns:
            Exit code
        """
        strategy = self.detect_strategy()

        print(f"Selected launch strategy: {strategy.value}")
        print(f"Backend: {self.backend}, GPUs: {self.num_gpus}")

        if strategy == LaunchStrategy.DIRECT:
            return self._run_direct()
        elif strategy == LaunchStrategy.TORCHRUN:
            return self._run_torchrun()
        elif strategy == LaunchStrategy.MULTIPROCESS:
            return self._run_multiprocess()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _run_direct(self) -> int:
        """Run directly without any launcher.

        Returns:
            Exit code
        """
        if self.command == "t2i":
            return self._run_t2i_worker()
        elif self.command == "edit":
            return self._run_edit_worker()
        else:
            raise ValueError(f"Unknown command: {self.command}")

    def _run_torchrun(self) -> int:
        """Run with torchrun for distributed data parallelism.

        Returns:
            Exit code
        """
        worker_script = self._get_worker_script()

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.num_gpus}",
            worker_script,
        ]

        # Add worker arguments
        cmd.extend(self._get_worker_args())

        print(f"Running: {' '.join(cmd)}")
        return subprocess.call(cmd)

    def _run_multiprocess(self) -> int:
        """Run with manual multiprocess for tensor parallelism.

        Returns:
            Exit code
        """
        # Get GPUs per model instance based on backend or model_type
        model_type = getattr(self.args, "model_type", None)

        if self.backend == "hunyuan_image_3" or model_type == "hunyuan_image_3":
            from diffgentor.utils.env import HunyuanImage3Env
            gpus_per_model = HunyuanImage3Env.gpus_per_model()
        elif self.backend == "deepgen" or model_type == "deepgen":
            from diffgentor.utils.env import DeepGenEnv
            gpus_per_model = DeepGenEnv.gpus_per_model()
        elif model_type == "emu35":
            from diffgentor.utils.env import Emu35Env
            gpus_per_model = Emu35Env.gpus_per_model()
        else:
            # Fallback - shouldn't reach here if detect_strategy() is correct
            gpus_per_model = 1

        num_instances = self.num_gpus // gpus_per_model

        if num_instances == 0:
            print(f"Error: Not enough GPUs ({self.num_gpus}) for {gpus_per_model} GPUs per model")
            return 1

        print(f"Launching {num_instances} instances with {gpus_per_model} GPUs each")

        # Get available GPUs
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            gpu_list = [int(g) for g in cuda_visible.split(",")]
        else:
            gpu_list = list(range(self.num_gpus))

        # Launch processes
        processes = []
        worker_script = self._get_worker_script()

        for i in range(num_instances):
            # Calculate GPU assignment
            start_idx = i * gpus_per_model
            instance_gpus = ",".join(str(gpu_list[start_idx + j]) for j in range(gpus_per_model))

            # Build command
            cmd = [sys.executable, worker_script]
            cmd.extend(self._get_worker_args())
            cmd.extend(["--local_rank", str(i), "--num_gpus", str(num_instances)])

            # Set environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = instance_gpus

            print(f"Instance {i}: CUDA_VISIBLE_DEVICES={instance_gpus}")
            proc = subprocess.Popen(cmd, env=env)
            processes.append((i, proc))

        # Wait for all processes
        failed = 0
        for i, proc in processes:
            ret = proc.wait()
            if ret != 0:
                print(f"Instance {i} failed with exit code {ret}")
                failed += 1

        return 1 if failed > 0 else 0

    def _get_worker_script(self) -> str:
        """Get path to worker script.

        Returns:
            Path to worker script
        """
        if self.command == "t2i":
            return str(Path(__file__).parent.parent / "workers" / "t2i_worker.py")
        elif self.command == "edit":
            return str(Path(__file__).parent.parent / "workers" / "edit_worker.py")
        else:
            raise ValueError(f"Unknown command: {self.command}")

    def _get_worker_args(self) -> List[str]:
        """Build worker arguments from parsed args.

        Returns:
            List of worker arguments
        """
        args = []

        # Common arguments
        args.extend(["--backend", self.args.backend])
        args.extend(["--model_name", self.args.model_name])

        if self.args.model_type:
            args.extend(["--model_type", self.args.model_type])

        args.extend(["--device", self.args.device])
        args.extend(["--torch_dtype", self.args.torch_dtype])

        if self.args.seed is not None:
            args.extend(["--seed", str(self.args.seed)])

        # Optimization arguments
        if self.args.optimize:
            args.extend(["--optimize", self.args.optimize])
        if self.args.attention_backend:
            args.extend(["--attention_backend", self.args.attention_backend])
        if self.args.cache_type:
            args.extend(["--cache_type", self.args.cache_type])
        if self.args.enable_compile:
            args.append("--enable_compile")
        if self.args.enable_cpu_offload:
            args.append("--enable_cpu_offload")
        if self.args.enable_vae_slicing:
            args.append("--enable_vae_slicing")
        if self.args.enable_vae_tiling:
            args.append("--enable_vae_tiling")

        # xDiT arguments are read from DG_XDIT_* environment variables, not CLI args
        # See docs/optimization.md for configuration details

        # Distributed arguments
        args.extend(["--node_rank", str(self.args.node_rank)])
        args.extend(["--num_nodes", str(self.args.num_nodes)])

        # Logging arguments
        if getattr(self.args, "log_dir", None):
            args.extend(["--log_dir", self.args.log_dir])

        # API backend arguments (for openai, google_genai)
        if self.backend in ("openai", "google_genai", "gemini"):
            args.extend(["--timeout", str(self.args.timeout)])
            args.extend(["--api_max_retries", str(self.args.api_max_retries)])
            args.extend(["--retry_delay", str(self.args.retry_delay)])
            args.extend(["--max_global_workers", str(self.args.max_global_workers)])
            args.extend(["--num_processes", str(self.args.num_processes)])

        # Command-specific arguments
        if self.command == "t2i":
            args.extend(self._get_t2i_args())
        elif self.command == "edit":
            args.extend(self._get_edit_args())

        return args

    def _get_t2i_args(self) -> List[str]:
        """Get T2I specific arguments.

        Returns:
            List of T2I arguments
        """
        args = []

        if self.args.prompt:
            args.extend(["--prompt", self.args.prompt])
        if self.args.prompts_file:
            args.extend(["--prompts_file", self.args.prompts_file])

        args.extend(["--output_dir", self.args.output_dir])
        args.extend(["--num_images_per_prompt", str(self.args.num_images_per_prompt)])

        if self.args.height:
            args.extend(["--height", str(self.args.height)])
        if self.args.width:
            args.extend(["--width", str(self.args.width)])

        if self.args.num_inference_steps is not None:
            args.extend(["--num_inference_steps", str(self.args.num_inference_steps)])
        if self.args.guidance_scale is not None:
            args.extend(["--guidance_scale", str(self.args.guidance_scale)])

        if self.args.negative_prompt:
            args.extend(["--negative_prompt", self.args.negative_prompt])

        args.extend(["--batch_size", str(self.args.batch_size)])
        args.extend(["--max_retries", str(self.args.max_retries)])

        if self.args.no_resume:
            args.append("--no_resume")

        return args

    def _get_edit_args(self) -> List[str]:
        """Get editing specific arguments.

        Returns:
            List of editing arguments
        """
        args = []

        args.extend(["--input", self.args.input_data])
        args.extend(["--output_dir", self.args.output_dir])

        if self.args.output_csv:
            args.extend(["--output_csv", self.args.output_csv])

        args.extend(["--instruction_key", self.args.instruction_key])

        if self.args.image_cache_dir:
            args.extend(["--image_cache_dir", self.args.image_cache_dir])

        args.extend(["--batch_size", str(self.args.batch_size)])
        if self.args.num_inference_steps is not None:
            args.extend(["--num_inference_steps", str(self.args.num_inference_steps)])
        if self.args.guidance_scale is not None:
            args.extend(["--guidance_scale", str(self.args.guidance_scale)])
        if self.args.true_cfg_scale is not None:
            args.extend(["--true_cfg_scale", str(self.args.true_cfg_scale)])
        args.extend(["--negative_prompt", self.args.negative_prompt])
        args.extend(["--max_retries", str(self.args.max_retries)])

        if self.args.filter_rows:
            args.extend(["--filter_rows", self.args.filter_rows])

        if self.args.no_resume:
            args.append("--no_resume")

        # Prompt enhancement
        if getattr(self.args, "prompt_enhance_type", None):
            args.extend(["--prompt_enhance_type", self.args.prompt_enhance_type])

        # Note: Model-specific arguments (bagel, step1x, emu35, dreamomni2, flux_kontext_official)
        # are configured via DG_* environment variables, not CLI arguments.

        return args

    def _run_t2i_worker(self) -> int:
        """Run T2I worker directly.

        Returns:
            Exit code
        """
        from diffgentor.workers.t2i_worker import run_t2i

        return run_t2i(self.args)

    def _run_edit_worker(self) -> int:
        """Run edit worker directly.

        Returns:
            Exit code
        """
        from diffgentor.workers.edit_worker import run_edit

        return run_edit(self.args)
