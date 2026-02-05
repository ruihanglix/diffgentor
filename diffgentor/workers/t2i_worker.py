# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""T2I worker for text-to-image generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from diffgentor.config import BackendConfig, OptimizationConfig, T2IConfig
from diffgentor.utils.exceptions import GenerationError, log_error
from diffgentor.utils.image import get_output_filename
from diffgentor.utils.logging import print_rank0
from diffgentor.workers.base import BaseWorker


class T2IWorker(BaseWorker):
    """Worker for text-to-image generation."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = self.build_config()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add T2I-specific arguments."""
        # Input/Output
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--prompts_file", type=str, default=None)
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument(
            "--output_name_column",
            type=str,
            default=None,
            help="Column name for custom output filename (supports paths like 'a/b/c.png')",
        )

        # Generation
        parser.add_argument("--num_images_per_prompt", type=int, default=1)
        parser.add_argument("--height", type=int, default=None)
        parser.add_argument("--width", type=int, default=None)
        parser.add_argument("--num_inference_steps", type=int, default=None)
        parser.add_argument("--guidance_scale", type=float, default=None)
        parser.add_argument("--negative_prompt", type=str, default=None)

        # Resume
        parser.add_argument("--no_resume", action="store_true")

    def build_config(self) -> T2IConfig:
        """Build T2I configuration from arguments."""
        from diffgentor.optimizations.manager import parse_optimization_string

        args = self.args

        # Build optimization config
        if args.optimize:
            opt_config = parse_optimization_string(args.optimize)
        else:
            opt_config = OptimizationConfig()

        # Override with explicit arguments
        opt_config.torch_dtype = args.torch_dtype
        if args.attention_backend:
            opt_config.attention_backend = args.attention_backend
        if args.cache_type:
            opt_config.cache_type = args.cache_type
        if args.enable_compile:
            opt_config.enable_compile = True
        if args.enable_cpu_offload:
            opt_config.enable_cpu_offload = True
        if args.enable_vae_slicing:
            opt_config.enable_vae_slicing = True
        if args.enable_vae_tiling:
            opt_config.enable_vae_tiling = True

        # Build backend config (xDiT params loaded from DG_XDIT_* env vars)
        backend_config = BackendConfig(
            backend=args.backend,
            model_name=args.model_name,
            model_type=args.model_type,
            device=args.device,
            num_gpus=args.num_gpus or 1,
            seed=args.seed,
        )

        return T2IConfig(
            backend_config=backend_config,
            optimization_config=opt_config,
            prompt=args.prompt,
            prompts_file=args.prompts_file,
            output_dir=args.output_dir,
            output_name_column=getattr(args, "output_name_column", None),
            num_images_per_prompt=args.num_images_per_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            resume=not args.no_resume,
            node_rank=args.node_rank,
            num_nodes=args.num_nodes,
        )

    def _load_prompts(self) -> list[dict[str, Any]]:
        """Load prompts from config."""
        if self.config.prompt:
            return [{"prompt": self.config.prompt, "index": 0}]

        if self.config.prompts_file:
            from diffgentor.utils.data import load_prompts

            return load_prompts(self.config.prompts_file)

        return []

    def run(self) -> int:
        """Run T2I generation."""
        config = self.config
        args = self.args

        # Print warning about T2I functionality
        print_rank0(
            "\n" + "=" * 80 + "\n"
            "[WARNING] The T2I (text-to-image) functionality has not been fully tested.\n"
            "Correctness is not guaranteed. It is recommended to use 'diffgentor edit'\n"
            "for image editing tasks, which has been more thoroughly validated.\n"
            + "=" * 80 + "\n"
        )

        self.print_header(
            "T2I Generation",
            Backend=config.backend_config.backend,
            Model=config.backend_config.model_name,
            Output=config.output_dir,
            Node=f"{config.node_rank}/{config.num_nodes}",
            Process=f"{self.process_rank}/{self.num_processes}",
        )

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Get backend
        from diffgentor.backends import get_backend

        backend = get_backend(config.backend_config, config.optimization_config)

        try:
            # Load model
            print_rank0("Loading model...")
            model_kwargs = {
                "timeout": args.timeout,
                "max_retries": args.api_max_retries,
                "retry_delay": args.retry_delay,
                "max_global_workers": args.max_global_workers,
                "num_processes": args.num_processes,
            }
            backend.load_model(**model_kwargs)

            # Load prompts
            all_prompts = self._load_prompts()
            print_rank0(f"Loaded {len(all_prompts)} prompts")

            if not all_prompts:
                print_rank0("No prompts to process")
                return 0

            # Resume: filter completed tasks
            if config.resume:
                from diffgentor.utils.image import normalize_custom_output_path

                def get_output_path(prompt_data):
                    idx = prompt_data.get("index", 0)
                    # Check for custom output name
                    custom_name = None
                    if config.output_name_column:
                        custom_name = prompt_data.get(config.output_name_column)
                        if custom_name:
                            custom_name = str(custom_name).strip()
                            if not custom_name:
                                custom_name = None

                    if custom_name:
                        # Use custom output path
                        sub_index = None if config.num_images_per_prompt == 1 else 0
                        output_filename = normalize_custom_output_path(custom_name, sub_index)
                        return Path(config.output_dir) / output_filename
                    else:
                        # Use default index-based naming
                        if config.num_images_per_prompt == 1:
                            return Path(config.output_dir) / get_output_filename(idx)
                        return Path(config.output_dir) / get_output_filename(idx, 0)

                all_prompts, skipped = self.filter_completed(all_prompts, get_output_path)
                print_rank0(
                    f"Resume: total={skipped + len(all_prompts)}, "
                    f"skipped={skipped}, remaining={len(all_prompts)}"
                )

                if not all_prompts:
                    print_rank0("All tasks already completed")
                    return 0

            # Distribute tasks
            _, prompts = self.distribute_tasks(
                all_prompts, config.node_rank, config.num_nodes
            )

            if not prompts:
                return 0

            # Generate
            total = len(prompts)
            success = 0
            failed = 0

            for i, prompt_data in enumerate(prompts):
                prompt = prompt_data.get("prompt", "")
                idx = prompt_data.get("index", i)

                self._log(f"[{i + 1}/{total}] Generating for prompt {idx}...")

                try:
                    images = backend.generate(
                        prompt=prompt,
                        negative_prompt=config.negative_prompt,
                        height=config.height,
                        width=config.width,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        num_images_per_prompt=config.num_images_per_prompt,
                        seed=config.backend_config.seed,
                    )

                    # Save images - use custom filename if available
                    custom_name = None
                    if config.output_name_column:
                        custom_name = prompt_data.get(config.output_name_column)
                        if custom_name:
                            custom_name = str(custom_name).strip()
                            if not custom_name:
                                custom_name = None

                    for j, img in enumerate(images):
                        sub_index = None if config.num_images_per_prompt == 1 else j

                        if custom_name:
                            # Use custom output path
                            from diffgentor.utils.image import save_image_to_custom_path

                            filename = save_image_to_custom_path(
                                img, config.output_dir, custom_name, sub_index
                            )
                        else:
                            # Use default index-based naming
                            filename = get_output_filename(idx, sub_index)
                            output_path = Path(config.output_dir) / filename
                            img.save(output_path)

                        self._log(f"Saved: {filename}")

                    success += 1

                except Exception as e:
                    log_error(
                        GenerationError(f"Failed to generate prompt {idx}", cause=e),
                        context=f"generate[{idx}]",
                        include_traceback=True,
                    )
                    failed += 1

            self.print_summary(success, failed)
            return 0 if failed == 0 else 1

        finally:
            backend.cleanup()


def parse_args() -> argparse.Namespace:
    """Parse worker arguments."""
    parser = T2IWorker.create_parser("T2I Worker")
    return parser.parse_args()


def run_t2i(args: argparse.Namespace) -> int:
    """Run T2I generation.

    Args:
        args: Parsed arguments or Namespace object

    Returns:
        Exit code
    """
    worker = T2IWorker(args)
    return worker.run_with_error_handling(worker.run)


def main():
    """Main entry point."""
    args = parse_args()
    sys.exit(run_t2i(args))


if __name__ == "__main__":
    main()
