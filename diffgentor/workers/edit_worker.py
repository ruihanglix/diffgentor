# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Edit worker for image editing."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

from diffgentor.config import BackendConfig, EditingConfig, OptimizationConfig
from diffgentor.utils.env import get_env_str
from diffgentor.utils.exceptions import EditingError, ImageLoadError, log_error
from diffgentor.utils.image import get_output_filename
from diffgentor.utils.logging import print_rank0
from diffgentor.workers.base import BaseWorker


class EditWorker(BaseWorker):
    """Worker for image editing."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.config = self.build_config()
        self._dataset = None
        self._pending_indices: List[int] = []

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add editing-specific arguments."""
        # Input/Output
        parser.add_argument("--input", type=str, required=True, dest="input_data")
        parser.add_argument("--output_dir", type=str, default="./output")
        parser.add_argument("--output_csv", type=str, default=None)
        parser.add_argument(
            "--output_name_column",
            type=str,
            default=None,
            help="Column name for custom output filename (supports paths like 'a/b/c.png')",
        )
        parser.add_argument("--instruction_key", type=str, default="instruction")
        parser.add_argument("--image_cache_dir", type=str, default=None)

        # Editing parameters
        parser.add_argument("--num_inference_steps", type=int, default=None)
        parser.add_argument("--guidance_scale", type=float, default=None)
        parser.add_argument("--true_cfg_scale", type=float, default=None)
        parser.add_argument("--negative_prompt", type=str, default=" ")

        # Filtering and resume
        parser.add_argument("--filter_rows", type=str, default=None)
        parser.add_argument("--no_resume", action="store_true")

        # Prompt enhancement
        parser.add_argument(
            "--prompt_enhance_type",
            type=str,
            default=None,
            choices=["qwen_image_edit", "glm_image", "flux2"],
            help="Type of prompt enhancement to use",
        )

    def build_config(self) -> EditingConfig:
        """Build editing configuration from arguments."""
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

        # Build backend config
        backend_config = BackendConfig(
            backend=args.backend,
            model_name=args.model_name,
            model_type=args.model_type,
            device=args.device,
            num_gpus=args.num_gpus or 1,
            seed=args.seed,
        )

        # API backend kwargs
        model_kwargs = {
            "timeout": args.timeout,
            "max_retries": args.api_max_retries,
            "retry_delay": args.retry_delay,
            "max_global_workers": args.max_global_workers,
            "num_processes": args.num_processes,
        }

        # Determine output_csv
        output_csv = args.output_csv
        if output_csv is None:
            output_csv = str(Path(args.output_dir) / "results.csv")

        return EditingConfig(
            backend_config=backend_config,
            optimization_config=opt_config,
            input_data=args.input_data,
            output_dir=args.output_dir,
            output_csv=output_csv,
            output_name_column=getattr(args, "output_name_column", None),
            instruction_key=args.instruction_key,
            image_cache_dir=args.image_cache_dir,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            true_cfg_scale=args.true_cfg_scale,
            negative_prompt=args.negative_prompt,
            max_retries=args.max_retries,
            filter_rows=args.filter_rows,
            resume=not args.no_resume,
            node_rank=args.node_rank,
            num_nodes=args.num_nodes,
            model_kwargs=model_kwargs,
            prompt_enhance_type=getattr(args, "prompt_enhance_type", None),
            prompt_enhance_api_key=get_env_str("PROMPT_ENHANCER_API_KEY"),
            prompt_enhance_api_base=get_env_str("PROMPT_ENHANCER_API_BASE"),
            prompt_enhance_model=get_env_str("PROMPT_ENHANCER_MODEL"),
        )

    def _apply_filter(
        self, rows: list[dict[str, Any]], filter_expr: str
    ) -> list[dict[str, Any]]:
        """Apply filter expression to rows."""
        indices = set()
        parts = filter_expr.split(",")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if ":" in part:
                range_parts = part.split(":", 1)
                start = int(range_parts[0]) if range_parts[0] else 0
                end = int(range_parts[1]) if range_parts[1] else len(rows)
                indices.update(range(start, end))
            else:
                indices.add(int(part))

        return [row for i, row in enumerate(rows) if i in indices]

    def _coerce_image(self, obj: Any) -> Image.Image | None:
        """Coerce common dataset image representations to PIL.Image."""
        if obj is None:
            return None
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, dict) and "bytes" in obj and obj["bytes"] is not None:
            try:
                return Image.open(io.BytesIO(obj["bytes"]))
            except Exception:
                return None
        # Handle numpy arrays
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return Image.fromarray(obj)
        except ImportError:
            pass
        except Exception:
            return None
        return None

    def _load_images(self, row: dict[str, Any]) -> list[Image.Image] | None:
        """Load input images for a row."""
        from diffgentor.utils.image import download_image, parse_input_images

        config = self.config

        # Check for embedded images (parquet)
        if row.get("_images_embedded", False):
            input_images = row.get("input_images", [])
            if input_images is None:
                return []
            try:
                if len(input_images) == 0:
                    return []
            except TypeError:
                return []

            images = []
            for item in input_images:
                img = self._coerce_image(item)
                if img is not None:
                    images.append(img)
            return images

        # CSV path
        input_images_str = row.get("input_images", "[]")
        image_urls = parse_input_images(input_images_str)

        if not image_urls:
            return []

        images = []
        for url in image_urls:
            # Resolve relative paths
            if not url.startswith(("http://", "https://")):
                if not Path(url).is_absolute():
                    url = str(Path(config.input_csv_dir) / url)

            img = download_image(url, config.image_cache_dir)
            if img is None:
                log_error(
                    ImageLoadError(f"Failed to download image: {url}"),
                    context="_load_images",
                    include_traceback=False,
                )
                return None
            images.append(img)

        return images

    def _get_csv_suffix(self) -> str:
        """Get CSV filename suffix based on node and process rank."""
        config = self.config
        parts = []

        if config.num_nodes > 1:
            parts.append(f"node{config.node_rank}")

        if self.num_processes > 1:
            parts.append(f"process{self.process_rank}")

        if parts:
            return "_" + "_".join(parts)
        return ""

    def _scan_data(self) -> bool:
        """
        Fast scan data and determine pending tasks.

        This method:
        1. Quickly scans the dataset indices (no image decoding)
        2. Checks which outputs already exist (resume)
        3. Distributes tasks across nodes/processes
        4. Prints detailed statistics

        Returns:
            True if there are pending tasks, False otherwise
        """
        from diffgentor.utils.data import create_lazy_dataset

        config = self.config

        print_rank0("=" * 60)
        print_rank0("Scanning data...")
        print_rank0("=" * 60)

        # Create lazy dataset (index_only mode for fast scanning)
        # Pass output_name_column for custom filename support
        self._dataset = create_lazy_dataset(
            config.input_data,
            load_mode="index_only",
            output_name_column=config.output_name_column,
        )

        # Get output filename generator (fallback when custom name not available)
        def _get_output_filename(idx: int) -> str:
            return get_output_filename(idx)

        # Scan and filter with distribution
        stats, self._pending_indices = self._dataset.scan_and_filter(
            output_dir=config.output_dir,
            get_output_filename=_get_output_filename,
            node_rank=config.node_rank,
            num_nodes=config.num_nodes,
            process_rank=self.process_rank,
            num_processes=self.num_processes,
        )

        # Print statistics
        print_rank0(f"  Source: {stats.source_file}")
        print_rank0(f"  Total samples: {stats.total}")
        if config.output_name_column:
            print_rank0(f"  Output name column: {config.output_name_column}")

        if config.resume:
            print_rank0(f"  Completed (skipped): {stats.completed}")
            all_pending = stats.total - stats.completed
            print_rank0(f"  All pending: {all_pending}")
        else:
            all_pending = stats.total

        # Distribution info
        if config.num_nodes > 1:
            print_rank0(f"  Node {config.node_rank}/{config.num_nodes}")

        if self.num_processes > 1:
            print_rank0(f"  Process {self.process_rank}/{self.num_processes}")

        print_rank0(f"  This process will handle: {stats.pending} samples")
        print_rank0("=" * 60)

        if stats.pending == 0:
            print_rank0("No pending tasks for this process.")
            return False

        return True

    def run(self) -> int:
        """Run image editing."""
        config = self.config

        self.print_header(
            "Image Editing",
            Backend=config.backend_config.backend,
            Model=config.backend_config.model_name,
            ModelType=config.backend_config.model_type,
            Input=config.input_data,
            Output=config.output_dir,
            Node=f"{config.node_rank}/{config.num_nodes}",
            Process=f"{self.process_rank}/{self.num_processes}",
            PromptEnhancement=config.prompt_enhance_type,
            PromptEnhanceModel=config.prompt_enhance_model or "default",
        )

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Step 1: Fast scan data BEFORE loading model
        if not self._scan_data():
            return 0

        # Step 2: Now load the model (expensive operation)
        print_rank0("Loading model...")

        # Initialize prompt enhancer
        prompt_enhancer = None
        if config.prompt_enhance_type:
            from diffgentor.prompt_enhance import get_prompt_enhancer

            prompt_enhancer = get_prompt_enhancer(
                enhancer_type=config.prompt_enhance_type,
                api_key=config.prompt_enhance_api_key,
                api_base=config.prompt_enhance_api_base,
                model=config.prompt_enhance_model,
                max_retries=config.max_retries,
            )
            print_rank0(f"Initialized prompt enhancer: {config.prompt_enhance_type}")

        # Get editing backend
        from diffgentor.backends.editing import get_editing_backend

        backend = get_editing_backend(config.backend_config, config.optimization_config)

        try:
            # Load model
            backend.load_model(**config.model_kwargs)

            # Setup Flux2 prompt enhancer if applicable
            if prompt_enhancer is not None and config.prompt_enhance_type == "flux2":
                from diffgentor.prompt_enhance.flux2 import Flux2PromptEnhancer

                if isinstance(prompt_enhancer, Flux2PromptEnhancer) and prompt_enhancer.mode == "diffusers":
                    if hasattr(backend, "pipe") and backend.pipe is not None:
                        pipe_class_name = backend.pipe.__class__.__name__
                        if pipe_class_name == "Flux2Pipeline":
                            prompt_enhancer.set_pipeline(backend.pipe)
                            print_rank0(f"Set Flux2 pipeline for prompt enhancer")
                        else:
                            print_rank0(
                                f"[Warning] Backend pipeline ({pipe_class_name}) "
                                f"does not support upsample_prompt"
                            )

            # Step 3: Load actual data for pending indices
            print_rank0(f"Loading data for {len(self._pending_indices)} pending samples...")
            rows = self._dataset.load_rows_by_indices(self._pending_indices)
            print_rank0(f"Loaded {len(rows)} rows with full data")

            # Build index -> row mapping for efficient lookup
            row_by_index = {int(row.get("index", -1)): row for row in rows}

            # Process pending tasks
            total = len(self._pending_indices)
            success = 0
            failed = 0
            output_rows = []

            for batch_start in range(0, total, config.batch_size):
                batch_end = min(batch_start + config.batch_size, total)
                batch_indices = self._pending_indices[batch_start:batch_end]

                self._log(
                    f"Processing batch "
                    f"{batch_start // config.batch_size + 1}/"
                    f"{(total + config.batch_size - 1) // config.batch_size}..."
                )

                for idx in batch_indices:
                    row = row_by_index.get(idx)
                    if row is None:
                        self._log(f"Skipping index {idx}: row not found")
                        failed += 1
                        continue

                    instruction = row.get(config.instruction_key, "")

                    # Load images
                    images = self._load_images(row)
                    if images is None:
                        self._log(f"Skipping row {idx}: failed to load images")
                        failed += 1
                        continue

                    # Apply prompt enhancement
                    enhanced_instruction = None
                    if prompt_enhancer is not None:
                        try:
                            enhanced_instruction = prompt_enhancer.enhance(instruction, images)
                            if enhanced_instruction != instruction:
                                self._log(
                                    f"Row {idx}: Enhanced prompt: "
                                    f"{enhanced_instruction[:100]}..."
                                )
                                instruction = enhanced_instruction
                        except Exception as e:
                            log_error(
                                e,
                                context=f"prompt_enhance[{idx}]",
                                include_traceback=True,
                            )
                            enhanced_instruction = None

                    try:
                        # Edit
                        edited = backend.edit(
                            images=images,
                            instruction=instruction,
                            num_inference_steps=config.num_inference_steps,
                            guidance_scale=config.guidance_scale,
                            true_cfg_scale=config.true_cfg_scale,
                            negative_prompt=config.negative_prompt,
                            seed=config.backend_config.seed,
                        )

                        if edited:
                            # Save - use custom filename if available
                            custom_name = None
                            if config.output_name_column:
                                custom_name = row.get(config.output_name_column)
                                if custom_name:
                                    custom_name = str(custom_name).strip()
                                    if not custom_name:
                                        custom_name = None

                            if custom_name:
                                # Use custom output path
                                from diffgentor.utils.image import save_image_to_custom_path

                                filename = save_image_to_custom_path(
                                    edited[0], config.output_dir, custom_name
                                )
                            else:
                                # Use default index-based naming
                                filename = backend.save_image(edited[0], config.output_dir, idx)

                            self._log(f"Saved: {filename}")

                            # Record result
                            row_copy = dict(row)
                            row_copy["output_image"] = filename
                            if enhanced_instruction is not None:
                                row_copy["instruction_enhanced"] = enhanced_instruction
                            # Capture CoT/recaption from backends that support it (e.g., hunyuan_image_3)
                            if hasattr(backend, "last_cot_text") and backend.last_cot_text:
                                row_copy["instruction_enhanced"] = backend.last_cot_text
                            output_rows.append(row_copy)
                            success += 1
                        else:
                            failed += 1

                    except Exception as e:
                        log_error(
                            EditingError(f"Failed to edit row {idx}", cause=e),
                            context=f"edit[{idx}]",
                            include_traceback=True,
                        )
                        failed += 1

                # Save intermediate CSV
                if output_rows:
                    from diffgentor.utils.data import save_csv_data

                    suffix = self._get_csv_suffix()
                    csv_path = Path(config.output_csv)
                    output_csv = str(csv_path.parent / f"{csv_path.stem}{suffix}{csv_path.suffix}")
                    save_csv_data(output_csv, output_rows)

            self.print_summary(success, failed)
            return 0 if failed == 0 else 1

        finally:
            backend.cleanup()


def parse_args() -> argparse.Namespace:
    """Parse worker arguments."""
    parser = EditWorker.create_parser("Edit Worker")
    return parser.parse_args()


def run_edit(args: argparse.Namespace) -> int:
    """Run image editing.

    Args:
        args: Parsed arguments

    Returns:
        Exit code
    """
    worker = EditWorker(args)
    return worker.run_with_error_handling(worker.run)


def main():
    """Main entry point."""
    args = parse_args()
    sys.exit(run_edit(args))


if __name__ == "__main__":
    main()
