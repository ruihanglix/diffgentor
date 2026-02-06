# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Main CLI entry point for diffgentor."""

import argparse
import sys
from typing import List, Optional


class RequiredFirstHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Custom formatter that shows required arguments before optional ones within each group."""

    def _format_usage(self, usage, actions, groups, prefix):
        # Sort actions to show required first in usage
        required = [a for a in actions if a.required and a.option_strings]
        optional = [a for a in actions if not a.required or not a.option_strings]
        sorted_actions = required + optional
        return super()._format_usage(usage, sorted_actions, groups, prefix)

    def add_arguments(self, actions):
        # Sort actions: required first, then optional
        required = [a for a in actions if a.required]
        optional = [a for a in actions if not a.required]
        sorted_actions = required + optional
        super().add_arguments(sorted_actions)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="diffgentor",
        description="A unified visual generation data synthesis factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-to-Image generation with diffusers
  diffgentor t2i --backend diffusers --model_name black-forest-labs/FLUX.1-dev --prompt "A cat"

  # T2I with xDiT multi-GPU
  diffgentor t2i --backend xdit --model_name FLUX.1-dev --num_gpus 4

  # Image editing with diffusers
  diffgentor edit --backend diffusers --model_name Qwen-Image-Edit --model_type qwen --input data.csv

  # Editing with OpenAI API
  diffgentor edit --backend openai --model_name gpt-image-1 --input data.csv
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # T2I subcommand
    t2i_parser = subparsers.add_parser(
        "t2i",
        help="Text-to-Image generation",
        formatter_class=RequiredFirstHelpFormatter,
    )
    _add_t2i_arguments(t2i_parser)

    # Edit subcommand
    edit_parser = subparsers.add_parser(
        "edit",
        help="Image editing",
        formatter_class=RequiredFirstHelpFormatter,
    )
    _add_edit_arguments(edit_parser)

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by all commands.

    Args:
        parser: Argument parser to add arguments to
    """
    # Backend arguments
    backend_group = parser.add_argument_group("Backend Options")
    backend_group.add_argument(
        "--backend",
        type=str,
        default="diffusers",
        choices=[
            "diffusers", "xdit", "openai", "google_genai", "gemini",
            "step1x", "bagel", "emu35", "dreamomni2", "flux_kontext_official", "hunyuan_image_3", "deepgen",
        ],
        help="Backend to use for inference",
    )
    backend_group.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (HuggingFace ID or local path)",
    )
    backend_group.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type for explicit pipeline selection (auto-detected if not specified)",
    )
    backend_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu, cuda:N)",
    )
    backend_group.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (auto-detect all available GPUs if not specified)",
    )
    backend_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Optimization arguments
    opt_group = parser.add_argument_group("Optimization Options")
    opt_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model weights",
    )
    opt_group.add_argument(
        "--optimize",
        type=str,
        default=None,
        help="Comma-separated optimization flags (e.g., 'torch_compile,vae_slicing,flash_attention')",
    )
    opt_group.add_argument(
        "--attention_backend",
        type=str,
        default=None,
        choices=["flash", "sage", "xformers"],
        help="Attention backend to use",
    )
    opt_group.add_argument(
        "--cache_type",
        type=str,
        default=None,
        choices=["deep_cache", "first_block_cache", "pab", "faster_cache", "cache_dit"],
        help="Cache acceleration type",
    )
    opt_group.add_argument(
        "--enable_compile",
        action="store_true",
        help="Enable torch.compile",
    )
    opt_group.add_argument(
        "--enable_cpu_offload",
        action="store_true",
        help="Enable model CPU offload",
    )
    opt_group.add_argument(
        "--enable_vae_slicing",
        action="store_true",
        help="Enable VAE slicing",
    )
    opt_group.add_argument(
        "--enable_vae_tiling",
        action="store_true",
        help="Enable VAE tiling",
    )

    # API backend options (for openai, google_genai)
    api_group = parser.add_argument_group("API Backend Options (for --backend openai/google_genai)")
    api_group.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds",
    )
    api_group.add_argument(
        "--api_max_retries",
        type=int,
        default=0,
        help="Max retry attempts for API requests (0 = no retry)",
    )
    api_group.add_argument(
        "--retry_delay",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds",
    )
    api_group.add_argument(
        "--max_global_workers",
        type=int,
        default=16,
        help="Total concurrent workers across all processes",
    )
    api_group.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="Number of worker processes (threads per process = max_global_workers / num_processes)",
    )

    # xDiT specific - Note: xDiT parameters are configured via DG_XDIT_* env vars
    # See docs/optimization.md for details

    # Multi-node arguments
    dist_group = parser.add_argument_group("Distributed Options")
    dist_group.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="Rank of current node (0-indexed)",
    )
    dist_group.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Total number of nodes",
    )

    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for log files (default: output_dir/logs/yyyymmdd_hhmm)",
    )


def _add_t2i_arguments(parser: argparse.ArgumentParser) -> None:
    """Add T2I specific arguments.

    Args:
        parser: Argument parser to add arguments to
    """
    _add_common_arguments(parser)

    # Input/Output
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation",
    )
    io_group.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to prompts file (JSONL, JSON, TXT, CSV)",
    )
    io_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images",
    )

    # Generation parameters
    gen_group = parser.add_argument_group("Generation Options")
    gen_group.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt",
    )
    gen_group.add_argument(
        "--height",
        type=int,
        default=None,
        help="Image height (model default if not specified)",
    )
    gen_group.add_argument(
        "--width",
        type=int,
        default=None,
        help="Image width (model default if not specified)",
    )
    gen_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of denoising steps (model default if not specified)",
    )
    gen_group.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale for classifier-free guidance (model default if not specified)",
    )
    gen_group.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for generation",
    )
    gen_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation",
    )

    # Execution
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for failed generations",
    )
    exec_group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous progress",
    )
    exec_group.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from previous progress",
    )


def _add_edit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add editing specific arguments.

    Args:
        parser: Argument parser to add arguments to
    """
    _add_common_arguments(parser)

    # Input/Output
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "--input",
        type=str,
        required=True,
        dest="input_data",
        help="Path to input data (CSV or Parquet directory)",
    )
    io_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save edited images",
    )
    io_group.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: output_dir/results.csv)",
    )
    io_group.add_argument(
        "--instruction_key",
        type=str,
        default="instruction",
        help="Column name for instruction in input data",
    )
    io_group.add_argument(
        "--image_cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded images",
    )

    # Editing parameters
    edit_group = parser.add_argument_group("Editing Options")
    edit_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing",
    )
    edit_group.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Number of denoising steps (model default if not specified)",
    )
    edit_group.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Guidance scale (model default if not specified)",
    )
    edit_group.add_argument(
        "--true_cfg_scale",
        type=float,
        default=None,
        help="True CFG scale for Qwen models (model default if not specified)",
    )
    edit_group.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="Negative prompt",
    )

    # Execution
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retries for failed edits",
    )
    exec_group.add_argument(
        "--filter_rows",
        type=str,
        default=None,
        help="Filter rows by index (e.g., '0:100', '0,5,10', '100:')",
    )
    exec_group.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from previous progress",
    )
    exec_group.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from previous progress",
    )

    # Prompt enhancement
    enhance_group = parser.add_argument_group("Prompt Enhancement Options")
    enhance_group.add_argument(
        "--prompt_enhance_type",
        type=str,
        default=None,
        choices=["qwen_image_edit", "glm_image", "flux2"],
        help="Type of prompt enhancement to use (API config via DG_PROMPT_ENHANCER_* env vars)",
    )


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point.

    Args:
        args: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 1

    # Import launcher
    from diffgentor.launcher import Launcher

    # Create launcher and execute
    launcher = Launcher(parsed_args)

    try:
        return launcher.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
