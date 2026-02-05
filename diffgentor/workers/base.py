# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Base worker class with common functionality for T2I and Editing workers."""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from diffgentor.utils.distributed import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
)
from diffgentor.utils.exceptions import log_error
from diffgentor.utils.logging import LoggingConfig, print_rank0, setup_logging


class BaseWorker(ABC):
    """Abstract base class for workers.

    Provides common functionality:
    - Argument parsing setup
    - Distributed mode detection
    - Task distribution (two-level: node + process)
    - Resume support
    - Progress reporting
    - Distributed logging
    """

    def __init__(self, args: argparse.Namespace):
        """Initialize worker.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self._setup_distributed()
        self._setup_logging()
        self._logger: Optional[logging.Logger] = None

    def _setup_distributed(self) -> None:
        """Setup distributed environment."""
        use_distributed = is_distributed() or get_world_size() > 1
        use_manual = (
            getattr(self.args, "num_gpus", None) is not None
            and self.args.num_gpus > 1
            and getattr(self.args, "local_rank", None) is not None
        )

        if use_distributed:
            self.num_processes = get_world_size()
            self.process_rank = get_rank()
            self.local_rank = get_local_rank()
            self.distributed_mode = "torchrun"
        elif use_manual:
            self.num_processes = self.args.num_gpus
            self.process_rank = self.args.local_rank
            self.local_rank = self.args.local_rank
            self.distributed_mode = "manual"
        else:
            self.num_processes = 1
            self.process_rank = 0
            self.local_rank = 0
            self.distributed_mode = None

    def _setup_logging(self) -> None:
        """Setup logging system for distributed environment."""
        # Get output_dir from args (may be set by subclass)
        output_dir = getattr(self.args, "output_dir", None)
        log_dir = getattr(self.args, "log_dir", None)
        node_rank = getattr(self.args, "node_rank", 0)
        num_nodes = getattr(self.args, "num_nodes", 1)

        config = LoggingConfig(
            log_dir=log_dir,
            output_dir=output_dir,
            rank=self.process_rank,
            local_rank=self.local_rank,
            node_rank=node_rank,
            num_nodes=num_nodes,
            world_size=self.num_processes,
        )

        self.log_dir = setup_logging(config)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this worker."""
        if self._logger is None:
            self._logger = logging.getLogger(f"diffgentor.worker.{self.__class__.__name__}")
        return self._logger

    def distribute_tasks(
        self,
        all_tasks: list[Any],
        node_rank: int,
        num_nodes: int,
    ) -> tuple[list[int], list[Any]]:
        """Distribute tasks using two-level distribution.

        Level 1: Distribute across nodes
        Level 2: Distribute across processes within node

        Args:
            all_tasks: All tasks to distribute
            node_rank: Current node rank
            num_nodes: Total number of nodes

        Returns:
            Tuple of (task_indices, tasks_for_this_process)
        """
        from diffgentor.utils.task_distribution import distribute_tasks
        total = len(all_tasks)

        # Level 1: Node-level distribution
        node_start, node_end = distribute_tasks(total, node_rank, num_nodes)
        node_indices = list(range(node_start, node_end))
        print_rank0(
            f"Node {node_rank}: Assigned tasks {node_start}-{node_end} "
            f"({len(node_indices)} tasks)"
        )

        # Level 2: Process-level distribution within node
        if self.num_processes > 1:
            proc_start, proc_end = distribute_tasks(
                len(node_indices),
                self.process_rank,
                self.num_processes,
            )
            process_indices = node_indices[proc_start:proc_end]
            self._log(
                f"Assigned {len(process_indices)} tasks "
                f"(local indices {proc_start}-{proc_end})"
            )
        else:
            process_indices = node_indices

        if not process_indices:
            self._log("No tasks to process")
            return [], []

        tasks = [all_tasks[i] for i in process_indices]
        return process_indices, tasks

    def filter_completed(
        self,
        tasks: list[Any],
        get_output_path: Callable[[Any], Path],
    ) -> tuple[list[Any], int]:
        """Filter out completed tasks for resume.

        Args:
            tasks: All tasks
            get_output_path: Function to get output path for a task

        Returns:
            Tuple of (remaining_tasks, skipped_count)
        """
        filtered = []
        for task in tasks:
            output_path = get_output_path(task)
            if not output_path.exists():
                filtered.append(task)

        skipped = len(tasks) - len(filtered)
        return filtered, skipped

    def _log(self, message: str) -> None:
        """Log message with process prefix.

        This method logs to both console (if local_rank=0) and file (always).

        Args:
            message: Message to log
        """
        self.logger.info(f"[Process {self.process_rank}] {message}")

    def print_header(self, title: str, **info: Any) -> None:
        """Print formatted header with info.

        Args:
            title: Header title
            **info: Key-value pairs to display
        """
        print_rank0("=" * 80)
        print_rank0(f"diffgentor {title}")
        print_rank0("=" * 80)
        for key, value in info.items():
            if value is not None:
                print_rank0(f"{key}: {value}")
        if self.distributed_mode:
            print_rank0(
                f"Distributed Mode: {self.distributed_mode} "
                f"(world_size={self.num_processes})"
            )
        print_rank0("=" * 80)

    def print_summary(self, success: int, failed: int) -> None:
        """Print completion summary."""
        self._log("=" * 60)
        self._log(f"Completed: {success} successful, {failed} failed")
        self._log("=" * 60)

    def run_with_error_handling(self, func: Callable[[], int]) -> int:
        """Run function with standardized error handling.

        Args:
            func: Function to run

        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            return func()
        except KeyboardInterrupt:
            print_rank0("\nInterrupted by user")
            return 130
        except Exception as e:
            log_error(e, context="Worker execution", include_traceback=True)
            return 1

    @abstractmethod
    def build_config(self) -> Any:
        """Build configuration from arguments.

        Returns:
            Configuration object
        """
        pass

    @abstractmethod
    def run(self) -> int:
        """Run the worker.

        Returns:
            Exit code
        """
        pass

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add worker-specific arguments to parser.

        Args:
            parser: Argument parser
        """
        pass

    @classmethod
    def add_common_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add common arguments shared by all workers.

        Args:
            parser: Argument parser
        """
        # Backend
        parser.add_argument("--backend", type=str, default="diffusers")
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, default=None)
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--seed", type=int, default=None)

        # Optimization
        parser.add_argument("--torch_dtype", type=str, default="bfloat16")
        parser.add_argument("--optimize", type=str, default=None)
        parser.add_argument("--attention_backend", type=str, default=None)
        parser.add_argument("--cache_type", type=str, default=None)
        parser.add_argument("--enable_compile", action="store_true")
        parser.add_argument("--enable_cpu_offload", action="store_true")
        parser.add_argument("--enable_vae_slicing", action="store_true")
        parser.add_argument("--enable_vae_tiling", action="store_true")

        # xDiT - Note: xDiT parameters are configured via DG_XDIT_* env vars

        # Execution
        parser.add_argument("--max_retries", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=1)

        # Distributed
        parser.add_argument("--node_rank", type=int, default=0)
        parser.add_argument("--num_nodes", type=int, default=1)
        parser.add_argument("--local_rank", type=int, default=None)
        parser.add_argument("--num_gpus", type=int, default=None)

        # Logging
        parser.add_argument("--log_dir", type=str, default=None)

        # API backend options
        parser.add_argument("--timeout", type=float, default=300.0)
        parser.add_argument("--api_max_retries", type=int, default=0)
        parser.add_argument("--retry_delay", type=float, default=1.0)
        parser.add_argument("--max_global_workers", type=int, default=16)
        parser.add_argument("--num_processes", type=int, default=4)

    @classmethod
    def create_parser(cls, description: str) -> argparse.ArgumentParser:
        """Create argument parser with common arguments.

        Args:
            description: Parser description

        Returns:
            Configured argument parser
        """
        parser = argparse.ArgumentParser(description=description)
        cls.add_common_arguments(parser)
        cls.add_arguments(parser)
        return parser
