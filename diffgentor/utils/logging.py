# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Logging utilities for diffgentor."""

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from diffgentor.utils.distributed import get_rank, is_main_process


@dataclass
class LoggingConfig:
    """Logging configuration for distributed environment."""

    log_dir: Optional[str] = None
    output_dir: Optional[str] = None
    rank: int = 0
    local_rank: int = 0
    node_rank: int = 0
    num_nodes: int = 1
    world_size: int = 1
    level: int = logging.INFO


class StreamRedirect:
    """Redirect stdout/stderr to logging system.

    This class intercepts write calls and routes them to the logger.
    The logger's handlers control where the output goes (console for rank 0,
    file for all processes).
    """

    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        original_stream,
    ):
        """Initialize StreamRedirect.

        Args:
            logger: Logger instance to write to
            level: Logging level for messages
            original_stream: Original stdout/stderr stream (kept for fileno/isatty)
        """
        self.logger = logger
        self.level = level
        self.original = original_stream
        self._buffer = ""

    def write(self, message: str) -> None:
        """Write message to logger and optionally to console.

        Args:
            message: Message to write
        """
        # Buffer incomplete lines
        self._buffer += message

        # Process complete lines
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        # Note: We don't write to original console here because the logger's
        # StreamHandler already outputs to console for is_console_process=True.
        # Writing to both would cause duplicate/interleaved output.

    def flush(self) -> None:
        """Flush the stream."""
        # Flush any remaining buffer content
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.rstrip())
            self._buffer = ""
        self.original.flush()

    def fileno(self) -> int:
        """Return file descriptor of original stream."""
        return self.original.fileno()

    def isatty(self) -> bool:
        """Check if original stream is a tty."""
        return self.original.isatty()


# Global state for logging setup
_logging_initialized = False
_original_stdout = None
_original_stderr = None
_log_dir: Optional[str] = None


def _suppress_third_party_logging(local_rank: int) -> None:
    """Suppress console output from third-party libraries for non-main processes.

    Args:
        local_rank: Local rank of current process
    """
    if local_rank != 0:
        # Suppress tqdm progress bars
        os.environ["TQDM_DISABLE"] = "1"

        # Suppress diffusers logging
        try:
            from diffusers.utils import logging as diffusers_logging

            diffusers_logging.disable_default_handler()
            diffusers_logging.disable_progress_bar()
        except ImportError:
            pass

        # Suppress transformers logging
        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.disable_default_handler()
            transformers_logging.disable_progress_bar()
        except ImportError:
            pass

        # Suppress other common libraries by setting high verbosity level
        for lib_name in ["torch", "torchvision", "PIL", "urllib3", "requests"]:
            lib_logger = logging.getLogger(lib_name)
            lib_logger.setLevel(logging.ERROR)


def _get_log_filename(node_rank: int, local_rank: int, num_nodes: int, world_size: int) -> str:
    """Generate log filename based on process info.

    Args:
        node_rank: Node rank
        local_rank: Local rank within node
        num_nodes: Total number of nodes
        world_size: Number of processes per node

    Returns:
        Log filename
    """
    # Single process on single node
    if num_nodes <= 1 and world_size <= 1:
        return "process_0.log"
    # Multi-node or multi-process: include both node and process info
    return f"node{node_rank}_process{local_rank}.log"


def setup_logging(config: LoggingConfig) -> str:
    """Setup logging system for distributed environment.

    This function configures the logging system with the following behavior:
    - Only local_rank=0 processes output to terminal
    - All processes write to individual log files
    - stdout/stderr are redirected to capture print() calls
    - Third-party library logging is controlled

    Args:
        config: Logging configuration

    Returns:
        Path to the log directory
    """
    global _logging_initialized, _original_stdout, _original_stderr, _log_dir

    if _logging_initialized:
        return _log_dir or ""

    # Determine log directory
    if config.log_dir:
        log_dir = config.log_dir
    elif config.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_dir = str(Path(config.output_dir) / "logs" / timestamp)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        log_dir = str(Path("./logs") / timestamp)

    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    _log_dir = log_dir

    # Determine if this process should output to console
    is_console_process = config.local_rank == 0

    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(config.level)

    # Remove existing handlers from root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - [Rank %(rank)s] - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a filter to add rank info
    class RankFilter(logging.Filter):
        def __init__(self, rank: int):
            super().__init__()
            self.rank = rank

        def filter(self, record: logging.LogRecord) -> bool:
            record.rank = self.rank
            return True

    rank_filter = RankFilter(config.rank)

    # Add StreamHandler only for local_rank=0
    if is_console_process:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(config.level)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(rank_filter)
        root_logger.addHandler(console_handler)

    # Add FileHandler for all processes
    log_filename = _get_log_filename(config.node_rank, config.local_rank, config.num_nodes, config.world_size)
    log_filepath = Path(log_dir) / log_filename
    file_handler = logging.FileHandler(str(log_filepath), mode="a", encoding="utf-8")
    file_handler.setLevel(config.level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(rank_filter)
    root_logger.addHandler(file_handler)

    # Configure diffgentor logger - remove any pre-existing handlers to avoid duplicate output
    diffgentor_logger = logging.getLogger("diffgentor")
    for handler in diffgentor_logger.handlers[:]:
        diffgentor_logger.removeHandler(handler)
    diffgentor_logger.setLevel(config.level)

    # Save original streams
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr

    # Create redirect logger
    redirect_logger = logging.getLogger("diffgentor.stdout")
    redirect_logger.setLevel(config.level)

    # Redirect stdout/stderr
    sys.stdout = StreamRedirect(redirect_logger, logging.INFO, _original_stdout)
    sys.stderr = StreamRedirect(redirect_logger, logging.WARNING, _original_stderr)

    # Suppress third-party logging for non-main processes
    _suppress_third_party_logging(config.local_rank)

    _logging_initialized = True

    # Log initialization message
    root_logger.info(
        f"Logging initialized: log_dir={log_dir}, rank={config.rank}, "
        f"local_rank={config.local_rank}, node_rank={config.node_rank}, "
        f"world_size={config.world_size}, console={is_console_process}"
    )

    return log_dir


def restore_streams() -> None:
    """Restore original stdout/stderr streams."""
    global _original_stdout, _original_stderr

    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr


def get_log_dir() -> Optional[str]:
    """Get the current log directory.

    Returns:
        Log directory path or None if not initialized
    """
    return _log_dir


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with the specified name and level.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only add handler if logging not initialized globally and no handlers exist
    if not _logging_initialized and not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def print_rank0(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Print message only on rank 0 (main process).

    Args:
        message: Message to print
        logger: Optional logger to use instead of print
    """
    if is_main_process():
        if logger:
            logger.info(message)
        else:
            print(message)


# Create default logger
default_logger = get_logger("diffgentor")
