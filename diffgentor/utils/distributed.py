# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Distributed utilities for diffgentor."""

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get the world size (total number of processes).

    Returns:
        World size, or 1 if not distributed
    """
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank() -> int:
    """Get the global rank of current process.

    Returns:
        Global rank, or 0 if not distributed
    """
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    """Get the local rank of current process.

    Returns:
        Local rank, or 0 if not distributed
    """
    if is_distributed():
        # Try to get from environment first
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return int(local_rank)
        # Fallback to global rank
        return dist.get_rank()
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0).

    Returns:
        True if main process
    """
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def get_device() -> torch.device:
    """Get the device for current process.

    Returns:
        torch.device for current process
    """
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def init_distributed(backend: str = "nccl") -> bool:
    """Initialize distributed training if not already initialized.

    Args:
        backend: Distributed backend to use

    Returns:
        True if distributed was initialized
    """
    if is_distributed():
        return True

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        return True

    return False


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if is_distributed():
        dist.destroy_process_group()
