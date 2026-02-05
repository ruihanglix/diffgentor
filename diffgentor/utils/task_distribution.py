# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Task distribution utilities for multi-node execution."""

from typing import Tuple


def distribute_tasks(
    total_tasks: int,
    node_rank: int,
    num_nodes: int,
) -> Tuple[int, int]:
    """Distribute tasks evenly across nodes.

    Args:
        total_tasks: Total number of tasks
        node_rank: Current node rank (0-indexed)
        num_nodes: Total number of nodes

    Returns:
        Tuple of (start_index, end_index) for this node
    """
    if num_nodes <= 1:
        return 0, total_tasks

    tasks_per_node = total_tasks // num_nodes
    remainder = total_tasks % num_nodes

    # Distribute remainder to first nodes
    if node_rank < remainder:
        start = node_rank * (tasks_per_node + 1)
        end = start + tasks_per_node + 1
    else:
        start = remainder * (tasks_per_node + 1) + (node_rank - remainder) * tasks_per_node
        end = start + tasks_per_node

    return start, end
