# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen configuration loader.

This module provides utilities for loading DeepGen model configurations
from Python config files in the config directory.

Usage:
    from diffgentor.models.deepgen.config import load_deepgen_config

    config = load_deepgen_config("deepgen")
    model_cfg = config["model"]
    num_queries = model_cfg.get("num_queries", 128)

Config files should define the following variables:
    - model: Model configuration dict
    - tokenizer: Tokenizer configuration dict
    - prompt_template: Prompt template dict
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List


def load_deepgen_config(config_name: str) -> Dict[str, Any]:
    """Load DeepGen config from config directory.

    Args:
        config_name: Config file name (without .py extension), e.g., "deepgen"

    Returns:
        Dict containing config attributes:
            - model: Model configuration dict
            - tokenizer: Tokenizer configuration dict
            - prompt_template: Prompt template dict

    Raises:
        FileNotFoundError: If config file does not exist

    Example:
        >>> config = load_deepgen_config("deepgen")
        >>> config["model"]["num_queries"]
        128
    """
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{config_name}.py"

    if not config_path.exists():
        available = get_available_configs()
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n" f"Available configs: {available}"
        )

    spec = importlib.util.spec_from_file_location(config_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "model": getattr(module, "model", {}),
        "tokenizer": getattr(module, "tokenizer", {}),
        "prompt_template": getattr(module, "prompt_template", {}),
    }


def get_available_configs() -> List[str]:
    """Get list of available config names.

    Returns:
        List of config file names (without .py extension)
    """
    config_dir = Path(__file__).parent
    return [f.stem for f in config_dir.glob("*.py") if f.stem != "__init__"]
