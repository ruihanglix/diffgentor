# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Base backend classes for diffgentor.

This module defines the abstract base classes for all backends:
- BasePipelineBackend: Common functionality for all pipeline-based backends
- BaseBackend: T2I generation backends
- BaseEditingBackend: Image editing backends
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from diffgentor.config import BackendConfig, OptimizationConfig


class BasePipelineBackend(ABC):
    """Abstract base class with common functionality for all pipeline backends.

    This class provides shared implementation for:
    - Configuration management
    - Model lifecycle (load, cleanup)
    - Optimization application
    - Context manager support
    - LoRA adapter management (optional, for diffusers-based backends)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        self.backend_config = backend_config
        self.optimization_config = optimization_config or OptimizationConfig()
        self.pipe = None
        self._initialized = False
        self.distributed_state = None

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.backend_config.model_name

    @property
    def model_type(self) -> Optional[str]:
        """Get model type."""
        return self.backend_config.model_type

    @property
    def device(self) -> str:
        """Get device string."""
        return self.backend_config.device

    @property
    def is_initialized(self) -> bool:
        """Check if backend is initialized."""
        return self._initialized

    @property
    def supports_lora(self) -> bool:
        """Whether this backend supports dynamic LoRA adapter management."""
        return False

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load and initialize the model.

        This method should:
        1. Load the model from model_name
        2. Move to appropriate device
        3. Apply optimizations if configured

        Args:
            **kwargs: Additional model loading arguments
        """
        pass

    def apply_optimizations(self) -> None:
        """Apply optimization configuration to the loaded model.

        Override in subclasses to implement backend-specific optimizations.
        """
        pass

    def cleanup(self) -> None:
        """Clean up resources.

        Override in subclasses if specific cleanup is needed.
        """
        self.pipe = None
        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False

    # ------------------------------------------------------------------
    # LoRA adapter management (optional)
    # ------------------------------------------------------------------

    def load_lora(self, lora_path: str, adapter_name: str, strength: float = 1.0) -> None:
        """Load a single LoRA adapter and activate it.

        Args:
            lora_path: Path to LoRA weights (safetensors file, directory, or HF repo ID)
            adapter_name: Unique nickname for this adapter
            strength: Adapter strength (scale), default 1.0
        """
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")

    def load_loras(
        self,
        lora_paths: List[str],
        adapter_names: List[str],
        strengths: List[float],
        targets: Optional[List[str]] = None,
    ) -> None:
        """Load multiple LoRA adapters and activate them all simultaneously.

        Args:
            lora_paths: Paths to LoRA weights
            adapter_names: Unique nicknames for each adapter
            strengths: Adapter strengths (scale factors)
            targets: Which component(s) to apply each adapter to.
                     Valid values depend on the backend; common values include
                     ``"all"`` (default), ``"transformer"``, ``"transformer_2"``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")

    def set_active_loras(self, adapter_names: List[str], strengths: List[float]) -> None:
        """Set which loaded adapters are active (on-the-fly, without fusing).

        Args:
            adapter_names: List of adapter nicknames to activate
            strengths: Corresponding strengths for each adapter
        """
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")

    def fuse_lora(self, adapter_names: Optional[List[str]] = None, lora_scale: float = 1.0) -> None:
        """Fuse active LoRA weights into the base model for faster inference.

        Args:
            adapter_names: Adapters to fuse (None = currently active ones)
            lora_scale: Scale factor for fusion
        """
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")

    def unfuse_lora(self) -> None:
        """Unfuse LoRA weights, restoring the base model."""
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")

    def unload_lora(self, adapter_name: Optional[str] = None) -> None:
        """Completely unload a LoRA adapter from memory.

        Args:
            adapter_name: Adapter to unload (None = unload all)
        """
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")

    def list_loras(self) -> Dict[str, Any]:
        """List loaded LoRA adapters and their status.

        Returns:
            Dict with ``loaded_adapters`` and ``active`` keys.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA management")


class BaseBackend(BasePipelineBackend):
    """Abstract base class for T2I generation backends.

    All T2I backends (diffusers, xdit, openai) must implement this interface.
    """

    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from prompts.

        Args:
            prompt: Single prompt or list of prompts
            **kwargs: Additional generation arguments

        Returns:
            List of generated PIL Images
        """
        pass


class BaseEditingBackend(BasePipelineBackend):
    """Abstract base class for image editing backends.

    Editing backends handle image-to-image editing tasks.
    """

    @abstractmethod
    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images based on instruction.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            **kwargs: Additional editing arguments

        Returns:
            List of edited PIL Images
        """
        pass

    def edit_batch(
        self,
        batch_data: List[Tuple[List[Image.Image], str, int]],
        **kwargs,
    ) -> List[Tuple[int, Optional[Image.Image]]]:
        """Edit a batch of images.

        Args:
            batch_data: List of (images, instruction, index) tuples
            **kwargs: Additional editing arguments

        Returns:
            List of (index, edited_image) tuples
        """
        from diffgentor.utils.exceptions import EditingError, log_error

        results = []
        for images, instruction, idx in batch_data:
            try:
                edited = self.edit(images, instruction, **kwargs)
                if edited:
                    results.append((idx, edited[0]))
                else:
                    results.append((idx, None))
            except Exception as e:
                log_error(
                    EditingError(f"Failed to edit index {idx}", cause=e),
                    context=f"edit_batch[{idx}]",
                    include_traceback=True,
                )
                results.append((idx, None))
        return results

    def save_image(
        self,
        image: Image.Image,
        output_dir: str,
        index: int,
    ) -> str:
        """Save edited image to file.

        Args:
            image: Image to save
            output_dir: Output directory
            index: Image index

        Returns:
            Saved filename
        """
        from pathlib import Path

        from diffgentor.utils.image import get_output_filename

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = get_output_filename(index)
        output_path = output_dir / filename
        image.save(output_path)

        return filename
