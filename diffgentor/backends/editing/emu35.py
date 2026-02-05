# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Emu3.5 backend using BAAI's Emu3.5 model."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.env import Emu35Env


class Emu35Backend(BaseEditingBackend):
    """Emu3.5 backend using BAAI's Emu3.5 model.

    Supports multi-GPU tensor parallelism for large model inference via device_map="auto".

    Requires the emu35 submodule to be initialized:
        git submodule update --init diffgentor/models/third_party/emu35

    Model-specific parameters via environment variables:
        DG_EMU35_VQ_PATH: Path to VisionTokenizer model (required)
        DG_EMU35_TOKENIZER_PATH: Path to tokenizer (optional, defaults to model_path)
        DG_EMU35_CFG: Classifier-free guidance scale (default: 3.0)
        DG_EMU35_MAX_NEW_TOKENS: Maximum new tokens to generate (default: 5120)
        DG_EMU35_IMAGE_AREA: Image area for resizing (default: 1048576)
        DG_EMU35_VQ_DEVICE: Device for VQ model (default: cuda:0)
        DG_EMU35_VQ_TYPE: VQ type, "ibq" or "dcae" (default: ibq)
        DG_EMU35_GPUS_PER_MODEL: Number of GPUs per model instance (default: 0, use all visible)

    Multi-GPU Usage:
        The Launcher automatically handles GPU assignment based on DG_EMU35_GPUS_PER_MODEL.
        Model is distributed across visible GPUs via device_map="auto".

        Examples:
            # Single model on 2 GPUs
            CUDA_VISIBLE_DEVICES=0,1 diffgentor edit --backend emu35 ...

            # 4 model instances, each on 2 GPUs (8 GPUs total)
            # Launcher spawns 4 processes, each with CUDA_VISIBLE_DEVICES set to 2 GPUs
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DG_EMU35_GPUS_PER_MODEL=2 \
                diffgentor edit --backend emu35 --model_type emu35 ...
            # Instance 0: GPU 0,1 | Instance 1: GPU 2,3 | Instance 2: GPU 4,5 | Instance 3: GPU 6,7
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize Emu3.5 backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._model = None
        self._tokenizer = None
        self._vq_model = None
        # Load env config
        self._env = Emu35Env.load()
        self._vq_path = self._env.vq_path
        self._tokenizer_path = self._env.tokenizer_path
        self._cfg = self._env.cfg
        self._max_new_tokens = self._env.max_new_tokens
        self._image_area = self._env.image_area
        self._vq_device = self._env.vq_device

    def load_model(self, **kwargs) -> None:
        """Load Emu3.5 model.

        Model-specific parameters are read from environment variables.
        Model automatically distributed across visible GPUs via device_map="auto".
        GPU assignment is handled by the Launcher based on DG_EMU35_GPUS_PER_MODEL.

        Args:
            **kwargs: Additional arguments
        """
        # Reload env config to get latest values
        self._env = Emu35Env.load()
        self._vq_path = self._env.vq_path
        self._tokenizer_path = self._env.tokenizer_path or self.model_name
        self._cfg = self._env.cfg
        self._max_new_tokens = self._env.max_new_tokens
        self._image_area = self._env.image_area
        self._vq_device = self._env.vq_device
        vq_type = os.environ.get("DG_EMU35_VQ_TYPE", "ibq")

        if not self._vq_path:
            raise ValueError(
                "DG_EMU35_VQ_PATH environment variable is required. "
                "Set it to the path of your VisionTokenizer model."
            )

        # Log GPU configuration
        # Note: CUDA_VISIBLE_DEVICES is already set by Launcher for multi-process mode
        visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        num_gpus = torch.cuda.device_count()
        print_rank0(f"[Emu3.5] CUDA_VISIBLE_DEVICES: {visible_gpus}")
        print_rank0(f"[Emu3.5] Available CUDA devices: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print_rank0(f"  GPU {i}: {props.name}, {props.total_memory / 1024**3:.1f} GB")

        # Use device_map="auto" to distribute model across all visible GPUs
        # VQ model goes to cuda:0 (first visible GPU)
        hf_device = "auto"
        vq_device = "cuda:0"

        # Add third-party path
        third_party_path = Path(__file__).parent.parent.parent / "models" / "third_party" / "emu35"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
            # Also add src to path
            src_path = third_party_path / "src"
            if src_path.exists():
                sys.path.insert(0, str(src_path))
        else:
            raise ImportError(
                f"Emu3.5 submodule not found at {third_party_path}. "
                "Run: git submodule update --init diffgentor/models/third_party/emu35"
            )

        try:
            from src.utils.model_utils import build_emu3p5

            print_rank0(f"Loading Emu3.5 from: {self.model_name}")
            print_rank0(f"VQ path: {self._vq_path}")
            print_rank0(f"Tokenizer path: {self._tokenizer_path}")
            print_rank0(f"HF device (device_map): {hf_device}, VQ device: {vq_device}")

            # Build model with device_map for multi-GPU support
            self._model, self._tokenizer, self._vq_model = build_emu3p5(
                model_path=self.model_name,
                tokenizer_path=self._tokenizer_path,
                vq_path=self._vq_path,
                vq_type=vq_type,
                model_device=hf_device,
                vq_device=vq_device,
            )

            self._initialized = True
            print_rank0("Emu3.5 model loaded successfully")

        except ImportError as e:
            raise ImportError(
                f"Failed to import Emu3.5 modules: {e}. "
                "Make sure the emu35 submodule is properly initialized and "
                "all dependencies are installed."
            )

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,  # Not used for autoregressive models
        guidance_scale: Optional[float] = None,  # Not directly used
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using Emu3.5.

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Not used for autoregressive models
            guidance_scale: Not directly used (use DG_EMU35_CFG env var)
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Normalize images
        if isinstance(images, Image.Image):
            images = [images]

        input_image = images[0] if images else None
        if input_image is None:
            raise ValueError("No input image provided")

        # Convert to RGB
        input_image = input_image.convert("RGB")

        try:
            from src.utils.generation_utils import generate, multimodal_decode
            from src.utils.input_utils import build_image, smart_resize

            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                import random
                random.seed(seed)

            # Resize image
            input_image = smart_resize(input_image, max_pixels=self._image_area)

            # Build image tokens
            image_str = build_image(
                input_image,
                cfg=self._get_cfg_object(),
                tokenizer=self._tokenizer,
                vq_model=self._vq_model,
            )

            # Build prompt
            template = "Edit this image according to the instruction: {question}\n<|IMAGE|>"
            prompt = template.format(question=instruction)
            prompt = prompt.replace("<|IMAGE|>", image_str)

            # Encode prompt
            input_ids = self._tokenizer.encode(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._model.device)

            # Add BOS token if needed
            bos_id = self._tokenizer.encode(self._tokenizer.bos_token)[0]
            if input_ids[0, 0] != bos_id:
                bos_tensor = torch.tensor([[bos_id]], device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([bos_tensor, input_ids], dim=1)

            # Unconditional prompt
            unc_prompt = "Generate an image."
            unconditional_ids = self._tokenizer.encode(
                unc_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self._model.device)

            # Generate
            cfg_obj = self._get_cfg_object()
            result_images = []

            for result_tokens in generate(
                cfg_obj,
                self._model,
                self._tokenizer,
                input_ids,
                unconditional_ids,
                None,
                force_same_image_size=True,
            ):
                try:
                    result = self._tokenizer.decode(result_tokens, skip_special_tokens=False)
                    mm_out = multimodal_decode(result, self._tokenizer, self._vq_model)

                    for item in mm_out:
                        if isinstance(item, list) and len(item) == 2:
                            if isinstance(item[1], Image.Image):
                                result_images.append(item[1])
                except Exception as e:
                    print_rank0(f"Failed to decode result: {e}")

            return result_images if result_images else [input_image]

        except Exception as e:
            print_rank0(f"Emu3.5 edit failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _get_cfg_object(self):
        """Create a config object for Emu3.5 generation.

        Returns:
            Config object with generation parameters
        """
        class Cfg:
            pass

        cfg = Cfg()
        cfg.cfg_scale = self._cfg
        cfg.max_new_tokens = self._max_new_tokens
        cfg.temperature = 1.0
        cfg.top_p = 1.0
        cfg.do_sample = True
        cfg.seed = 6666

        # Special tokens
        cfg.special_tokens = {
            "BOS": self._tokenizer.bos_token,
            "EOS": self._tokenizer.eos_token,
            "BOI": self._tokenizer.boi_token,
            "EOI": self._tokenizer.eoi_token,
        }
        cfg.special_token_ids = {}
        for k, v in cfg.special_tokens.items():
            cfg.special_token_ids[k] = self._tokenizer.encode(v)[0]

        return cfg

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._model is not None:
            del self._model
            self._model = None
        if self._vq_model is not None:
            del self._vq_model
            self._vq_model = None
        self._tokenizer = None
        torch.cuda.empty_cache()
