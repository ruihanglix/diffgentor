# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Flux Kontext Official backend using BFL's official implementation."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.env import FluxKontextEnv


class FluxKontextOfficialBackend(BaseEditingBackend):
    """Flux Kontext Official backend using BFL's official implementation.

    Uses vendored flux1 third-party code from diffgentor/models/third_party/flux1/.

    Model-specific parameters via environment variables:
        DG_FLUX_KONTEXT_OFFLOAD: Enable model offloading (default: false)
        DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH: Max sequence length (default: 512)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize Flux Kontext Official backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._t5 = None
        self._clip = None
        self._model = None
        self._ae = None
        self._offload = FluxKontextEnv.offload()
        self._max_sequence_length = FluxKontextEnv.max_sequence_length()

    def load_model(self, **kwargs) -> None:
        """Load Flux Kontext Official model.

        Model-specific parameters are read from environment variables:
            DG_FLUX_KONTEXT_OFFLOAD: Enable CPU offload for reduced VRAM
            DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH: Max sequence length

        Args:
            **kwargs: Additional arguments
        """
        # Read from environment variables
        self._offload = FluxKontextEnv.offload()
        self._max_sequence_length = FluxKontextEnv.max_sequence_length()

        # Add third-party path
        third_party_path = Path(__file__).parent.parent.parent / "models" / "third_party" / "flux1"
        flux_src_path = third_party_path / "src"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
            if flux_src_path.exists():
                sys.path.insert(0, str(flux_src_path))
        else:
            raise ImportError(
                f"Flux1 vendored code not found at {third_party_path}. "
                "Please reinstall diffgentor: pip install diffgentor"
            )

        try:
            from flux.util import load_ae, load_clip, load_flow_model, load_t5

            print_rank0(f"Loading Flux Kontext Official from: {self.model_name}")
            print_rank0(f"Offload enabled: {self._offload}")

            device = self.backend_config.device
            torch_device = torch.device(device)

            # Load model components
            self._t5 = load_t5(torch_device, max_length=self._max_sequence_length)
            self._clip = load_clip(torch_device)
            self._model = load_flow_model(
                "flux-dev-kontext",
                device="cpu" if self._offload else torch_device,
            )
            self._ae = load_ae(
                "flux-dev-kontext",
                device="cpu" if self._offload else torch_device,
            )

            self._initialized = True
            print_rank0("Flux Kontext Official model loaded")

        except ImportError as e:
            raise ImportError(
                f"Failed to import flux module: {e}. "
                "Make sure the flux1 submodule is properly initialized."
            )

    def _encode_extra_image(
        self,
        image: Image.Image,
        image_index: int,
        torch_device: torch.device,
    ) -> tuple:
        """Encode an additional conditioning image for multi-image Kontext input.

        Each extra image is encoded through the AE and assigned a unique positional ID
        (image_index) in the first dimension of img_cond_ids so the model can distinguish
        multiple conditioning images.

        Args:
            image: PIL image to encode.
            image_index: Positional index for this image (2 for second image, 3 for third, etc.).
                The first conditioning image uses index 1 (set by prepare_kontext), and the
                noise image uses index 0.
            torch_device: Device to run encoding on.

        Returns:
            Tuple of (img_cond_seq, img_cond_seq_ids) tensors ready to concatenate.
        """
        from flux.util import PREFERED_KONTEXT_RESOLUTIONS
        from einops import rearrange, repeat

        img = image.convert("RGB")
        w_orig, h_orig = img.size
        aspect_ratio = w_orig / h_orig

        # Match to closest preferred Kontext resolution
        _, width, height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
        )
        width = 2 * int(width / 16)
        height = 2 * int(height / 16)

        img = img.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).float() / 127.5 - 1.0
        img_tensor = rearrange(img_tensor, "h w c -> 1 c h w")

        with torch.no_grad():
            img_cond = self._ae.encode(img_tensor.to(torch_device))

        img_cond = img_cond.to(torch.bfloat16)
        img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        # Build positional IDs: first dim = image_index, then h/w grid
        img_cond_ids = torch.zeros(height // 2, width // 2, 3)
        img_cond_ids[..., 0] = image_index
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=1)

        return img_cond, img_cond_ids.to(torch_device)

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit images using Flux Kontext Official.

        Supports multi-image conditioning: the first image is processed through
        prepare_kontext (assigned positional ID 1), and each additional image is
        encoded separately and concatenated along the sequence dimension with
        incrementing positional IDs (2, 3, ...).

        Args:
            images: Input image(s) to edit
            instruction: Editing instruction
            num_inference_steps: Number of inference steps (default: 30)
            guidance_scale: Guidance scale (default: 2.5)
            seed: Random seed
            **kwargs: Additional arguments (width, height for target size)

        Returns:
            List of edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 30
        if guidance_scale is None:
            guidance_scale = 2.5

        # Normalize images
        if isinstance(images, Image.Image):
            images = [images]

        if not images:
            raise ValueError("No input image provided")

        # Convert all images to RGB
        images = [img.convert("RGB") for img in images]

        temp_paths: List[str] = []
        try:
            from flux.sampling import denoise, get_schedule, prepare_kontext, unpack

            device = self.backend_config.device
            torch_device = torch.device(device)

            # Set seed
            rng = torch.Generator(device="cpu")
            if seed is not None:
                rng.manual_seed(seed)
            else:
                seed = rng.seed()

            print_rank0(f"Generating with seed {seed}")

            # Save first image to temp file for prepare_kontext
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_paths.append(f.name)
                images[0].save(f.name)

            # Get target size from kwargs or use input image size
            target_width = kwargs.get("width")
            target_height = kwargs.get("height")

            # Offload TEs and AE to GPU if needed
            if self._offload:
                self._t5 = self._t5.to(torch_device)
                self._clip = self._clip.to(torch_device)
                self._ae = self._ae.to(torch_device)

            # Prepare input with first conditioning image
            inp, height, width = prepare_kontext(
                t5=self._t5,
                clip=self._clip,
                prompt=instruction,
                ae=self._ae,
                img_cond_path=temp_paths[0],
                seed=seed,
                device=torch_device,
                target_width=target_width,
                target_height=target_height,
            )

            # Remove img_cond_orig (not consumed by denoise)
            inp.pop("img_cond_orig", None)

            # Encode additional conditioning images and concatenate
            if len(images) > 1:
                print_rank0(f"Multi-image input: encoding {len(images)} conditioning images")
                for extra_idx, extra_image in enumerate(images[1:], start=2):
                    extra_cond, extra_ids = self._encode_extra_image(
                        extra_image, image_index=extra_idx, torch_device=torch_device
                    )
                    inp["img_cond_seq"] = torch.cat(
                        (inp["img_cond_seq"], extra_cond), dim=1
                    )
                    inp["img_cond_seq_ids"] = torch.cat(
                        (inp["img_cond_seq_ids"], extra_ids), dim=1
                    )

            # Offload TEs and AE, load transformer to GPU
            if self._offload:
                self._t5 = self._t5.cpu()
                self._clip = self._clip.cpu()
                self._ae = self._ae.cpu()
                torch.cuda.empty_cache()
                self._model = self._model.to(torch_device)

            # Get schedule
            timesteps = get_schedule(
                num_inference_steps,
                inp["img"].shape[1],
                shift=True,
            )

            # Denoise
            x = denoise(
                self._model,
                **inp,
                timesteps=timesteps,
                guidance=guidance_scale,
            )

            # Offload transformer, load AE decoder to GPU
            if self._offload:
                self._model = self._model.cpu()
                torch.cuda.empty_cache()
                self._ae = self._ae.to(torch_device)

            # Decode latents to pixel space
            x = unpack(x.float(), height, width)
            with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                x = self._ae.decode(x)

            if self._offload:
                self._ae = self._ae.cpu()
                torch.cuda.empty_cache()

            # Convert to PIL
            x = x.clamp(-1, 1)
            x = x[0].permute(1, 2, 0)
            x = (127.5 * (x + 1.0)).cpu().byte().numpy()
            result_image = Image.fromarray(x)

            return [result_image]

        except Exception as e:
            print_rank0(f"Flux Kontext edit failed: {e}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            # Clean up all temp files
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._t5 is not None:
            del self._t5
            self._t5 = None
        if self._clip is not None:
            del self._clip
            self._clip = None
        if self._model is not None:
            del self._model
            self._model = None
        if self._ae is not None:
            del self._ae
            self._ae = None
        torch.cuda.empty_cache()
