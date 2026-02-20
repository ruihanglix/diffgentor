# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DreamOmni2 backend for multimodal instruction-based generation/editing."""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image

from diffgentor.backends.base import BaseEditingBackend
from diffgentor.config import BackendConfig, OptimizationConfig
from diffgentor.utils.logging import print_rank0
from diffgentor.utils.env import DreamOmni2Env


class DreamOmni2Backend(BaseEditingBackend):
    """DreamOmni2 backend for multimodal instruction-based generation/editing.

    DreamOmni2 combines FLUX.1-Kontext with Qwen2.5-VL for instruction understanding.

    Requires the dreamomni2 submodule to be initialized:
        git submodule update --init diffgentor/models/third_party/dreamomni2

    Model-specific parameters via environment variables:
        DG_DREAMOMNI2_VLM_PATH: Path to VLM model (Qwen2.5-VL) - required for instruction enhancement
        DG_DREAMOMNI2_LORA_PATH: Path to LoRA weights (gen_lora or edit_lora)
        DG_DREAMOMNI2_TASK_TYPE: Task type - "generation" or "editing" (default: editing)
        DG_DREAMOMNI2_OUTPUT_HEIGHT: Output image height (default: from input)
        DG_DREAMOMNI2_OUTPUT_WIDTH: Output image width (default: from input)
    """

    def __init__(
        self,
        backend_config: BackendConfig,
        optimization_config: Optional[OptimizationConfig] = None,
    ):
        """Initialize DreamOmni2 backend.

        Args:
            backend_config: Backend configuration
            optimization_config: Optional optimization configuration
        """
        super().__init__(backend_config, optimization_config)
        self._pipe = None
        self._vlm = None
        self._processor = None
        self._vlm_path = DreamOmni2Env.vlm_path()
        self._lora_path = DreamOmni2Env.lora_path()
        self._task_type = DreamOmni2Env.task_type()
        self._output_height = DreamOmni2Env.output_height()
        self._output_width = DreamOmni2Env.output_width()

    def load_model(self, **kwargs) -> None:
        """Load DreamOmni2 model.

        Model-specific parameters are read from environment variables:
            DG_DREAMOMNI2_VLM_PATH: Path to VLM model (Qwen2.5-VL)
            DG_DREAMOMNI2_LORA_PATH: Path to LoRA weights
            DG_DREAMOMNI2_TASK_TYPE: Task type - "generation" or "editing"
            DG_DREAMOMNI2_OUTPUT_HEIGHT: Output image height
            DG_DREAMOMNI2_OUTPUT_WIDTH: Output image width

        Args:
            **kwargs: Additional arguments
        """
        # Read from environment variables
        self._vlm_path = DreamOmni2Env.vlm_path()
        self._lora_path = DreamOmni2Env.lora_path()
        self._task_type = DreamOmni2Env.task_type()
        self._output_height = DreamOmni2Env.output_height()
        self._output_width = DreamOmni2Env.output_width()

        # Add third-party path
        third_party_path = Path(__file__).parent.parent.parent / "models" / "third_party" / "dreamomni2"
        if third_party_path.exists():
            sys.path.insert(0, str(third_party_path))
        else:
            raise ImportError(
                f"DreamOmni2 vendored code not found at {third_party_path}. "
                "Please reinstall diffgentor: pip install diffgentor"
            )

        try:
            from dreamomni2.pipeline_dreamomni2 import DreamOmni2Pipeline

            print_rank0(f"Loading DreamOmni2 from: {self.model_name}")
            print_rank0(f"VLM path: {self._vlm_path}")
            print_rank0(f"LoRA path: {self._lora_path}")
            print_rank0(f"Task type: {self._task_type}")

            # Load base pipeline (FLUX.1-Kontext)
            self._pipe = DreamOmni2Pipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            )

            # Move to device
            device = self.backend_config.device
            if device == "cuda" and torch.cuda.is_available():
                self._pipe = self._pipe.to(device)
            elif device == "cuda" and not torch.cuda.is_available():
                print_rank0("WARNING: CUDA requested but not available, model will run on CPU")
            elif device != "cuda":
                self._pipe = self._pipe.to(device)

            # Load LoRA if provided
            if self._lora_path:
                adapter_name = "edit" if self._task_type == "editing" else "gen"
                self._pipe.load_lora_weights(self._lora_path, adapter_name=adapter_name)
                self._pipe.set_adapters([adapter_name], adapter_weights=[1.0])
                print_rank0(f"Loaded LoRA weights from: {self._lora_path}")

            # Load VLM for instruction understanding
            if self._vlm_path:
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

                self._vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self._vlm_path,
                    torch_dtype=torch.bfloat16,
                    device_map="cuda",
                )
                self._processor = AutoProcessor.from_pretrained(self._vlm_path)
                print_rank0(f"Loaded VLM from: {self._vlm_path}")

            self._initialized = True
            print_rank0("DreamOmni2 model loaded")

        except ImportError as e:
            raise ImportError(
                f"Failed to import DreamOmni2 modules: {e}. "
                "Make sure the dreamomni2 submodule is properly initialized."
            )

    def edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Edit/generate images using DreamOmni2.

        Args:
            images: Input image(s) - can be empty for generation task
            instruction: Editing/generation instruction
            num_inference_steps: Number of inference steps (default: 30)
            guidance_scale: Guidance scale (default: 3.5)
            seed: Random seed
            **kwargs: Additional arguments

        Returns:
            List of generated/edited PIL Images
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call load_model() first.")

        # Apply defaults
        if num_inference_steps is None:
            num_inference_steps = 30
        if guidance_scale is None:
            guidance_scale = 3.5

        # Normalize images
        if isinstance(images, Image.Image):
            images = [images]

        # Convert images to RGB and resize
        processed_images = []
        for img in images:
            img = img.convert("RGB")
            img = self._resize_input(img)
            processed_images.append(img)

        # Process instruction with VLM if available
        enhanced_prompt = instruction
        if self._vlm is not None and processed_images:
            enhanced_prompt = self._enhance_prompt_with_vlm(
                processed_images,
                instruction,
            )
            print_rank0(f"Enhanced prompt: {enhanced_prompt[:100]}...")

        # Prepare generator
        generator = None
        if seed is not None:
            device = self._pipe.device if hasattr(self._pipe, "device") else "cuda"
            generator = torch.Generator(device=device).manual_seed(seed)

        # Determine output size
        if processed_images:
            height = kwargs.get("height") or processed_images[0].height
            width = kwargs.get("width") or processed_images[0].width
        else:
            height = kwargs.get("height") or self._output_height
            width = kwargs.get("width") or self._output_width

        # Run generation/editing
        if self._task_type == "generation" or not processed_images:
            # Text-to-image generation
            output = self._pipe(
                prompt=enhanced_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            # Image editing
            output = self._pipe(
                images=processed_images,
                prompt=enhanced_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        # Extract images
        if hasattr(output, "images"):
            return output.images
        else:
            return [output]

    def _resize_input(self, image: Image.Image) -> Image.Image:
        """Resize input image to a preferred Kontext resolution.

        Args:
            image: Input image

        Returns:
            Resized image
        """
        try:
            from utils.vprocess import resizeinput
            return resizeinput(image)
        except ImportError:
            # Fallback: simple resize to max dimension
            max_dim = 1024
            w, h = image.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                # Round to nearest 16
                new_w = (new_w // 16) * 16
                new_h = (new_h // 16) * 16
                return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            return image

    def _enhance_prompt_with_vlm(
        self,
        images: List[Image.Image],
        instruction: str,
    ) -> str:
        """Enhance prompt using VLM understanding of the images.

        Args:
            images: Input images
            instruction: Original instruction

        Returns:
            Enhanced prompt
        """
        if self._vlm is None or self._processor is None:
            return instruction

        try:
            from utils.vprocess import process_vision_info

            # Build message content
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            
            prefix = " It is editing task." if self._task_type == "editing" else " It is generation task."
            content.append({"type": "text", "text": instruction + prefix})

            messages = [{"role": "user", "content": content}]

            # Process with VLM
            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Generate
            generated_ids = self._vlm.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=4096,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            # Extract content (remove <gen> tags if present)
            if output_text.startswith("<gen>") and output_text.endswith("</gen>"):
                output_text = output_text[5:-6]

            return output_text.strip() or instruction

        except Exception as e:
            print_rank0(f"VLM enhancement failed: {e}, using original instruction")
            return instruction

    def cleanup(self) -> None:
        """Clean up resources."""
        super().cleanup()
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._vlm is not None:
            del self._vlm
            self._vlm = None
        self._processor = None
        torch.cuda.empty_cache()
