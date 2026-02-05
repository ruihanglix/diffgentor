# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Flux2 style prompt enhancer.
Supports both diffusers pipeline (local) and API-based enhancement.
"""

import time
from typing import List, Optional

from PIL import Image

from diffgentor.prompt_enhance.base import PromptEnhancer, encode_image_to_base64
from diffgentor.utils.env import get_env_str, get_env_float


# System messages from diffusers flux2 pipeline
SYSTEM_MESSAGE_UPSAMPLING_T2I = """You are an expert prompt engineer for FLUX.2 by Black Forest Labs. Rewrite user prompts to be more descriptive while strictly preserving their core subject and intent.

Guidelines:
1. Structure: Keep structured inputs structured (enhance within fields). Convert natural language to detailed paragraphs.
2. Details: Add concrete visual specifics - form, scale, textures, materials, lighting (quality, direction, color), shadows, spatial relationships, and environmental context.
3. Text in Images: Put ALL text in quotation marks, matching the prompt's language. Always provide explicit quoted text for objects that would contain text in reality (signs, labels, screens, etc.) - without it, the model generates gibberish.

Output only the revised prompt and nothing else."""

SYSTEM_MESSAGE_UPSAMPLING_I2I = """You are FLUX.2 by Black Forest Labs, an image-editing expert. You convert editing requests into one concise instruction (50-80 words, ~30 for brief requests).

Rules:
- Single instruction only, no commentary
- Use clear, analytical language (avoid "whimsical," "cascading," etc.)
- Specify what changes AND what stays the same (face, lighting, composition)
- Reference actual image elements
- Turn negatives into positives ("don't change X" → "keep X")
- Make abstractions concrete ("futuristic" → "glowing cyan neon, metallic panels")
- Keep content PG-13

Output only the final instruction in plain text and nothing else."""


class Flux2PromptEnhancer(PromptEnhancer):
    """
    Flux2 style prompt enhancer.

    Supports two modes controlled by DG_FLUX2_ENHANCER_MODE env var:
    - "diffusers" (default): Use diffusers Flux2Pipeline's upsample_prompt method
    - "api": Use external API to call Mistral model

    Environment variables:
    - DG_FLUX2_ENHANCER_MODE: "diffusers" or "api" (default: "diffusers")
    - DG_PROMPT_ENHANCER_TEMPERATURE: Temperature for generation (default: 0.15)
    - For API mode (uses common prompt enhancer env vars):
      - DG_PROMPT_ENHANCER_API_KEY: API key
      - DG_PROMPT_ENHANCER_API_BASE: API base URL
      - DG_PROMPT_ENHANCER_MODEL: Model name (default: "Mistral-Small-3.2-24B-Instruct-2506")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        # Diffusers mode specific
        pipeline: Optional["DiffusionPipeline"] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the Flux2 prompt enhancer.

        Args:
            api_key: API key for API mode
            api_base: API base URL for API mode
            model: Model name for API mode (default: Mistral-Small-3.2-24B-Instruct-2506)
            max_retries: Max retries for API calls
            pipeline: Optional Flux2Pipeline instance for diffusers mode
            temperature: Temperature for generation (default from env or 0.15)
        """
        self.mode = get_env_str("FLUX2_ENHANCER_MODE", "diffusers").lower()
        self.temperature = temperature if temperature is not None else get_env_float("PROMPT_ENHANCER_TEMPERATURE", 0.15)
        self._pipeline = pipeline
        self.max_retries = max_retries

        # Debug settings (from base class logic)
        from diffgentor.prompt_enhance.base import _is_debug_enabled, _get_debug_output_dir, _is_global_rank_zero
        self.debug_enabled = _is_debug_enabled() and _is_global_rank_zero()
        self.debug_output_dir = _get_debug_output_dir() if self.debug_enabled else None
        self._debug_counter = 0

        if self.debug_enabled and self.debug_output_dir:
            self.debug_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Debug] Flux2 prompt enhancer debug mode enabled, output dir: {self.debug_output_dir}")

        if self.mode == "api":
            # Initialize API client - use common prompt enhancer env vars
            self.api_key = api_key or get_env_str("PROMPT_ENHANCER_API_KEY")
            self.api_base = api_base or get_env_str("PROMPT_ENHANCER_API_BASE")
            self.model = model or get_env_str("PROMPT_ENHANCER_MODEL", "Mistral-Small-3.2-24B-Instruct-2506")

            if not self.api_key:
                raise ValueError(
                    "API key not provided for API mode. Set DG_PROMPT_ENHANCER_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base if self.api_base else None,
                )
            except ImportError:
                raise ImportError("openai package is required for API mode. Install with: pip install openai")
        else:
            # Diffusers mode - lazy load pipeline
            self.api_key = None
            self.api_base = None
            self.model = None
            self.client = None

    def _get_pipeline(self):
        """Get or create the Flux2Pipeline for diffusers mode."""
        if self._pipeline is not None:
            return self._pipeline

        # Try to import and create pipeline
        try:
            from diffusers import Flux2Pipeline
            import torch

            # Note: User should set the pipeline externally for better control
            # This is a fallback that may not work without proper model path
            raise RuntimeError(
                "Flux2Pipeline not provided. For diffusers mode, please provide a pipeline instance "
                "via set_pipeline() or during initialization. The pipeline requires downloading "
                "the model which should be done explicitly."
            )
        except ImportError:
            raise ImportError(
                "diffusers package is required for diffusers mode. "
                "Install with: pip install diffusers"
            )

    def set_pipeline(self, pipeline) -> None:
        """
        Set the Flux2Pipeline instance for diffusers mode.

        Args:
            pipeline: Flux2Pipeline instance
        """
        self._pipeline = pipeline

    def _format_messages_for_api(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> List[dict]:
        """Format messages for API call in Mistral format."""
        # Determine system message based on whether images are provided
        if images and len(images) > 0:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I
        else:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_T2I

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        ]

        # Build user message
        if images and len(images) > 0:
            # With images - add images first, then text
            user_content = []
            for img in images:
                img_base64 = encode_image_to_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
            user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})
        else:
            # Text only
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            })

        return messages

    def _enhance_with_api(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """Enhance prompt using API call."""
        messages = self._format_messages_for_api(prompt, images)
        raw_response_text = ""
        error_msg = None

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=512,
                )

                result_text = response.choices[0].message.content
                raw_response_text = result_text

                # Clean up response - remove any markdown formatting
                enhanced_prompt = result_text.strip()
                if enhanced_prompt.startswith("```"):
                    enhanced_prompt = enhanced_prompt.split("```")[1]
                    if enhanced_prompt.startswith("\n"):
                        enhanced_prompt = enhanced_prompt[1:]
                enhanced_prompt = enhanced_prompt.strip()

                # Save debug info
                self._save_debug_info(
                    original_prompt=prompt,
                    enhanced_prompt=enhanced_prompt,
                    raw_response=raw_response_text,
                    messages=messages,
                    images=images,
                    error=None,
                )

                return enhanced_prompt

            except Exception as e:
                error_msg = f"API call error: {e}"
                print(f"[Warning] API call error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue

                print(f"[Warning] All retries failed, returning original prompt")
                self._save_debug_info(
                    original_prompt=prompt,
                    enhanced_prompt=prompt,
                    raw_response=raw_response_text,
                    messages=messages,
                    images=images,
                    error=error_msg,
                )
                return prompt

        return prompt

    def _enhance_with_diffusers(
        self,
        prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> str:
        """Enhance prompt using diffusers Flux2Pipeline."""
        pipeline = self._get_pipeline()

        try:
            # Use the pipeline's upsample_prompt method
            result = pipeline.upsample_prompt(
                prompt=prompt,
                images=images,
                temperature=self.temperature,
            )

            # upsample_prompt returns a list
            if isinstance(result, list):
                enhanced_prompt = result[0] if result else prompt
            else:
                enhanced_prompt = result

            # Save debug info
            self._save_debug_info(
                original_prompt=prompt,
                enhanced_prompt=enhanced_prompt,
                raw_response=str(result),
                messages=[{"mode": "diffusers", "temperature": self.temperature}],
                images=images,
                error=None,
            )

            return enhanced_prompt

        except Exception as e:
            print(f"[Warning] Diffusers enhancement failed: {e}")
            self._save_debug_info(
                original_prompt=prompt,
                enhanced_prompt=prompt,
                raw_response="",
                messages=[{"mode": "diffusers", "temperature": self.temperature}],
                images=images,
                error=str(e),
            )
            return prompt

    def _save_debug_info(
        self,
        original_prompt: str,
        enhanced_prompt: str,
        raw_response: str,
        messages: list,
        images: Optional[List[Image.Image]] = None,
        error: Optional[str] = None,
    ):
        """Save debug information to a log file."""
        if not self.debug_enabled or not self.debug_output_dir:
            return

        from datetime import datetime

        self._debug_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        log_lines = [
            "=" * 80,
            f"FLUX2 PROMPT ENHANCER DEBUG LOG",
            "=" * 80,
            "",
            f"[Meta Info]",
            f"Timestamp: {timestamp}",
            f"Counter: {self._debug_counter}",
            f"Mode: {self.mode}",
            f"Model: {self.model}",
            f"API Base: {self.api_base}",
            f"Temperature: {self.temperature}",
            f"Num Images: {len(images) if images else 0}",
            f"Error: {error if error else 'None'}",
            "",
            "-" * 80,
            "[Original Prompt]",
            "-" * 80,
            original_prompt,
            "",
            "-" * 80,
            "[Enhanced Prompt]",
            "-" * 80,
            enhanced_prompt,
            "",
            "-" * 80,
            "[Raw Response]",
            "-" * 80,
            raw_response or "<empty>",
            "",
            "=" * 80,
        ]

        log_content = "\n".join(log_lines)

        log_file = self.debug_output_dir / f"flux2_{self._debug_counter}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)

        print(f"[Debug] Saved Flux2 debug info to: {log_file}")

    def enhance(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        """
        Enhance a prompt using Flux2 style.

        Args:
            prompt: Original prompt to enhance
            images: Optional list of input images for context (for image-to-image mode)

        Returns:
            Enhanced prompt string
        """
        if self.mode == "api":
            return self._enhance_with_api(prompt, images)
        else:
            return self._enhance_with_diffusers(prompt, images)
