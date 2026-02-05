# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Qwen Image Edit style prompt enhancer.
Optimized for image editing tasks with detailed rewriting rules.
"""

import json
import time
from typing import List, Optional

from PIL import Image

from diffgentor.prompt_enhance.base import PromptEnhancer, encode_image_to_base64


class QwenImageEditEnhancer(PromptEnhancer):
    """
    Prompt enhancer using the Qwen Image Edit style prompt.
    This enhancer is optimized for image editing tasks.
    """

    # System prompt for edit instruction enhancement
    EDIT_SYSTEM_PROMPT = '''# Edit Prompt Enhancer
You are a professional edit prompt enhancer. Your task is to generate a direct and specific edit prompt based on the user-provided instruction and the image input conditions.
Please strictly follow the enhancing rules below:
## 1. General Principles
- Keep the enhanced prompt **direct and specific**.
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.
- All added objects or modifications must align with the logic and style of the edited input image's overall scene.
## 2. Task-Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:
    > Original: "Add an animal"
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.
### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Keep the original language of the text, and keep the capitalization.
- Both adding new text and replacing existing text are text replacement tasks, For example:
    - Replace "xx" to "yy"
    - Replace the mask / bounding box to "yy"
    - Replace the visual object to "yy"
- Specify text position, color, and layout only if user has required.
- If font is specified, keep the original language of the font.
### 3. Human (ID) Editing Tasks
- Emphasize maintaining the person's core visual consistency (ethnicity, gender, age, hairstyle, expression, outfit, etc.).
- If modifying appearance (e.g., clothes, hairstyle), ensure the new element is consistent with the original style.
- **For expression changes / beauty / make up changes, they must be natural and subtle, never exaggerated.**
- Example:
    > Original: "Change the person's hat"
    > Rewritten: "Replace the man's hat with a dark brown beret; keep smile, short hair, and gray jacket unchanged"
### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:
    > Original: "Disco style"
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, colorful tones"
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.
- **Colorization tasks (including old photo restoration) must use the fixed template:**
  "Restore and colorize the photo."
- Clearly specify the object to be modified. For example:
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.
- If there are other changes, place the style description at the end.
### 5. Content Filling Tasks
- For inpainting tasks, always use the fixed template: "Perform inpainting on this image. The original caption is: ".
- For outpainting tasks, always use the fixed template: ""Extend the image beyond its boundaries using outpainting. The original caption is: ".
### 6. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified. For example:
    > Original: "Replace the subject of picture 1 with the subject of picture 2"
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged"
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.
## 3. Rationale and Logic Checks
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" should be logically corrected.
- Add missing key information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edge, etc.).
# Output Format Example
```json
{
   "Rewritten": "..."
}'''

    def enhance(self, prompt: str, images: Optional[List[Image.Image]] = None) -> str:
        """
        Enhance an edit prompt using the Qwen Image Edit style.

        Args:
            prompt: Original edit instruction
            images: Optional list of input images for context (all images will be sent)

        Returns:
            Enhanced prompt string
        """
        # Build messages
        messages = [
            {"role": "system", "content": self.EDIT_SYSTEM_PROMPT},
        ]

        # Build user message content
        user_content = []

        # Add all images if available (supports multi-image input)
        if images and len(images) > 0:
            for img in images:
                img_base64 = encode_image_to_base64(img)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })

        # Add text prompt
        user_content.append({
            "type": "text",
            "text": f"User Input: {prompt}\n\nRewritten Prompt:"
        })

        messages.append({"role": "user", "content": user_content})

        # Variables for debug output
        raw_response_text = ""
        error_msg = None

        # Call API with retries
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"} if "gpt" in self.model.lower() or "qwen" in self.model.lower() else None,
                )

                result_text = response.choices[0].message.content
                raw_response_text = result_text  # Store for debug

                # Parse JSON response
                result_text = result_text.strip()
                if result_text.startswith("```json"):
                    result_text = result_text[7:]
                if result_text.startswith("```"):
                    result_text = result_text[3:]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()

                result = json.loads(result_text)
                enhanced_prompt = result.get("Rewritten", prompt)
                enhanced_prompt = enhanced_prompt.strip().replace("\n", " ")

                # Save debug info on success
                self._save_debug_info(
                    original_prompt=prompt,
                    enhanced_prompt=enhanced_prompt,
                    raw_response=raw_response_text,
                    messages=messages,
                    images=images,
                    error=None,
                )

                return enhanced_prompt

            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error: {e}"
                print(f"[Warning] JSON parse error on attempt {attempt + 1}/{self.max_retries}: {e}")
                print(f"[Warning] Raw response: {raw_response_text[:500]}...")
                if attempt < self.max_retries - 1:
                    continue
                # If all retries fail, return original prompt
                print(f"[Warning] All retries failed, returning original prompt")
                # Save debug info on failure
                self._save_debug_info(
                    original_prompt=prompt,
                    enhanced_prompt=prompt,
                    raw_response=raw_response_text,
                    messages=messages,
                    images=images,
                    error=error_msg,
                )
                return prompt

            except Exception as e:
                error_msg = f"API call error: {e}"
                print(f"[Warning] API call error on attempt {attempt + 1}/{self.max_retries}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                    continue
                # If all retries fail, return original prompt
                print(f"[Warning] All retries failed, returning original prompt")
                # Save debug info on failure
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
