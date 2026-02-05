# Prompt Enhancement

Diffgentor supports optional prompt enhancement to improve editing instructions before processing. This uses LLM APIs to rewrite vague or simple prompts into more detailed, generation-friendly descriptions.

## Quick Start

```bash
# Enable prompt enhancement with Qwen style
export DG_PROMPT_ENHANCER_API_KEY=your_api_key
export DG_PROMPT_ENHANCER_API_BASE=https://api.example.com/v1
export DG_PROMPT_ENHANCER_MODEL=gpt-4o

diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --prompt_enhance_type qwen_image_edit
```

## Enhancer Types

| Type | Description | Best For |
|------|-------------|----------|
| `qwen_image_edit` | Optimized for image editing tasks | Add/delete/replace, text editing, style conversion |
| `glm_image` | GLM-Image style optimization | General image generation/editing |
| `flux2` | FLUX.2 style prompt enhancement | T2I generation, image editing with FLUX.2 |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_PROMPT_ENHANCER_API_KEY` | API key for LLM service | - |
| `DG_PROMPT_ENHANCER_API_BASE` | Base URL for API | OpenAI default |
| `DG_PROMPT_ENHANCER_MODEL` | Model name to use | `gpt-4o` |
| `DG_PROMPT_ENHANCER_TEMPERATURE` | Generation temperature | `0.15` |
| `DG_PROMPT_ENHANCER_DEBUG` | Enable debug mode | `false` |
| `DG_PROMPT_ENHANCER_DEBUG_DIR` | Debug output directory | `./debug_output` |

### Flux2 Specific Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_FLUX2_ENHANCER_MODE` | Mode: `diffusers` or `api` | `diffusers` |

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--prompt_enhance_type` | Enhancer type (`qwen_image_edit`, `glm_image`, `flux2`) |

## Enhancer Details

### qwen_image_edit

Designed for image editing tasks with detailed rewriting rules:

- **Add/Delete/Replace**: Supplements missing details (category, color, size, position)
- **Text Editing**: Preserves text content in quotes, handles replacement tasks
- **Human Editing**: Maintains identity consistency, natural expression changes
- **Style Conversion**: Extracts key visual characteristics from reference
- **Content Filling**: Uses fixed templates for inpainting/outpainting

**Example transformation:**

```
Original: "Add an animal"
Enhanced: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"
```

**Features:**
- Supports multi-image input for context
- Returns JSON response with `Rewritten` field
- Auto-retries on API failures

### glm_image

Optimized for GLM-Image model with automatic language detection:

- **Chinese Input**: Uses Chinese system prompt, outputs Chinese description
- **English Input**: Uses English system prompt, outputs English description

**Optimization strategies:**
- Realistic portrait scenes
- Text-centric scenes
- General image scenes

**Features:**
- Preserves proper nouns (names, brands, IPs)
- Specifies visual style (photography, illustration, 3D, etc.)
- Handles text content with quotation marks

### flux2

FLUX.2 style prompt enhancement supporting both text-to-image and image-to-image modes. Based on the official FLUX.2 prompt upsampling approach.

**Two operation modes:**

- **diffusers mode** (default): Uses the Flux2Pipeline's built-in `upsample_prompt` method, sharing the same Mistral model used for text encoding. No additional model loading required.
- **api mode**: Uses external API to call Mistral-Small-3.2-24B-Instruct-2506 model (or custom model).

**T2I Enhancement (no images provided):**
- Rewrites prompts to be more descriptive while preserving core subject and intent
- Adds concrete visual specifics: form, scale, textures, materials, lighting, shadows
- Handles text in images with quotation marks

**I2I Enhancement (with images):**
- Converts editing requests into concise instructions (50-80 words)
- Uses clear, analytical language
- Specifies what changes AND what stays the same

**Example transformation (T2I):**

```
Original: "A cat on the moon"
Enhanced: "A fluffy orange tabby cat sitting on the gray, crater-covered surface of the moon, with Earth visible in the starry black sky background, soft rim lighting from the sun"
```

**Example transformation (I2I):**

```
Original: "Make it futuristic"
Enhanced: "Transform the scene with glowing cyan neon accents, metallic panels, and holographic displays. Keep the original composition, subject pose, and lighting direction intact."
```

**Features:**
- Supports both T2I and I2I modes (auto-detected by image presence)
- Shares Mistral model with Flux2Pipeline in diffusers mode
- Configurable temperature (default: 0.15)

## Usage Examples

### Basic Enhancement

```bash
# Using OpenAI-compatible API
export DG_PROMPT_ENHANCER_API_KEY=sk-xxx
export DG_PROMPT_ENHANCER_API_BASE=https://api.openai.com/v1
export DG_PROMPT_ENHANCER_MODEL=gpt-4o

diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --prompt_enhance_type qwen_image_edit
```

### With Custom API Endpoint

```bash
# Using self-hosted or alternative API
export DG_PROMPT_ENHANCER_API_KEY=your_key
export DG_PROMPT_ENHANCER_API_BASE=https://your-api.example.com/v1
export DG_PROMPT_ENHANCER_MODEL=qwen-vl-plus

diffgentor edit --backend openai \
    --model_name gpt-image-1 \
    --input data.csv \
    --prompt_enhance_type glm_image
```

### Inline Variables

```bash
DG_PROMPT_ENHANCER_API_KEY=sk-xxx \
DG_PROMPT_ENHANCER_MODEL=gpt-4o \
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --prompt_enhance_type qwen_image_edit
```

### Flux2 with Diffusers Mode

```bash
# Default mode - uses pipeline's built-in Mistral model
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2 \
    --input data.csv \
    --prompt_enhance_type flux2
```

### Flux2 with API Mode

```bash
# Use external API for prompt enhancement
export DG_FLUX2_ENHANCER_MODE=api
export DG_PROMPT_ENHANCER_API_KEY=your_key
export DG_PROMPT_ENHANCER_API_BASE=https://api.example.com/v1
export DG_PROMPT_ENHANCER_MODEL=Mistral-Small-3.2-24B-Instruct-2506

diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2 \
    --input data.csv \
    --prompt_enhance_type flux2
```

## Debug Mode

Enable debug mode to save enhancement logs:

```bash
export DG_PROMPT_ENHANCER_DEBUG=true
export DG_PROMPT_ENHANCER_DEBUG_DIR=./debug_output

diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --prompt_enhance_type qwen_image_edit
```

Debug logs are saved to `{DG_PROMPT_ENHANCER_DEBUG_DIR}/prompt_enhancer/{idx}.log` with:
- Original prompt
- Enhanced prompt
- Raw LLM response
- Error messages (if any)

## Programmatic Usage

### Basic Usage

```python
from diffgentor.prompt_enhance import get_prompt_enhancer
from PIL import Image

# Initialize enhancer
enhancer = get_prompt_enhancer(
    enhancer_type="qwen_image_edit",
    api_key="your-api-key",
    api_base="https://api.example.com/v1",
    model="gpt-4o"
)

# Enhance with image context
image = Image.open("input.jpg")
enhanced = enhancer.enhance(
    prompt="Add a cat",
    images=[image]
)
print(enhanced)
# Output: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"
```

### Flux2 with Diffusers Mode

```python
from diffusers import Flux2Pipeline
from diffgentor.prompt_enhance import Flux2PromptEnhancer

# Load Flux2 pipeline (includes Mistral model)
pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2")

# Create enhancer and share the pipeline's Mistral model
enhancer = Flux2PromptEnhancer()
enhancer.set_pipeline(pipe)

# T2I enhancement (no images)
enhanced = enhancer.enhance("A cat on the moon")

# I2I enhancement (with images)
image = Image.open("input.jpg")
enhanced = enhancer.enhance("Make it futuristic", images=[image])

# Use the same pipeline for generation
result = pipe(enhanced, ...)
```

### Flux2 with API Mode

```python
import os
from diffgentor.prompt_enhance import Flux2PromptEnhancer

# Set API mode via environment variable
os.environ["DG_FLUX2_ENHANCER_MODE"] = "api"
os.environ["DG_PROMPT_ENHANCER_API_KEY"] = "your-api-key"
os.environ["DG_PROMPT_ENHANCER_API_BASE"] = "https://api.example.com/v1"

# Or pass directly
enhancer = Flux2PromptEnhancer(
    api_key="your-api-key",
    api_base="https://api.example.com/v1",
    model="Mistral-Small-3.2-24B-Instruct-2506"
)

enhanced = enhancer.enhance("A beautiful sunset")
```

## Custom Enhancer

Create a custom enhancer by extending `PromptEnhancer`:

```python
from diffgentor.prompt_enhance.base import PromptEnhancer
from diffgentor.prompt_enhance.registry import register_enhancer

class MyCustomEnhancer(PromptEnhancer):
    def enhance(self, prompt, images=None):
        # Your enhancement logic
        messages = [
            {"role": "system", "content": "Your system prompt"},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

# Register the enhancer
register_enhancer(
    "my_custom",
    "your_module.my_enhancer",
    "MyCustomEnhancer"
)
```

Then use it:

```bash
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --prompt_enhance_type my_custom
```

## Notes

- Prompt enhancement adds latency (LLM API call per prompt)
- Enhancement only applies to the instruction, not the image
- On API failure, original prompt is used after max retries
- `qwen_image_edit` supports multiple input images for better context
- `glm_image` does not use input images for enhancement
- `flux2` in diffusers mode shares the Mistral model with Flux2Pipeline (no extra memory)
- `flux2` auto-detects T2I vs I2I mode based on whether images are provided
