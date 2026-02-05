# Diffusers Editing Backend

The diffusers backend supports multiple HuggingFace-based editing models with automatic pipeline detection.

## Default Parameter Behavior

When `--num_inference_steps` or `--guidance_scale` are not specified, each model automatically uses its recommended default parameters. This ensures optimal performance without manual tuning.

## Supported Models

| Model Type | Pipeline Class | Multi-Image | True CFG |
|------------|----------------|-------------|----------|
| `qwen` | QwenImageEditPlusPipeline | ✓ | ✓ |
| `qwen_singleimg` | QwenImageEditPipeline | ✗ | ✓ |
| `flux2` | Flux2Pipeline | ✓ | ✗ |
| `flux2_klein` | Flux2Pipeline | ✓ | ✗ |
| `flux1_kontext` | FluxKontextPipeline | ✓ | ✗ |
| `longcat` | LongCatImageEditPipeline | ✓ | ✗ |
| `glm_image` | GlmImagePipeline | ✗ | ✗ |

## Model Auto-Detection

Models are auto-detected by name pattern:

| Pattern | Model Type |
|---------|------------|
| `Qwen-Image-Edit-2511` | qwen |
| `Qwen-Image-Edit-2509` | qwen |
| `Qwen-Image-Edit-2511` | qwen |
| `Qwen-Image-Edit` | qwen_singleimg |
| `FLUX.2-dev` | flux2 |
| `FLUX.2-klein` | flux2_klein |
| `FLUX.1-Kontext` | flux1_kontext |
| `LongCat` | longcat |
| `GLM-Image` | glm_image |

Or specify explicitly with `--model_type`.

## Qwen-Image-Edit

Alibaba's multimodal image editing model series.

### Qwen-Image-Edit-2511 (Multi-Image)

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --model_type qwen \
    --input data.csv \
    --num_inference_steps 40 \
    --guidance_scale 1.0 \
    --true_cfg_scale 4.0
```

**Model Default Parameters:**
- Steps: 40
- Guidance Scale: 1.0
- True CFG Scale: 4.0

### Qwen-Image-Edit (Single-Image)

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit \
    --model_type qwen_singleimg \
    --input data.csv \
    --num_inference_steps 50 \
    --guidance_scale 1.0 \
    --true_cfg_scale 4.0
```

**Model Default Parameters:**
- Steps: 50
- Guidance Scale: 1.0
- True CFG Scale: 4.0

### Qwen-Image-Edit with Prompt Enhancement

Qwen models support the `qwen_image_edit` prompt enhancer via API:

```bash
export DG_PROMPT_ENHANCER_API_KEY=your_api_key
export DG_PROMPT_ENHANCER_API_BASE=https://api.example.com/v1
export DG_PROMPT_ENHANCER_MODEL="Qwen3-VL-235B-A22B-Instruct-FP8"

diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --model_type qwen \
    --input data.csv \
    --prompt_enhance_type qwen_image_edit
```

See [Prompt Enhancement Guide](../prompt_enhance.md) for more details.

## FLUX.2 Models

Black Forest Labs FLUX 2.0 editing models.

### FLUX.2-dev

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-dev \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-dev \
    --model_type flux2 \
    --input data.csv \
    --num_inference_steps 28 \
    --guidance_scale 4.0
```

**Model Default Parameters:**
- Steps: 28
- Guidance Scale: 4.0

### FLUX.2 with Prompt Enhancement

FLUX.2 supports the `flux2` prompt enhancer with two modes:

**Diffusers Mode (Default):**

Uses the pipeline's built-in Mistral model for prompt enhancement. No additional model loading required.

```bash
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-dev \
    --model_type flux2 \
    --input data.csv \
    --prompt_enhance_type flux2
```

**API Mode:**

Uses external API for prompt enhancement (e.g., Mistral-Small-3.2-24B-Instruct-2506).

```bash
export DG_FLUX2_ENHANCER_MODE=api
export DG_PROMPT_ENHANCER_API_KEY=your_api_key
export DG_PROMPT_ENHANCER_API_BASE=https://api.example.com/v1
export DG_PROMPT_ENHANCER_MODEL=Mistral-Small-3.2-24B-Instruct-2506

diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-dev \
    --model_type flux2 \
    --input data.csv \
    --prompt_enhance_type flux2
```

See [Prompt Enhancement Guide](../prompt_enhance.md) for more details.

### FLUX.2-klein (4B/9B)

Faster, smaller FLUX 2.0 variants.

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-klein \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-klein \
    --model_type flux2_klein \
    --input data.csv \
    --num_inference_steps 4 \
    --guidance_scale 1.0
```

**Model Default Parameters:**
- Steps: 4
- Guidance Scale: 1.0

## FLUX.1-Kontext

FLUX 1.0 with context-aware editing.

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.1-Kontext-dev \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.1-Kontext-dev \
    --model_type flux1_kontext \
    --input data.csv \
    --num_inference_steps 30 \
    --guidance_scale 2.5
```

**Model Default Parameters:**
- Steps: 30
- Guidance Scale: 2.5

## LongCat

LongCat image editing pipeline with negative prompt support.

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name LongCat/LongCat-Image-Edit \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name LongCat/LongCat-Image-Edit \
    --model_type longcat \
    --input data.csv \
    --num_inference_steps 50 \
    --guidance_scale 4.5 \
    --negative_prompt "blurry, low quality"
```

**Model Default Parameters:**
- Steps: 50
- Guidance Scale: 4.5

## GLM-Image

GLM-based image generation/editing.

```bash
# Use model default parameters (recommended)
diffgentor edit --backend diffusers \
    --model_name THUDM/GLM-Image \
    --input data.csv

# Or specify custom parameters
diffgentor edit --backend diffusers \
    --model_name THUDM/GLM-Image \
    --model_type glm_image \
    --input data.csv \
    --num_inference_steps 50 \
    --guidance_scale 1.5
```

**Model Default Parameters:**
- Steps: 50
- Guidance Scale: 1.5

### GLM-Image with Prompt Enhancement

GLM-Image supports the `glm_image` prompt enhancer via API:

```bash
export DG_PROMPT_ENHANCER_API_KEY=your_api_key
export DG_PROMPT_ENHANCER_API_BASE=https://api.example.com/v1
export DG_PROMPT_ENHANCER_MODEL="GLM-4.7"

diffgentor edit --backend diffusers \
    --model_name THUDM/GLM-Image \
    --model_type glm_image \
    --input data.csv \
    --prompt_enhance_type glm_image
```

See [Prompt Enhancement Guide](../prompt_enhance.md) for more details.

## Optimizations

All diffusers models support the standard optimization options:

```bash
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --enable_compile \
    --enable_vae_slicing \
    --enable_vae_tiling \
    --enable_cpu_offload \
    --attention_backend flash \
    --torch_dtype bfloat16
```

See [Optimization Guide](../optimization.md) for details.
