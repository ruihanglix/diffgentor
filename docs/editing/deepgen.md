# DeepGen Backend

DeepGen is a unified visual generation model based on AR (Qwen2.5-VL) + Diffusion (SD3.5), supporting both text-to-image generation and image editing.

## Architecture

The model architecture consists of:
- **AR Model (Qwen2.5-VL)**: Language/vision understanding module
- **Diffusion Model (SD3.5)**: Image generation module
- **Connector**: Bridges LLM hidden states to DiT input space

## Prerequisites

Install additional dependencies:

```bash
pip install -e ".[deepgen]"
```

Required packages:
- `einops`
- `peft`
- `transformers>=4.40.0`
- `diffusers>=0.31.0`

## Configuration

DeepGen uses a Python config file system. Configuration is loaded from `diffgentor/models/deepgen/config/` directory.

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DG_DEEPGEN_CONFIG` | Config file name (e.g., `deepgen`) | **Yes** |
| `DG_DEEPGEN_DIFFUSION_MODEL_PATH` | Diffusion model path (SD3.5) | **Yes** |
| `DG_DEEPGEN_AR_MODEL_PATH` | AR model path (Qwen2.5-VL) | **Yes** |
| `DG_DEEPGEN_CHECKPOINT` | Model checkpoint path | No |
| `DG_DEEPGEN_MAX_LENGTH` | Maximum sequence length | No (default: 1024) |

### CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--guidance_scale` | CFG guidance scale | 4.0 |
| `--num_inference_steps` | Number of denoising steps | 50 |
| `--height` | Output image height | 512 |
| `--width` | Output image width | 512 |
| `--negative_prompt` | Negative prompt for CFG | "" |
| `--seed` | Random seed | Random |

## Basic Usage

### Image Editing

```bash
DG_DEEPGEN_CONFIG=deepgen \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output
```

### Text-to-Image Generation

```bash
DG_DEEPGEN_CONFIG=deepgen \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompt "A beautiful sunset over the ocean" \
    --output_dir ./output
```

## Model Files

The model requires two base models and an optional checkpoint:

### Diffusion Model (SD3.5)

Download from HuggingFace:
```bash
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ./sd3.5-medium
```

### AR Model (Qwen2.5-VL)

Download from HuggingFace:
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen2.5-vl-3b
```

### Checkpoint

The checkpoint contains the trained connector and LoRA weights. It should be a `.safetensors` or `.pt` file.

## Examples

### Basic Image Editing

```bash
DG_DEEPGEN_CONFIG=deepgen \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output
```

### Custom CFG Scale

```bash
DG_DEEPGEN_CONFIG=deepgen \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output \
    --guidance_scale 6.0 \
    --num_inference_steps 30
```

### Higher Resolution Output

```bash
DG_DEEPGEN_CONFIG=deepgen \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output \
    --height 1024 \
    --width 1024
```

## Custom Configuration

To create a custom configuration, add a new Python file in `diffgentor/models/deepgen/config/`:

```python
# diffgentor/models/deepgen/config/deepgen_large.py
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import SD3Transformer2DModel
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    trust_remote_code=True,
    padding_side="right",
)

prompt_template = dict(
    IMG_START_TOKEN="<|vision_start|>",
    IMG_END_TOKEN="<|vision_end|>",
    IMG_CONTEXT_TOKEN="<|image_pad|>",
    # ... other template fields
)

model = dict(
    num_queries=256,  # More queries
    connector=dict(
        hidden_size=4096,  # Larger connector
        intermediate_size=16384,
        num_hidden_layers=8,
        num_attention_heads=64,
        _attn_implementation="flash_attention_2",
    ),
    # ... other model config
    lora_rank=128,
    lora_alpha=256,
)
```

Then use it with:
```bash
DG_DEEPGEN_CONFIG=deepgen_large \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen ...
```

## Notes

- **VRAM Requirements**: ~24GB+ for the full model (Qwen2.5-VL-3B + SD3.5-Medium)
- **Multi-GPU**: Currently supports single GPU inference
- **Supported Formats**: Input images can be URLs or local paths
- **Output Format**: PNG images

## Troubleshooting

### Out of Memory

If you encounter OOM errors, try:
1. Reduce `--height` and `--width`
2. Use a smaller Qwen model (e.g., Qwen2.5-VL-2B)
3. Enable CPU offloading (if supported)

### Model Loading Errors

Ensure all model paths are correct and the checkpoint is compatible with the base models.

### Config Not Found

Make sure `DG_DEEPGEN_CONFIG` is set to a valid config name (without `.py` extension) that exists in `diffgentor/models/deepgen/config/`.
