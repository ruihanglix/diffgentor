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

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_DEEPGEN_CONFIG` | Config file name | `deepgen` |
| `DG_DEEPGEN_DIFFUSION_MODEL_PATH` | Diffusion model path (SD3.5) | **Required** |
| `DG_DEEPGEN_AR_MODEL_PATH` | AR model path (Qwen2.5-VL) | **Required** |
| `DG_DEEPGEN_MAX_LENGTH` | Maximum sequence length | `1024` |
| `DG_DEEPGEN_GPUS_PER_MODEL` | Number of GPUs per model instance | `1` |
| `DG_DEEPGEN_DEBUG_CHECKPOINT` | Enable checkpoint loading debug log | `false` |
| `DG_DEEPGEN_IMAGE_RESIZE_MODE` | Image resize mode (see below) | `fix_pixels` |

### Image Resize Modes

The `DG_DEEPGEN_IMAGE_RESIZE_MODE` environment variable controls how input images are resized:

| Mode | Description | Use Case |
|------|-------------|----------|
| `fix_pixels` | Keep total pixel count constant (`ratio = 512 / sqrt(h*w)`), align to 32 | Best for maintaining image quality |
| `dynamic` | Keep aspect ratio, limit max edge to 512, align to 32 | Best for preserving aspect ratio |
| `direct` | Force resize to exact `--height` x `--width` from CLI | When exact output size is required |

**Note**: For `direct` mode, you must specify `--height` and `--width` CLI arguments.

### CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Path to model checkpoint (.safetensors or .pt) | **Required** |
| `--guidance_scale` | CFG guidance scale | `4.0` |
| `--num_inference_steps` | Number of denoising steps | `50` |
| `--height` | Output image height | Auto (based on resize mode) |
| `--width` | Output image width | Auto (based on resize mode) |
| `--negative_prompt` | Negative prompt for CFG | (descriptive default) |
| `--seed` | Random seed | Random |

## Basic Usage

### Image Editing

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output
```

### Text-to-Image Generation

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompt "A beautiful sunset over the ocean" \
    --output_dir ./output
```

## Model Files

The model requires two base models and a checkpoint:

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

The checkpoint contains the trained connector and LoRA weights. It should be a `.safetensors` or `.pt` file. Pass it via `--model_name`.

## Examples

### Basic Image Editing

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output
```

### Custom CFG Scale

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output \
    --guidance_scale 6.0 \
    --num_inference_steps 30
```

### Fixed Output Size (Direct Resize Mode)

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_IMAGE_RESIZE_MODE=direct \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output \
    --height 512 \
    --width 512
```

### Preserve Aspect Ratio (Dynamic Resize Mode)

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_IMAGE_RESIZE_MODE=dynamic \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output
```

### Multi-GPU (8 instances on 8 GPUs)

```bash
DG_DEEPGEN_GPUS_PER_MODEL=1 \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --num_gpus 8 \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output
```

### Debug Checkpoint Loading

```bash
DG_DEEPGEN_DEBUG_CHECKPOINT=1 \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --output_dir ./output \
    --log_dir ./logs
```

This will write detailed checkpoint loading info to `./logs/checkpoint_debug.log`.

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
- **Multi-GPU**: Supports multiple model instances via `DG_DEEPGEN_GPUS_PER_MODEL`
- **Supported Formats**: Input images can be URLs or local paths
- **Output Format**: PNG images
- **ViT Input Size**: Fixed at 448 (not configurable)

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
