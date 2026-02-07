# DeepGen T2I Backend

DeepGen is a unified visual generation model based on Qwen2.5-VL + SD3.5, supporting both text-to-image generation and image editing.

## Architecture

The model architecture consists of:
- **Qwen2.5-VL**: Language understanding module (AR model)
- **SD3.5 Transformer**: Image generation module (Diffusion model)
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

## Configuration System

DeepGen uses a config file system to manage model parameters. Config files are located in `diffgentor/models/deepgen/config/`.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_DEEPGEN_CONFIG` | Config file name | `deepgen` |
| `DG_DEEPGEN_DIFFUSION_MODEL_PATH` | Path to SD3.5 diffusion model | Required |
| `DG_DEEPGEN_AR_MODEL_PATH` | Path to Qwen2.5-VL AR model | Required |
| `DG_DEEPGEN_MAX_LENGTH` | Max sequence length | `1024` |
| `DG_DEEPGEN_GPUS_PER_MODEL` | Number of GPUs per model instance | `1` |
| `DG_DEEPGEN_DEBUG_CHECKPOINT` | Enable checkpoint loading debug log | `false` |
| `DG_DEEPGEN_IMAGE_RESIZE_MODE` | Image resize mode (for editing) | `fix_pixels` |

### Image Resize Modes (for Editing)

The `DG_DEEPGEN_IMAGE_RESIZE_MODE` environment variable controls how input images are resized:

| Mode | Description |
|------|-------------|
| `fix_pixels` | Keep total pixel count constant, align to 32 |
| `dynamic` | Keep aspect ratio, limit max edge to 512, align to 32 |
| `direct` | Force resize to exact `--height` x `--width` from CLI |

### CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_name` | Path to model checkpoint (.safetensors or .pt) | Required |
| `--guidance_scale` | CFG guidance scale | `4.0` |
| `--num_inference_steps` | Number of denoising steps | `50` |
| `--height` | Output image height | `512` |
| `--width` | Output image width | `512` |
| `--negative_prompt` | Negative prompt for CFG | `""` |
| `--seed` | Random seed | Random |

### Config File

The config file (e.g., `diffgentor/models/deepgen/config/deepgen.py`) contains model-specific parameters:

- `num_queries`: Number of query tokens
- `connector_hidden_size`: Connector hidden size
- `connector_num_layers`: Number of connector layers
- `lora_rank`: LoRA rank
- `lora_alpha`: LoRA alpha
- `prompt_template`: Prompt template for generation

## Basic Usage

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

### SD3.5 Model (Diffusion Model)

Download from HuggingFace:
```bash
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ./sd3.5-medium
```

### Qwen2.5-VL Model (AR Model)

Download from HuggingFace:
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen2.5-vl-3b
```

### Checkpoint

The checkpoint contains the trained connector and LoRA weights. It should be a `.safetensors` or `.pt` file. Pass it via `--model_name`.

## Examples

### Basic Text-to-Image

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompt "A futuristic cityscape at night" \
    --output_dir ./output
```

### Batch Generation from File

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompts_file prompts.jsonl \
    --output_dir ./output
```

### Custom CFG Scale and Steps

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompt "A detailed landscape painting" \
    --output_dir ./output \
    --guidance_scale 6.0 \
    --num_inference_steps 30
```

### Higher Resolution Output

```bash
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompt "A high-resolution portrait" \
    --output_dir ./output \
    --height 1024 \
    --width 1024
```

### Using Custom Config

```bash
DG_DEEPGEN_CONFIG=my_custom_config \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompt "A beautiful sunset" \
    --output_dir ./output
```

### Multi-GPU (8 instances on 8 GPUs)

```bash
DG_DEEPGEN_GPUS_PER_MODEL=1 \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor t2i --backend deepgen \
    --num_gpus 8 \
    --model_name /path/to/checkpoint.safetensors \
    --prompts_file prompts.jsonl \
    --output_dir ./output
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 50 |
| `--guidance_scale` | CFG guidance scale | 4.0 |
| `--negative_prompt` | Negative prompt for CFG | "" |
| `--seed` | Random seed | Random |
| `--height` | Output image height | 512 |
| `--width` | Output image width | 512 |

## Notes

- **VRAM Requirements**: ~24GB+ for the full model (Qwen2.5-VL-3B + SD3.5-Medium)
- **Multi-GPU**: Supports multiple model instances via `DG_DEEPGEN_GPUS_PER_MODEL`
- **Output Format**: PNG images
- **ViT Input Size**: Fixed at 448 (not configurable)

## Troubleshooting

### Out of Memory

If you encounter OOM errors, try:
1. Reduce `--height` and `--width` CLI parameters
2. Use a smaller Qwen model (e.g., Qwen2.5-VL-2B)
3. Enable CPU offloading (if supported)

### Model Loading Errors

Ensure all model paths are correct and the checkpoint is compatible with the base models.

### Config Not Found

If you get a config not found error, ensure:
1. `DG_DEEPGEN_CONFIG` is set to a valid config name (default: `deepgen`)
2. The config file exists in `diffgentor/models/deepgen/config/`
