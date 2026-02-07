# DeepGen T2I Backend

DeepGen is a unified visual generation model based on Qwen2.5-VL + SD3.5, supporting both text-to-image generation and image editing.

## Architecture

The model architecture consists of:
- **Qwen2.5-VL**: Language understanding module
- **SD3.5 Transformer**: Image generation module
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

## Basic Usage

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompt "A beautiful sunset over the ocean" \
    --output_dir ./output
```

## CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--guidance_scale` | CFG guidance scale | `4.0` |
| `--num_inference_steps` | Number of denoising steps | `50` |
| `--height` | Output image height | `512` |
| `--width` | Output image width | `512` |
| `--negative_prompt` | Negative prompt for CFG | `""` |
| `--seed` | Random seed | Random |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_DEEPGEN_SD3_MODEL_PATH` | Path to SD3.5 model | Required |
| `DG_DEEPGEN_QWEN_MODEL_PATH` | Path to Qwen2.5-VL model | Required |
| `DG_DEEPGEN_CHECKPOINT` | Path to model checkpoint | None |
| `DG_DEEPGEN_CFG_PROMPT` | CFG negative prompt | `""` |
| `DG_DEEPGEN_HEIGHT` | Default output image height | `512` |
| `DG_DEEPGEN_WIDTH` | Default output image width | `512` |
| `DG_DEEPGEN_NUM_STEPS` | Default number of inference steps | `50` |
| `DG_DEEPGEN_NUM_QUERIES` | Number of query tokens | `128` |
| `DG_DEEPGEN_CONNECTOR_HIDDEN_SIZE` | Connector hidden size | `2048` |
| `DG_DEEPGEN_CONNECTOR_NUM_LAYERS` | Number of connector layers | `6` |
| `DG_DEEPGEN_VIT_INPUT_SIZE` | ViT input size | `448` |
| `DG_DEEPGEN_LORA_RANK` | LoRA rank | `64` |

## Model Files

The model requires two base models and an optional checkpoint:

### SD3.5 Model

Download from HuggingFace:
```bash
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ./sd3.5-medium
```

### Qwen2.5-VL Model

Download from HuggingFace:
```bash
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen2.5-vl-3b
```

### Checkpoint

The checkpoint contains the trained connector and LoRA weights. It should be a `.safetensors` or `.pt` file.

## Examples

### Basic Text-to-Image

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompt "A futuristic cityscape at night" \
    --output_dir ./output
```

### Batch Generation from File

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompts_file prompts.jsonl \
    --output_dir ./output
```

### Custom CFG Scale and Steps

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompt "A detailed landscape painting" \
    --output_dir ./output \
    --guidance_scale 6.0 \
    --num_inference_steps 30
```

### Higher Resolution Output

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompt "A high-resolution portrait" \
    --output_dir ./output \
    --height 1024 \
    --width 1024
```

### Alternative Model Path Specification

You can also specify model paths via `--model_name`:

```bash
diffgentor t2i --backend deepgen \
    --model_name "/path/to/sd3.5,/path/to/qwen2.5-vl" \
    --prompt "A beautiful sunset" \
    --output_dir ./output
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 50 (or `DG_DEEPGEN_NUM_STEPS`) |
| `--guidance_scale` | CFG guidance scale | 4.0 |
| `--negative_prompt` | Negative prompt for CFG | "" |
| `--seed` | Random seed | Random |
| `--height` | Output image height | 512 (or `DG_DEEPGEN_HEIGHT`) |
| `--width` | Output image width | 512 (or `DG_DEEPGEN_WIDTH`) |

## Notes

- **VRAM Requirements**: ~24GB+ for the full model (Qwen2.5-VL-3B + SD3.5-Medium)
- **Multi-GPU**: Currently supports single GPU inference
- **Output Format**: PNG images

## Troubleshooting

### Out of Memory

If you encounter OOM errors, try:
1. Reduce `DG_DEEPGEN_HEIGHT` and `DG_DEEPGEN_WIDTH`
2. Use a smaller Qwen model (e.g., Qwen2.5-VL-2B)
3. Enable CPU offloading (if supported)

### Model Loading Errors

Ensure all model paths are correct and the checkpoint is compatible with the base models.
