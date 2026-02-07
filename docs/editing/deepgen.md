# DeepGen Backend

DeepGen is a unified visual generation model based on Qwen2.5-VL + SD3.5, supporting both text-to-image generation and image editing.

## Architecture

The model architecture consists of:
- **Qwen2.5-VL**: Language/vision understanding module
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

### Image Editing

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output
```

### Text-to-Image Generation

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor t2i --backend deepgen \
    --model_name deepgen \
    --prompt "A beautiful sunset over the ocean" \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_DEEPGEN_SD3_MODEL_PATH` | Path to SD3.5 model | Required |
| `DG_DEEPGEN_QWEN_MODEL_PATH` | Path to Qwen2.5-VL model | Required |
| `DG_DEEPGEN_CHECKPOINT` | Path to model checkpoint | None |
| `DG_DEEPGEN_CFG_SCALE` | CFG guidance scale | `4.0` |
| `DG_DEEPGEN_CFG_PROMPT` | CFG negative prompt | `""` |
| `DG_DEEPGEN_HEIGHT` | Output image height | `512` |
| `DG_DEEPGEN_WIDTH` | Output image width | `512` |
| `DG_DEEPGEN_NUM_STEPS` | Number of inference steps | `50` |
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

### Basic Image Editing

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output
```

### Custom CFG Scale

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
DG_DEEPGEN_CFG_SCALE=6.0 \
DG_DEEPGEN_NUM_STEPS=30 \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output
```

### Higher Resolution Output

```bash
DG_DEEPGEN_SD3_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_QWEN_MODEL_PATH=/path/to/qwen2.5-vl \
DG_DEEPGEN_CHECKPOINT=/path/to/checkpoint.safetensors \
DG_DEEPGEN_HEIGHT=1024 \
DG_DEEPGEN_WIDTH=1024 \
diffgentor edit --backend deepgen \
    --model_name deepgen \
    --input data.csv \
    --output_dir ./output
```

### Alternative Model Path Specification

You can also specify model paths via `--model_name`:

```bash
diffgentor edit --backend deepgen \
    --model_name "/path/to/sd3.5,/path/to/qwen2.5-vl" \
    --input data.csv \
    --output_dir ./output
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 50 |
| `--guidance_scale` | CFG guidance scale | 4.0 |
| `--seed` | Random seed | Random |

## Notes

- **VRAM Requirements**: ~24GB+ for the full model (Qwen2.5-VL-3B + SD3.5-Medium)
- **Multi-GPU**: Currently supports single GPU inference
- **Supported Formats**: Input images can be URLs or local paths
- **Output Format**: PNG images

## Troubleshooting

### Out of Memory

If you encounter OOM errors, try:
1. Reduce `DG_DEEPGEN_HEIGHT` and `DG_DEEPGEN_WIDTH`
2. Use a smaller Qwen model (e.g., Qwen2.5-VL-2B)
3. Enable CPU offloading (if supported)

### Model Loading Errors

Ensure all model paths are correct and the checkpoint is compatible with the base models.
