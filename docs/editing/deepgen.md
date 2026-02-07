# DeepGen Backend

DeepGen is a unified image generation model combining Qwen2.5-VL and SD3.5. It supports both text-to-image generation and instruction-based image editing.

## Features

- **Unified Model**: Single model for both T2I and image editing
- **Multi-Image Support**: Can process multiple reference images
- **Dynamic Resolution**: Supports variable input/output resolutions
- **Multi-GPU**: Distributed inference via `device_map="auto"`

## Prerequisites

DeepGen requires:

1. **SD3.5 Model**: Transformer, VAE, and scheduler components
2. **Qwen2.5-VL Model**: Language/vision encoder
3. **Checkpoint**: Fine-tuned DeepGen weights (`.safetensors` or `.pt`)

## Basic Usage

```bash
DG_DEEPGEN_DIFFUSION_PATH=/path/to/diffusion_model \
DG_DEEPGEN_QWEN_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.pt \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_DEEPGEN_DIFFUSION_PATH` | Path to diffusion model (**required**) | - |
| `DG_DEEPGEN_QWEN_PATH` | Path to Qwen2.5-VL model (**required**) | - |
| `DG_DEEPGEN_CONFIG` | Config name from `configs/` folder | `deepgen_v1` |
| `DG_DEEPGEN_GPUS_PER_MODEL` | GPUs per model instance | `0` (all visible) |
| `DG_DEEPGEN_CFG_PROMPT` | CFG prompt for unconditional | `""` |
| `DG_DEEPGEN_DEBUG` | Debug level for checkpoint loading (0-3) | `0` |

### Config-Based Architecture

Model-specific parameters (connector config, num_queries, etc.) are defined in config files located at `diffgentor/models/deepgen_v1/configs/`. The config file is selected via `DG_DEEPGEN_CONFIG`.

Each config file contains:
- `prompt_template`: Token definitions for image/text formatting
- `connector_config`: Connector architecture parameters (hidden_size, intermediate_size, layers, heads, etc.)
- `model_config`: Model parameters (num_queries, vit_input_size, max_length, etc.)

## Model Files

### Diffusion Model Structure

```
/path/to/diffusion_model/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── scheduler/
    └── scheduler_config.json
```

### Qwen2.5-VL Model Structure

```
/path/to/qwen2.5-vl/
├── config.json
├── model-*.safetensors
├── tokenizer.json
└── ...
```

### DeepGen Checkpoint

The checkpoint file (`.pt` or `.safetensors`) contains:
- Connector weights
- Projector weights
- Meta queries
- LoRA weights (if applicable)

## Examples

### Image Editing

```bash
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
DG_DEEPGEN_CONFIG=deepgen_v1 \
diffgentor edit --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --input data.csv \
    --guidance_scale 4.0 \
    --num_inference_steps 50
```

### Text-to-Image Generation

```bash
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
DG_DEEPGEN_CONFIG=deepgen_v1 \
diffgentor t2i --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --prompt "A cat sitting on a windowsill" \
    --height 1024 \
    --width 1024
```

### Multi-GPU Inference

```bash
# Use 2 GPUs per model instance
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DG_DEEPGEN_GPUS_PER_MODEL=2 \
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
DG_DEEPGEN_CONFIG=deepgen_v1 \
diffgentor edit --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --input data.csv
```

### Custom Config

To use a different model configuration, create a new config file in `diffgentor/models/deepgen_v1/configs/` and specify it via `DG_DEEPGEN_CONFIG`:

```bash
# Use custom config file: diffgentor/models/deepgen_v1/configs/my_custom_config.py
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
DG_DEEPGEN_CONFIG=my_custom_config \
diffgentor edit --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --input data.csv
```

## CLI Parameters

Common parameters passed via CLI:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--guidance_scale` | Classifier-free guidance scale | `4.0` |
| `--num_inference_steps` | Number of denoising steps | `50` |
| `--seed` | Random seed for reproducibility | Random |
| `--batch_size` | Batch size for inference | `1` |

## Architecture

DeepGen consists of three main components:

1. **Qwen2.5-VL**: Processes text and images, outputs hidden states
2. **Connector**: Transforms LLM hidden states to DiT-compatible embeddings
3. **SD3.5 Transformer**: Generates images via diffusion

```
Text/Image Input
      │
      ▼
┌─────────────┐
│ Qwen2.5-VL  │  (Language/Vision Encoder)
└─────────────┘
      │
      ▼ (Hidden States from multiple layers)
┌─────────────┐
│  Connector  │  (6-layer Transformer)
└─────────────┘
      │
      ▼ (Pooled + Sequence Embeddings)
┌─────────────┐
│    SD3.5    │  (Diffusion Transformer)
└─────────────┘
      │
      ▼
  Output Image
```

## Notes

- **VRAM Requirements**: ~24GB+ for 7B Qwen model with SD3.5
- **Flash Attention**: Recommended for better performance (set in config file's `connector_config._attn_implementation`)
- **Checkpoint Format**: Supports both `.safetensors` and `.pt` formats
- **Dynamic Resolution**: Input images are automatically resized while maintaining aspect ratio
- **Config Files**: Model architecture is defined in config files, not environment variables

## Debug Mode

Set `DG_DEEPGEN_DEBUG` to enable checkpoint loading debug report. This helps verify that all weights (including LoRA) are loaded correctly.

### Debug Levels

| Level | Output |
|-------|--------|
| `0` | Off (default) |
| `1` | Load summary + LoRA check + zero-weight warnings |
| `2` | + Full missing/unexpected keys list + LoRA details + key weight stats |
| `3` | + All loaded keys + all weight statistics |

### Usage

```bash
# Basic debug (summary only)
DG_DEEPGEN_DEBUG=1 \
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
diffgentor edit --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --input data.csv

# Detailed debug (recommended for troubleshooting)
DG_DEEPGEN_DEBUG=2 \
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
diffgentor edit --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --input data.csv
```

### Debug Report Location

The debug report is written to `{log_dir}/deepgen_checkpoint_debug.log`.

Only rank 0 writes the debug report in distributed environments.

### Example Debug Report

```
================================================================================
DeepGen Checkpoint Load Debug Report
================================================================================
Time: 2026-02-06 14:30:25
Checkpoint: /path/to/checkpoint.pt
Debug Level: 2

=== Basic Info ===
Checkpoint file size: 1.20 GB
Checkpoint keys: 156
Model trainable keys: 180

=== Load Summary ===
Successfully loaded: 150 keys
Missing keys: 30
Unexpected keys: 6

=== LoRA Weights ===
LoRA modules in checkpoint: 24
LoRA modules loaded: 24
All LoRA weights loaded: YES

=== Weight Statistics ===
projector_1.weight:
  shape: (12288, 2048), dtype: torch.bfloat16
  mean: 0.001234, std: 0.023456
  min: -0.123456, max: 0.115678
  non-zero: 100.0% (25165824/25165824)

meta_queries:
  shape: (128, 3584), dtype: torch.bfloat16
  mean: 0.000123, std: 0.016789
  min: -0.089012, max: 0.092345
  non-zero: 100.0% (458752/458752)

=== Verification Result ===
[PASS] All checked weights are non-zero
[PASS] All LoRA weights loaded successfully
[WARN] 30 missing key(s) (may be expected for frozen components)
================================================================================
```
