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
| `DG_DEEPGEN_GPUS_PER_MODEL` | GPUs per model instance | `0` (all visible) |
| `DG_DEEPGEN_CFG_PROMPT` | CFG prompt for unconditional | `""` |
| `DG_DEEPGEN_NUM_QUERIES` | Number of connector queries | `128` |
| `DG_DEEPGEN_MAX_LENGTH` | Maximum sequence length | `1024` |
| `DG_DEEPGEN_VIT_INPUT_SIZE` | Vision encoder input size | `448` |
| `DG_DEEPGEN_CONNECTOR_HIDDEN_SIZE` | Connector hidden dimension | `2048` |
| `DG_DEEPGEN_CONNECTOR_LAYERS` | Number of connector layers | `6` |
| `DG_DEEPGEN_CONNECTOR_HEADS` | Connector attention heads | `32` |
| `DG_DEEPGEN_ATTN_IMPL` | Attention implementation | `flash_attention_2` |

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
diffgentor edit --backend deepgen \
    --model_name /checkpoints/deepgen.pt \
    --input data.csv
```

### Custom Connector Configuration

```bash
DG_DEEPGEN_DIFFUSION_PATH=/models/UniPic2-SD3.5M-Kontext-2B \
DG_DEEPGEN_QWEN_PATH=/models/Qwen2.5-VL-3B-Instruct \
DG_DEEPGEN_NUM_QUERIES=256 \
DG_DEEPGEN_CONNECTOR_LAYERS=8 \
DG_DEEPGEN_CONNECTOR_HEADS=64 \
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
- **Flash Attention**: Recommended for better performance (`DG_DEEPGEN_ATTN_IMPL=flash_attention_2`)
- **Checkpoint Format**: Supports both `.safetensors` and `.pt` formats
- **Dynamic Resolution**: Input images are automatically resized while maintaining aspect ratio
