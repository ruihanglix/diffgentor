# HunyuanImage-3.0-Instruct Backend

HunyuanImage-3.0 is Tencent's native multimodal model that unifies multimodal understanding and generation within an autoregressive framework. It features 80B total parameters (13B activated per token) with 64 MoE experts.

## Features

- **Image Editing (TI2I)**: Text-Image-to-Image generation
- **Multi-Image Fusion**: Combine up to 3 input images
- **CoT Reasoning**: Chain-of-thought reasoning for better understanding
- **Prompt Self-Rewrite**: Automatic prompt enhancement and expansion
- **Distilled Version**: 8-step inference for faster generation

## Prerequisites

Download model weights from HuggingFace:

```bash
# Distilled version (recommended, 8-step inference)
hf download tencent/HunyuanImage-3.0-Instruct-Distil --local-dir ./HunyuanImage-3-Instruct-Distil

# Full version (50-step inference)
hf download tencent/HunyuanImage-3.0-Instruct --local-dir ./HunyuanImage-3-Instruct
```

> **Note**: The directory name should not contain dots (rename from `HunyuanImage-3.0-Instruct-Distil` to `HunyuanImage-3-Instruct-Distil`) to avoid issues with transformers loading.

### Optional: FlashInfer for 3x Faster Inference

```bash
pip install flashinfer-python==0.5.0
```

## Basic Usage

```bash
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --input data.csv \
    --output_dir ./output \
    --num_inference_steps 8
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL` | Number of GPUs per model instance (0=all visible) | `0` |
| `DG_HUNYUAN_IMAGE_3_ATTN_IMPL` | Attention implementation | `sdpa` |
| `DG_HUNYUAN_IMAGE_3_MOE_IMPL` | MoE implementation (`eager` or `flashinfer`) | `eager` |
| `DG_HUNYUAN_IMAGE_3_MOE_DROP_TOKENS` | Enable MoE token dropping | `true` |
| `DG_HUNYUAN_IMAGE_3_USE_SYSTEM_PROMPT` | System prompt type | `en_unified` |
| `DG_HUNYUAN_IMAGE_3_BOT_TASK` | Task type | `think_recaption` |
| `DG_HUNYUAN_IMAGE_3_INFER_ALIGN_IMAGE_SIZE` | Align output size to input | `true` |
| `DG_HUNYUAN_IMAGE_3_MAX_NEW_TOKENS` | Maximum new tokens for text generation | `2048` |
| `DG_HUNYUAN_IMAGE_3_USE_TAYLOR_CACHE` | Use Taylor Cache when sampling | `false` |

### System Prompt Types

| Value | Description |
|-------|-------------|
| `None` | No system prompt |
| `dynamic` | Dynamic selection |
| `en_vanilla` | Basic English prompt |
| `en_recaption` | Prompt with recaption |
| `en_think_recaption` | Think + recaption |
| `en_unified` | Unified prompt (recommended) |
| `custom` | Custom system prompt |

### Task Types

| Value | Description |
|-------|-------------|
| `image` | Direct image generation |
| `auto` | Automatic text output |
| `recaption` | Rewrite prompt → generate image |
| `think_recaption` | Think → rewrite → generate image (recommended) |

## Examples

### Basic Editing (Distilled Model)

```bash
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --input data.csv \
    --num_inference_steps 8
```

### Full Model with More Steps

```bash
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct \
    --input data.csv \
    --num_inference_steps 50
```

### With FlashInfer Acceleration

```bash
DG_HUNYUAN_IMAGE_3_MOE_IMPL=flashinfer \
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --input data.csv \
    --num_inference_steps 8
```

### Direct Image Generation (No CoT)

```bash
DG_HUNYUAN_IMAGE_3_BOT_TASK=image \
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --input data.csv
```

## Multi-GPU Setup

The model requires significant GPU memory (≥8×80GB recommended). You can configure GPU distribution:

### Single Instance on All GPUs

```bash
# Use all 8 GPUs for one model instance
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --input data.csv
```

### Multiple Instances (Tensor Parallelism)

```bash
# 2 model instances, each on 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL=4 \
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --model_type hunyuan_image_3 \
    --input data.csv
# Instance 0: GPU 0,1,2,3
# Instance 1: GPU 4,5,6,7
```

### 4 Instances on 8 GPUs

```bash
# 4 model instances, each on 2 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL=2 \
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --model_type hunyuan_image_3 \
    --input data.csv
```

> **Note**: When using `DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL`, you must also specify `--model_type hunyuan_image_3` for the Launcher to correctly detect the launch strategy.

## Multi-Image Input

HunyuanImage-3.0 supports up to 3 input images for multi-image fusion tasks. In your CSV, provide comma-separated image paths:

```csv
image_url,instruction
"img1.png,img2.png","Based on the logo in image 1 and the style of image 2, create a new design"
"photo1.jpg,photo2.jpg,photo3.jpg","Combine elements from all three images"
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of diffusion steps | `8` (distilled) / `50` (full) |
| `--seed` | Random seed for reproducibility | `42` |

## Notes

- **VRAM Requirements**: ≥8×80GB recommended for full model
- **First Run Slower**: When FlashInfer is enabled, the first inference may take ~10 minutes due to kernel compilation. Subsequent runs will be much faster.
- **Model Naming**: Avoid dots in directory names when loading models
- **Autoregressive**: The model uses an autoregressive framework with diffusion decoder
- **CoT Reasoning**: The `think_recaption` mode enables chain-of-thought reasoning for better understanding of complex instructions

## Troubleshooting

### Out of Memory

- Reduce the number of GPUs per instance or use more GPUs
- Try the distilled model with fewer inference steps
- Enable `moe_drop_tokens` (default: true)

### Slow First Inference with FlashInfer

This is expected behavior. FlashInfer compiles kernels on first run, which takes ~10 minutes. Subsequent runs will be significantly faster.

### Model Loading Issues

Ensure the model directory name does not contain dots. Rename:
- `HunyuanImage-3.0-Instruct-Distil` → `HunyuanImage-3-Instruct-Distil`

## References

- [HuggingFace Model Card](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct-Distil)
- [GitHub Repository](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)
- [Technical Report](https://arxiv.org/pdf/2509.23951)
