# DreamOmni2 Backend

DreamOmni2 combines FLUX.1-Kontext with Qwen2.5-VL for instruction-understanding enhanced image generation and editing.

## Prerequisites

Install with DreamOmni2 dependencies:

```bash
pip install "diffgentor[dreamomni2]"
```

## Basic Usage

```bash
diffgentor edit --backend dreamomni2 \
    --model_name /path/to/dreamomni2 \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_DREAMOMNI2_VLM_PATH` | Path to VLM model (Qwen2.5-VL) for instruction enhancement | - |
| `DG_DREAMOMNI2_LORA_PATH` | Path to LoRA weights (gen_lora or edit_lora) | - |
| `DG_DREAMOMNI2_TASK_TYPE` | Task type (`generation` or `editing`) | `generation` |
| `DG_DREAMOMNI2_OUTPUT_HEIGHT` | Output image height | `1024` |
| `DG_DREAMOMNI2_OUTPUT_WIDTH` | Output image width | `1024` |

## Model Files

Base model directory:

```
/path/to/dreamomni2/
├── config.json
├── model weights (FLUX.1-Kontext base)
└── ...
```

Optional VLM for prompt enhancement:

```
/path/to/qwen2.5-vl/
├── config.json
└── model weights...
```

Optional LoRA weights:

```
/path/to/lora/
├── gen_lora/    # For generation
└── edit_lora/   # For editing
```

## Examples

### Basic Editing

```bash
DG_DREAMOMNI2_TASK_TYPE=editing \
diffgentor edit --backend dreamomni2 \
    --model_name /path/to/dreamomni2 \
    --input data.csv \
    --num_inference_steps 30 \
    --guidance_scale 3.5
```

### With VLM Enhancement

```bash
DG_DREAMOMNI2_VLM_PATH=/models/Qwen2.5-VL-7B-Instruct \
DG_DREAMOMNI2_TASK_TYPE=editing \
diffgentor edit --backend dreamomni2 \
    --model_name /path/to/dreamomni2 \
    --input data.csv
```

### With LoRA

```bash
DG_DREAMOMNI2_LORA_PATH=/models/dreamomni2-edit-lora \
DG_DREAMOMNI2_TASK_TYPE=editing \
diffgentor edit --backend dreamomni2 \
    --model_name /path/to/dreamomni2 \
    --input data.csv
```

### Generation Mode (No Input Image)

```bash
DG_DREAMOMNI2_TASK_TYPE=generation \
DG_DREAMOMNI2_OUTPUT_HEIGHT=1024 \
DG_DREAMOMNI2_OUTPUT_WIDTH=1024 \
diffgentor edit --backend dreamomni2 \
    --model_name /path/to/dreamomni2 \
    --input prompts.csv
```

### Full Configuration

```bash
DG_DREAMOMNI2_VLM_PATH=/models/Qwen2.5-VL-7B-Instruct \
DG_DREAMOMNI2_LORA_PATH=/models/dreamomni2-edit-lora \
DG_DREAMOMNI2_TASK_TYPE=editing \
DG_DREAMOMNI2_OUTPUT_HEIGHT=1024 \
DG_DREAMOMNI2_OUTPUT_WIDTH=1024 \
diffgentor edit --backend dreamomni2 \
    --model_name /path/to/dreamomni2 \
    --input data.csv \
    --num_inference_steps 30 \
    --guidance_scale 3.5 \
    --seed 42
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 30 |
| `--guidance_scale` | Guidance scale | 3.5 |
| `--seed` | Random seed | Random |

## VLM Prompt Enhancement

When `DG_DREAMOMNI2_VLM_PATH` is set, the VLM will:

1. Analyze the input image(s)
2. Understand the editing instruction in context
3. Generate an enhanced prompt for better results

This is especially useful for complex or ambiguous instructions.

## Notes

- **VRAM Requirements**: ~24GB without VLM, ~40GB+ with VLM
- **Task Type**: Use `editing` for image editing, `generation` for text-to-image
- **LoRA**: Different LoRA weights may be trained for generation vs editing tasks
- **Multi-Image**: Supports multiple input images for context
