# Step1X-Edit Backend

Step1X-Edit is an image editing model from UnifyModel. Diffgentor supports both v1.0 and v1.1 versions.

## Prerequisites

Initialize the submodule:

```bash
git submodule update --init diffgentor/models/third_party/step1x_edit
```

## Basic Usage

```bash
diffgentor edit --backend step1x \
    --model_name /path/to/step1x-edit \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_STEP1X_VERSION` | Model version (`v1.0` or `v1.1`) | `v1.1` |
| `DG_STEP1X_SIZE_LEVEL` | Size level for image processing | `512` |
| `DG_STEP1X_OFFLOAD` | Enable CPU offload (`true`/`false`) | `false` |
| `DG_STEP1X_QUANTIZED` | Use fp8 quantization (`true`/`false`) | `false` |

## Model Files

The model directory should contain:

```
/path/to/step1x-edit/
├── vae.safetensors
├── step1x-edit-i1258.safetensors      # v1.0
├── step1x-edit-v1p1-official.safetensors  # v1.1
└── Qwen2.5-VL-7B-Instruct/
    ├── config.json
    └── ...
```

## Examples

### Basic v1.1

```bash
DG_STEP1X_VERSION=v1.1 \
diffgentor edit --backend step1x \
    --model_name /path/to/step1x-edit \
    --input data.csv \
    --num_inference_steps 28 \
    --guidance_scale 6.0
```

### v1.0 with Offloading

```bash
DG_STEP1X_VERSION=v1.0 \
DG_STEP1X_OFFLOAD=true \
diffgentor edit --backend step1x \
    --model_name /path/to/step1x-edit \
    --input data.csv
```

### With FP8 Quantization

```bash
DG_STEP1X_VERSION=v1.1 \
DG_STEP1X_QUANTIZED=true \
DG_STEP1X_SIZE_LEVEL=768 \
diffgentor edit --backend step1x \
    --model_name /path/to/step1x-edit \
    --input data.csv
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 28 |
| `--guidance_scale` | CFG guidance scale | 6.0 |
| `--negative_prompt` | Negative prompt | "" |
| `--seed` | Random seed | Random |

## Notes

- **VRAM Requirements**: ~24GB for standard inference, less with offloading
- **Size Level**: Higher values produce larger output images but require more VRAM
- **v1.1 vs v1.0**: v1.1 generally produces higher quality results
