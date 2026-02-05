# Flux Kontext Official Backend

This backend uses Black Forest Labs' official Flux Kontext implementation for high-quality image editing.

## Prerequisites

Initialize the submodule:

```bash
git submodule update --init diffgentor/models/third_party/flux1
```

## Basic Usage

```bash
diffgentor edit --backend flux_kontext_official \
    --model_name /path/to/flux-kontext \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_FLUX_KONTEXT_OFFLOAD` | Enable model offloading (`true`/`false`) | `false` |
| `DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH` | Maximum sequence length for T5 | `512` |

## Model Files

The model directory should contain BFL's official flux-kontext weights:

```
/path/to/flux-kontext/
├── flux1-dev-kontext.safetensors (or similar)
├── ae.safetensors
└── other model files...
```

Note: T5 and CLIP encoders will be loaded from HuggingFace cache.

## Examples

### Basic Usage

```bash
diffgentor edit --backend flux_kontext_official \
    --model_name /path/to/flux-kontext \
    --input data.csv \
    --num_inference_steps 30 \
    --guidance_scale 2.5
```

### With CPU Offloading (Low VRAM)

```bash
DG_FLUX_KONTEXT_OFFLOAD=true \
diffgentor edit --backend flux_kontext_official \
    --model_name /path/to/flux-kontext \
    --input data.csv
```

### Longer Prompts

```bash
DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH=1024 \
diffgentor edit --backend flux_kontext_official \
    --model_name /path/to/flux-kontext \
    --input data.csv
```

### Custom Resolution

```bash
diffgentor edit --backend flux_kontext_official \
    --model_name /path/to/flux-kontext \
    --input data.csv \
    --num_inference_steps 30 \
    --guidance_scale 2.5
```

Note: Output resolution can be specified per-row in the input CSV using `height` and `width` columns.

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 30 |
| `--guidance_scale` | Guidance scale | 2.5 |
| `--seed` | Random seed | Random |

## CPU Offloading

When `DG_FLUX_KONTEXT_OFFLOAD=true`:

1. T5 and CLIP are loaded to GPU only during encoding
2. Flow model is loaded to GPU only during denoising
3. Autoencoder is loaded to GPU only during decoding

This reduces peak VRAM usage significantly at the cost of speed.

## Notes

- **VRAM Requirements**: ~40GB without offload, ~16GB with offload
- **Speed**: Offloading significantly increases generation time
- **Quality**: BFL's official implementation provides reference-quality results
- **Sequence Length**: Longer sequences allow more detailed prompts but use more memory
