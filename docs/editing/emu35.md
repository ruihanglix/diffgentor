# Emu3.5 Backend

Emu3.5 is BAAI's autoregressive multimodal model supporting image generation and editing through next-token prediction.

## Prerequisites

Initialize the submodule:

```bash
git submodule update --init diffgentor/models/third_party/emu35
```

**Required**: VisionTokenizer (VQ) model must be downloaded separately.

## Basic Usage

```bash
DG_EMU35_VQ_PATH=/path/to/vq_model \
diffgentor edit --backend emu35 \
    --model_name /path/to/emu35 \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_EMU35_VQ_PATH` | Path to VisionTokenizer model (**required**) | - |
| `DG_EMU35_TOKENIZER_PATH` | Path to tokenizer | Model path |
| `DG_EMU35_CFG` | Classifier-free guidance scale | `3.0` |
| `DG_EMU35_MAX_NEW_TOKENS` | Maximum new tokens to generate | `5120` |
| `DG_EMU35_IMAGE_AREA` | Image area for resizing (pixels) | `1048576` |
| `DG_EMU35_HF_DEVICE` | Device for HuggingFace model | `auto` |
| `DG_EMU35_VQ_DEVICE` | Device for VQ model | `cuda:0` |
| `DG_EMU35_VQ_TYPE` | VQ type (`ibq` or `dcae`) | `ibq` |

## Model Files

Required files:

```
/path/to/emu35/
├── config.json
├── model weights...
└── tokenizer files...

/path/to/vq_model/
├── config.json
└── model weights...
```

## Examples

### Basic Usage

```bash
DG_EMU35_VQ_PATH=/models/emu3-vq \
diffgentor edit --backend emu35 \
    --model_name /models/emu35 \
    --input data.csv
```

### Custom CFG and Token Limit

```bash
DG_EMU35_VQ_PATH=/models/emu3-vq \
DG_EMU35_CFG=5.0 \
DG_EMU35_MAX_NEW_TOKENS=10240 \
diffgentor edit --backend emu35 \
    --model_name /models/emu35 \
    --input data.csv
```

### Multi-GPU Setup

```bash
DG_EMU35_VQ_PATH=/models/emu3-vq \
DG_EMU35_HF_DEVICE=auto \
DG_EMU35_VQ_DEVICE=cuda:0 \
diffgentor edit --backend emu35 \
    --model_name /models/emu35 \
    --input data.csv
```

### With DCAE VQ Type

```bash
DG_EMU35_VQ_PATH=/models/emu3-dcae \
DG_EMU35_VQ_TYPE=dcae \
diffgentor edit --backend emu35 \
    --model_name /models/emu35 \
    --input data.csv
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--seed` | Random seed | Random |

Note: Emu3.5 uses autoregressive generation, so `num_inference_steps` and `guidance_scale` are not applicable. Use environment variables for CFG control.

## Notes

- **VRAM Requirements**: ~40GB+ depending on model size and max tokens
- **Autoregressive**: Unlike diffusion models, Emu3.5 generates tokens sequentially
- **VQ Model**: The VisionTokenizer is separate and must be downloaded
- **Image Area**: Controls the maximum input image resolution (width × height)
