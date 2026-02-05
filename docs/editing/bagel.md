# BAGEL Backend

BAGEL is ByteDance's multimodal model supporting both image understanding and generation.

## Prerequisites

Initialize the submodule:

```bash
git submodule update --init diffgentor/models/third_party/bagel
```

Install additional dependencies:

```bash
pip install -e ".[bagel]"
```

## Basic Usage

```bash
diffgentor edit --backend bagel \
    --model_name /path/to/bagel \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_BAGEL_CFG_TEXT_SCALE` | Text CFG scale | `3.0` |
| `DG_BAGEL_CFG_IMG_SCALE` | Image CFG scale | `1.5` |
| `DG_BAGEL_CFG_INTERVAL` | CFG interval (comma-separated) | `0.4,1.0` |
| `DG_BAGEL_TIMESTEP_SHIFT` | Timestep shift value | `3.0` |
| `DG_BAGEL_NUM_TIMESTEPS` | Number of denoising timesteps | `50` |
| `DG_BAGEL_THINK` | Enable thinking mode (`true`/`false`) | `false` |

## Model Files

The model directory should contain:

```
/path/to/bagel/
├── llm_config.json
├── vit_config.json
├── ae.safetensors
├── ema.safetensors
└── tokenizer files...
```

## Examples

### Basic Usage

```bash
diffgentor edit --backend bagel \
    --model_name /path/to/bagel \
    --input data.csv
```

### Custom CFG Scales

```bash
DG_BAGEL_CFG_TEXT_SCALE=4.0 \
DG_BAGEL_CFG_IMG_SCALE=2.0 \
DG_BAGEL_CFG_INTERVAL=0.3,0.9 \
diffgentor edit --backend bagel \
    --model_name /path/to/bagel \
    --input data.csv
```

### With Thinking Mode

```bash
DG_BAGEL_THINK=true \
diffgentor edit --backend bagel \
    --model_name /path/to/bagel \
    --input data.csv
```

### Longer Generation

```bash
DG_BAGEL_NUM_TIMESTEPS=100 \
DG_BAGEL_TIMESTEP_SHIFT=4.0 \
diffgentor edit --backend bagel \
    --model_name /path/to/bagel \
    --input data.csv
```

## Multi-Image Editing

BAGEL supports multi-image input for editing. Input images will be processed together:

```csv
image_url,image_url_2,instruction
/path/to/img1.jpg,/path/to/img2.jpg,"Combine these two images"
```

## Generation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_inference_steps` | Number of timesteps | 50 |
| `--seed` | Random seed | Random |

Note: `guidance_scale` is controlled via environment variables for BAGEL.

## Notes

- **VRAM Requirements**: ~30GB+ per GPU for BAGEL-7B-MoT model
- **Multi-GPU**: Uses data parallelism (torchrun) - each GPU loads a complete model and processes different data
- **Thinking Mode**: Enables extended reasoning for complex edits
