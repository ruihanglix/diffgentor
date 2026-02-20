# Environment Variables

Diffgentor uses `DG_` prefixed environment variables for model-specific parameters. This keeps the CLI clean while allowing fine-grained control.

## Naming Convention

```
DG_{MODEL}_{PARAMETER}
```

Examples:
- `DG_STEP1X_VERSION=v1.1`
- `DG_BAGEL_CFG_TEXT_SCALE=3.0`
- `DG_EMU35_VQ_PATH=/path/to/vq`

## Step1X-Edit

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_STEP1X_VERSION` | string | `v1.1` | Model version (`v1.0` or `v1.1`) |
| `DG_STEP1X_SIZE_LEVEL` | int | `512` | Size level for image processing |
| `DG_STEP1X_OFFLOAD` | bool | `false` | Enable CPU offload |
| `DG_STEP1X_QUANTIZED` | bool | `false` | Use fp8 quantization |

```bash
DG_STEP1X_VERSION=v1.1 \
DG_STEP1X_SIZE_LEVEL=768 \
DG_STEP1X_OFFLOAD=true \
diffgentor edit --backend step1x --model_name /path/to/step1x --input data.csv
```

## BAGEL

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_BAGEL_CFG_TEXT_SCALE` | float | `3.0` | Text CFG scale |
| `DG_BAGEL_CFG_IMG_SCALE` | float | `1.5` | Image CFG scale |
| `DG_BAGEL_CFG_INTERVAL` | tuple | `0.4,1.0` | CFG interval (comma-separated) |
| `DG_BAGEL_TIMESTEP_SHIFT` | float | `3.0` | Timestep shift value |
| `DG_BAGEL_NUM_TIMESTEPS` | int | `50` | Number of denoising timesteps |
| `DG_BAGEL_THINK` | bool | `false` | Enable thinking mode |

```bash
DG_BAGEL_CFG_TEXT_SCALE=4.0 \
DG_BAGEL_CFG_IMG_SCALE=2.0 \
DG_BAGEL_CFG_INTERVAL=0.3,0.9 \
diffgentor edit --backend bagel --model_name /path/to/bagel --input data.csv
```

## Emu3.5

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_EMU35_VQ_PATH` | string | **required** | Path to VisionTokenizer model |
| `DG_EMU35_TOKENIZER_PATH` | string | model_path | Path to tokenizer |
| `DG_EMU35_CFG` | float | `3.0` | Classifier-free guidance scale |
| `DG_EMU35_MAX_NEW_TOKENS` | int | `5120` | Max new tokens to generate |
| `DG_EMU35_IMAGE_AREA` | int | `1048576` | Image area for resizing |
| `DG_EMU35_HF_DEVICE` | string | `auto` | Device for HuggingFace model |
| `DG_EMU35_VQ_DEVICE` | string | `cuda:0` | Device for VQ model |
| `DG_EMU35_VQ_TYPE` | string | `ibq` | VQ type (`ibq` or `dcae`) |

```bash
DG_EMU35_VQ_PATH=/models/emu3-vq \
DG_EMU35_CFG=5.0 \
DG_EMU35_MAX_NEW_TOKENS=10240 \
diffgentor edit --backend emu35 --model_name /path/to/emu35 --input data.csv
```

## DreamOmni2

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_DREAMOMNI2_VLM_PATH` | string | - | Path to VLM model (Qwen2.5-VL) |
| `DG_DREAMOMNI2_LORA_PATH` | string | - | Path to LoRA weights |
| `DG_DREAMOMNI2_TASK_TYPE` | string | `generation` | Task type (`generation` or `editing`) |
| `DG_DREAMOMNI2_OUTPUT_HEIGHT` | int | `1024` | Output image height |
| `DG_DREAMOMNI2_OUTPUT_WIDTH` | int | `1024` | Output image width |

```bash
DG_DREAMOMNI2_VLM_PATH=/models/Qwen2.5-VL-7B-Instruct \
DG_DREAMOMNI2_TASK_TYPE=editing \
diffgentor edit --backend dreamomni2 --model_name /path/to/dreamomni2 --input data.csv
```

## Flux Kontext Official

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_FLUX_KONTEXT_OFFLOAD` | bool | `false` | Enable model offloading |
| `DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH` | int | `512` | Max sequence length for T5 |

```bash
DG_FLUX_KONTEXT_OFFLOAD=true \
DG_FLUX_KONTEXT_MAX_SEQUENCE_LENGTH=1024 \
diffgentor edit --backend flux_kontext_official --model_name /path/to/flux --input data.csv
```

## Serve Mode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_SERVE_NUM_INFERENCE_STEPS` | int | model default | Override denoising steps for all requests |
| `DG_SERVE_GUIDANCE_SCALE` | float | model default | Override guidance scale for all requests |

Multi-GPU serving also uses backend-specific `DG_*_GPUS_PER_MODEL` variables (see each backend section) to determine how many GPUs each model replica requires. The server automatically selects an in-process pool (1 GPU/model) or subprocess pool (>1 GPU/model).

```bash
DG_SERVE_NUM_INFERENCE_STEPS=28 \
DG_SERVE_GUIDANCE_SCALE=3.5 \
diffgentor serve --mode t2i --backend diffusers --model_name black-forest-labs/FLUX.1-dev
```

See [Serve Mode](serve.md) for full documentation.

## Boolean Values

Boolean environment variables accept:
- **True**: `true`, `1`, `yes`, `on`
- **False**: `false`, `0`, `no`, `off` (or unset)

```bash
# Enable
DG_STEP1X_OFFLOAD=true
DG_STEP1X_OFFLOAD=1
DG_STEP1X_OFFLOAD=yes

# Disable
DG_STEP1X_OFFLOAD=false
DG_STEP1X_OFFLOAD=0
# or simply don't set it
```

## Tuple/List Values

Comma-separated values for tuples:

```bash
# CFG interval tuple
DG_BAGEL_CFG_INTERVAL=0.4,1.0

# Parsed as: (0.4, 1.0)
```

## Setting Multiple Variables

### Shell Export

```bash
export DG_STEP1X_VERSION=v1.1
export DG_STEP1X_SIZE_LEVEL=768
diffgentor edit --backend step1x --model_name /path/to/step1x --input data.csv
```

### Inline

```bash
DG_STEP1X_VERSION=v1.1 DG_STEP1X_SIZE_LEVEL=768 \
diffgentor edit --backend step1x --model_name /path/to/step1x --input data.csv
```

### .env File

Create a `.env` file:

```bash
# .env
DG_STEP1X_VERSION=v1.1
DG_STEP1X_SIZE_LEVEL=768
DG_EMU35_VQ_PATH=/models/emu3-vq
```

Load with:

```bash
source .env
diffgentor edit --backend step1x --model_name /path/to/step1x --input data.csv
```

Or use a tool like `dotenv`:

```bash
dotenv run diffgentor edit --backend step1x --model_name /path/to/step1x --input data.csv
```

## Debugging

Check environment variables:

```bash
# Print all DG_ variables
env | grep ^DG_

# Or in Python
python -c "import os; print({k:v for k,v in os.environ.items() if k.startswith('DG_')})"
```
