# Image Editing

Diffgentor supports instruction-based image editing with multiple backends and models.

## Supported Backends

| Backend | Model | Multi-Image | Description |
|---------|-------|-------------|-------------|
| `diffusers` | Qwen-Image-Edit-* | ✓ | Qwen multimodal editing model |
| `diffusers` | Qwen-Image-Edit | ✗ | Qwen single-image editing |
| `diffusers` | FLUX.2-dev | ✗ | FLUX 2.0 editing model |
| `diffusers` | FLUX.1-Kontext | ✗ | FLUX Kontext editing |
| `diffusers` | LongCat | ✗ | LongCat image editing |
| `openai` | gpt-image-1 | ✓ | OpenAI GPT-Image API |
| `google_genai` | gemini-2.5-flash-image | ✓ | Google Gemini native image |
| `step1x` | Step1X-Edit | ✗ | Step1X-Edit v1.0/v1.1 |
| `bagel` | BAGEL | ✓ | ByteDance BAGEL |
| `emu35` | Emu3.5 | ✓ | BAAI Emu3.5 autoregressive |
| `dreamomni2` | DreamOmni2 | ✓ | FLUX-Kontext + Qwen2.5-VL |
| `flux_kontext_official` | Flux-Kontext | ✓ | BFL official implementation |
| `hunyuan_image_3` | HunyuanImage-3.0 | ✓ | Tencent HunyuanImage-3.0 with CoT |
| `deepgen` | DeepGen | ✓ | Qwen2.5-VL + SD3.5 unified model |

## Quick Start

```bash
# Basic editing with Qwen-Image-Edit
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --output_dir ./edited

# With OpenAI
export OPENAI_API_KEY=your_key
diffgentor edit --backend openai \
    --model_name gpt-image-1 \
    --input data.csv

# With Google Gemini
export GEMINI_API_KEY=your_key
diffgentor edit --backend google_genai \
    --model_name gemini-2.5-flash-image \
    --input data.csv
```

## Input Data Format

### CSV Format

```csv
image_url,instruction
https://example.com/cat.jpg,"Make the cat wear a hat"
/local/path/dog.png,"Change the background to a beach"
```

### Parquet Directory

Place `.parquet` files in a directory:

```bash
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input /path/to/parquet_dir
```

Required columns:
- `image_url` or `image`: Image URL or local path
- `instruction`: Editing instruction (configurable via `--instruction_key`)

## CLI Options

### Input/Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to input CSV or Parquet directory (required) | - |
| `--output_dir` | Directory to save edited images | `./output` |
| `--output_csv` | Path to output CSV file | `output_dir/results.csv` |
| `--output_name_column` | Column name for custom output filename | - |
| `--instruction_key` | Column name for instruction | `instruction` |
| `--image_cache_dir` | Directory to cache downloaded images | - |

### Editing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num_inference_steps` | Number of denoising steps | 40 |
| `--guidance_scale` | Guidance scale | 4.0 |
| `--true_cfg_scale` | True CFG scale (Qwen models) | 4.0 |
| `--negative_prompt` | Negative prompt | " " |
| `--batch_size` | Batch size | 1 |

### Execution Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max_retries` | Max retries for failed edits | 3 |
| `--filter_rows` | Filter rows by index (e.g., `0:100`, `0,5,10`) | - |
| `--resume` / `--no_resume` | Resume from previous progress | True |

## Backend-Specific Guides

- [Diffusers Models](diffusers.md) - Qwen, FLUX, LongCat
- [Step1X-Edit](step1x.md) - Step1X-Edit v1.0/v1.1
- [BAGEL](bagel.md) - ByteDance BAGEL
- [Emu3.5](emu35.md) - BAAI Emu3.5
- [DreamOmni2](dreamomni2.md) - DreamOmni2
- [Flux Kontext Official](flux_kontext.md) - BFL official
- [HunyuanImage-3.0](hunyuan_image_3.md) - Tencent HunyuanImage-3.0
- [DeepGen](deepgen.md) - Qwen2.5-VL + SD3.5 unified model
- [OpenAI](openai.md) - GPT-Image API

## Additional Features

- [Prompt Enhancement](../prompt_enhance.md) - LLM-based prompt enhancement

## Examples

### Basic Usage

```bash
# Edit with Qwen-Image-Edit-2511
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --model_type qwen \
    --input data.csv \
    --num_inference_steps 40 \
    --guidance_scale 1.0 \
    --true_cfg_scale 4.0
```

### Process Specific Rows

```bash
# Process first 100 rows
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --filter_rows 0:100

# Process specific indices
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --filter_rows 0,10,20,30
```

### With Third-Party Models

```bash
# Step1X-Edit
DG_STEP1X_VERSION=v1.1 DG_STEP1X_SIZE_LEVEL=512 \
diffgentor edit --backend step1x \
    --model_name /path/to/step1x \
    --input data.csv

# BAGEL
DG_BAGEL_CFG_TEXT_SCALE=3.0 DG_BAGEL_CFG_IMG_SCALE=1.5 \
diffgentor edit --backend bagel \
    --model_name /path/to/bagel \
    --input data.csv

# Emu3.5
DG_EMU35_VQ_PATH=/path/to/vq_model \
diffgentor edit --backend emu35 \
    --model_name /path/to/emu35 \
    --input data.csv

# HunyuanImage-3.0
diffgentor edit --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --input data.csv \
    --num_inference_steps 8

# DeepGen
DG_DEEPGEN_CONFIG=deepgen \
DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5 \
DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl \
diffgentor edit --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --input data.csv \
    --guidance_scale 4.0
```

## Output

The editing process generates:

1. **Edited Images**: Saved to `output_dir/`
2. **Results CSV**: Contains original data plus `output_image` column

### Default Output Naming

By default, output files are named using zero-padded indices:

- Single image: `{index:06d}.png` (e.g., `000000.png`, `000001.png`)
- Multiple images per prompt: `{index:06d}_{sub_index:02d}.png` (e.g., `000000_00.png`, `000000_01.png`)

### Custom Output Naming

Use `--output_name_column` to specify a column in your CSV/Parquet that contains custom output paths:

```bash
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --output_dir ./results \
    --output_name_column output_path
```

**Input CSV example:**

```csv
index,input_images,instruction,output_path
0,cat.jpg,"Add a hat",animals/cat_with_hat
1,dog.jpg,"Change background",animals/dog_beach.jpg
2,bird.png,"Make it fly",birds/flying/001.png
```

**Output behavior:**

| Column Value | Output File |
|--------------|-------------|
| `aaa/bb/1` | `results/aaa/bb/1.png` |
| `aaa/bb/1.png` | `results/aaa/bb/1.png` |
| `aaa/bb/1.jpg` | `results/aaa/bb/1.jpg` |
| `aaa/bb/1.jpeg` | `results/aaa/bb/1.jpeg` |

**Features:**
- Supported formats: `.png`, `.jpg`, `.jpeg` (other extensions default to `.png`)
- Parent directories are automatically created
- Resume support: checks for existing files with custom names
- For multiple images per prompt, sub-index is appended: `aaa/bb/1_00.png`, `aaa/bb/1_01.png`

**Results CSV:**

```csv
index,input_images,instruction,output_path,output_image
0,cat.jpg,"Add a hat",animals/cat_with_hat,animals/cat_with_hat.png
1,dog.jpg,"Change background",animals/dog_beach.jpg,animals/dog_beach.jpg
```
