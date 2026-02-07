# Text-to-Image Generation

Diffgentor supports text-to-image generation with multiple backends and optimization strategies.

## Supported Backends

### 1. Diffusers Backend

The default backend using HuggingFace diffusers with automatic pipeline detection.

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A futuristic city at sunset" \
    --output_dir ./output
```

**Supported Models:**
- FLUX.1-dev, FLUX.1-schnell
- Stable Diffusion 3 (SD3)
- Stable Diffusion XL (SDXL)
- Stable Diffusion 1.5/2.1
- Any diffusers-compatible model

### 2. xDiT Backend (Multi-GPU)

High-performance multi-GPU inference using xDiT parallelism strategies.

**Note:** xDiT parallelism parameters are configured via `DG_XDIT_*` environment variables.

```bash
# Set parallelism via environment variables
export DG_XDIT_ULYSSES_DEGREE=2
export DG_XDIT_RING_DEGREE=2

diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4 \
    --prompts_file prompts.jsonl
```

**Parallelism Options (via DG_XDIT_* env vars):**
- `DG_XDIT_DATA_PARALLEL_DEGREE`: Data parallelism (default: 1)
- `DG_XDIT_ULYSSES_DEGREE`: Ulysses sequence parallelism (default: 1)
- `DG_XDIT_RING_DEGREE`: Ring sequence parallelism (default: 1)
- `DG_XDIT_PIPEFUSION_DEGREE`: PipeFusion parallelism (default: 1)
- `DG_XDIT_USE_CFG_PARALLEL`: Enable CFG parallelism (default: false)

### 3. OpenAI Backend

Use OpenAI's image generation API.

```bash
export OPENAI_API_KEY=your_key
diffgentor t2i --backend openai \
    --model_name dall-e-3 \
    --prompt "A serene mountain landscape"
```

### 4. Google GenAI Backend (Gemini)

Use Google's Gemini native image models.

```bash
export GEMINI_API_KEY=your_key
diffgentor t2i --backend google_genai \
    --model_name gemini-2.5-flash-image \
    --prompt "A futuristic cityscape"
```

See [Google GenAI Guide](google_genai.md) for details.

### 5. DeepGen Backend

Use the DeepGen unified model (Qwen2.5-VL + SD3.5) for text-to-image generation.

```bash
# Set required environment variables
export DG_DEEPGEN_CONFIG=deepgen
export DG_DEEPGEN_DIFFUSION_MODEL_PATH=/path/to/sd3.5
export DG_DEEPGEN_AR_MODEL_PATH=/path/to/qwen2.5-vl

diffgentor t2i --backend deepgen \
    --model_name /path/to/checkpoint.safetensors \
    --prompt "A futuristic cityscape" \
    --guidance_scale 4.0
```

**CLI Parameters:**
- `--model_name`: Path to model checkpoint (.safetensors or .pt)
- `--guidance_scale`: CFG guidance scale (default: 4.0)
- `--num_inference_steps`: Inference steps (default: 50)
- `--height`: Output height (default: 512)
- `--width`: Output width (default: 512)

**Environment Variables:**
- `DG_DEEPGEN_CONFIG`: Config file name (required, e.g., `deepgen`)
- `DG_DEEPGEN_DIFFUSION_MODEL_PATH`: Path to SD3.5 diffusion model (required)
- `DG_DEEPGEN_AR_MODEL_PATH`: Path to Qwen2.5-VL AR model (required)
- `DG_DEEPGEN_MAX_LENGTH`: Max sequence length (default: 1024)

See [DeepGen Guide](deepgen.md) for details.

## CLI Options

### Input/Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt` | Single prompt for generation | - |
| `--prompts_file` | Path to prompts file (JSONL, JSON, TXT, CSV) | - |
| `--output_dir` | Directory to save generated images | `./output` |
| `--output_name_column` | Column name for custom output filename | - |

### Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--height` | Image height | Model default |
| `--width` | Image width | Model default |
| `--num_inference_steps` | Number of denoising steps | 28 |
| `--guidance_scale` | Classifier-free guidance scale | 3.5 |
| `--num_images_per_prompt` | Images per prompt | 1 |
| `--negative_prompt` | Negative prompt | - |
| `--seed` | Random seed for reproducibility | - |
| `--batch_size` | Batch size for generation | 1 |

### Backend Options

| Option | Description | Default |
|--------|-------------|---------|
| `--backend` | Backend to use (`diffusers`, `xdit`, `openai`) | `diffusers` |
| `--model_name` | Model name or path (required) | - |
| `--model_type` | Explicit pipeline selection | Auto-detected |
| `--device` | Device (`cuda`, `cpu`, `cuda:N`) | `cuda` |
| `--num_gpus` | Number of GPUs | 1 |

### Optimization Options

| Option | Description | Default |
|--------|-------------|---------|
| `--torch_dtype` | Data type (`float16`, `bfloat16`, `float32`) | `bfloat16` |
| `--enable_compile` | Enable torch.compile | False |
| `--enable_cpu_offload` | Enable model CPU offload | False |
| `--enable_vae_slicing` | Enable VAE slicing | False |
| `--enable_vae_tiling` | Enable VAE tiling | False |
| `--attention_backend` | Attention backend (`flash`, `sage`, `xformers`) | - |
| `--cache_type` | Cache acceleration type | - |

## Prompts File Formats

### JSONL Format
```jsonl
{"prompt": "A cat sleeping on a couch", "negative_prompt": "blurry"}
{"prompt": "A dog playing in the park"}
```

### JSON Format
```json
[
  {"prompt": "A mountain landscape", "height": 1024, "width": 1024},
  {"prompt": "Ocean sunset"}
]
```

### TXT Format (one prompt per line)
```
A beautiful sunset over the ocean
A futuristic cityscape at night
```

### CSV Format
```csv
prompt,negative_prompt,height,width
"A forest path",blurry,1024,1024
"City skyline",,768,1024
```

## Examples

### Basic Generation

```bash
# Generate a single image
diffgentor t2i --backend diffusers \
    --model_name stabilityai/stable-diffusion-3-medium \
    --prompt "A cyberpunk street scene" \
    --height 1024 --width 1024

# Generate multiple images
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --prompts_file prompts.jsonl \
    --num_images_per_prompt 4 \
    --output_dir ./generated
```

### With Optimizations

```bash
# Enable torch.compile and VAE optimizations
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "Detailed landscape painting" \
    --enable_compile \
    --enable_vae_slicing \
    --enable_vae_tiling \
    --attention_backend flash
```

### Multi-GPU with xDiT

```bash
# 4-GPU with Ulysses parallelism
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4 \
    --ulysses_degree 4 \
    --prompts_file large_prompts.jsonl

# 8-GPU with hybrid parallelism
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --ulysses_degree 2 \
    --ring_degree 2 \
    --data_parallel_degree 2 \
    --use_cfg_parallel
```

### Multi-Node Distributed

```bash
# Node 0
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --num_nodes 2 \
    --node_rank 0 \
    --prompts_file prompts.jsonl

# Node 1
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --num_nodes 2 \
    --node_rank 1 \
    --prompts_file prompts.jsonl
```

## Model-Specific Notes

### FLUX Models

```bash
# FLUX.1-dev (high quality, slower)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --num_inference_steps 50 \
    --guidance_scale 3.5

# FLUX.1-schnell (fast, 4 steps)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --num_inference_steps 4 \
    --guidance_scale 0.0
```

### Stable Diffusion 3

```bash
diffgentor t2i --backend diffusers \
    --model_name stabilityai/stable-diffusion-3-medium \
    --num_inference_steps 28 \
    --guidance_scale 7.0
```

### SDXL

```bash
diffgentor t2i --backend diffusers \
    --model_name stabilityai/stable-diffusion-xl-base-1.0 \
    --num_inference_steps 30 \
    --guidance_scale 7.5 \
    --height 1024 --width 1024
```

## Output Naming

### Default Output Naming

By default, output files are named using zero-padded indices:

- Single image per prompt: `{index:06d}.png` (e.g., `000000.png`, `000001.png`)
- Multiple images per prompt: `{index:06d}_{sub_index:02d}.png` (e.g., `000000_00.png`, `000000_01.png`)

### Custom Output Naming

When using a prompts file (CSV, JSONL, JSON), you can specify a column/field for custom output paths:

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.csv \
    --output_dir ./results \
    --output_name_column output_path
```

**Prompts CSV example:**

```csv
index,prompt,output_path
0,"A beautiful sunset",landscapes/sunset
1,"Mountain peak","mountains/peak_001.jpg"
2,"Ocean waves",seascapes/waves.png
```

**Output behavior:**

| Column Value | Output File |
|--------------|-------------|
| `landscapes/sunset` | `results/landscapes/sunset.png` |
| `mountains/peak_001.jpg` | `results/mountains/peak_001.jpg` |
| `seascapes/waves.png` | `results/seascapes/waves.png` |

**Features:**
- Supported formats: `.png`, `.jpg`, `.jpeg` (other extensions default to `.png`)
- Parent directories are automatically created
- Resume support: checks for existing files with custom names
- For `num_images_per_prompt > 1`, sub-index is appended: `landscapes/sunset_00.png`, `landscapes/sunset_01.png`
