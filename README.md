# Diffgentor

A unified visual generation data synthesis tool for batch image generation and editing, designed for GenArena evaluation and beyond.

## Abstract

Diffgentor is an efficient pipeline for batch image generation using various image generation and editing models. It supports multiple backends including diffusers, OpenAI API, Google GenAI (Gemini), and third-party models like Step1X-Edit, BAGEL, and Emu3.5.

Key features:
- **Multiple Backends**: diffusers, xDiT (multi-GPU), OpenAI, Google GenAI, and third-party models
- **Batch Processing**: Efficient batch inference with multi-process/multi-thread support
- **GenArena Integration**: Generate model outputs for GenArena pairwise evaluation
- **Optimization Suite**: VAE slicing/tiling, torch.compile, attention backends, and more

## Quick Start

### Installation

**Option 1: pip install**

```bash
pip install diffgentor
```

**Option 2: From source with third-party models**

```bash
git clone --recursive https://github.com/ruihanglix/diffgentor.git
cd diffgentor
pip install -e .
```

> The `--recursive` flag initializes git submodules for third-party models (Step1X-Edit, BAGEL, Emu3.5, DreamOmni2, etc.)

### Download GenArena Dataset

```bash
hf download rhli/genarena --repo-type dataset --local-dir ./data
```

### Generate Images for MultiRef Subset

Example using FLUX.2 [klein] 4B model:

```bash
diffgentor edit --backend diffusers \
    --model_name black-forest-labs/FLUX.2-klein-4B \
    --input ./data/multiref/ \
    --output_dir ./output/multiref/FLUX2-klein-4B/
```

## Supported Backends

| Backend | Type | Description |
|---------|------|-------------|
| `diffusers` | T2I / Editing | HuggingFace diffusers with auto pipeline detection |
| `xdit` | T2I | Multi-GPU inference with xDiT parallelism |
| `openai` | T2I / Editing | OpenAI API (GPT-Image, DALL-E) |
| `google_genai` | T2I / Editing | Google GenAI (Gemini native image models) |
| `step1x` | Editing | Step1X-Edit model |
| `bagel` | Editing | ByteDance BAGEL model |
| `emu35` | Editing | BAAI Emu3.5 model |
| `dreamomni2` | Editing | DreamOmni2 (FLUX.1-Kontext + Qwen2.5-VL) |
| `flux_kontext_official` | Editing | BFL official Flux Kontext |
| `hunyuan_image_3` | Editing | Tencent HunyuanImage-3.0-Instruct |

## Documentation

| Document | Description |
|----------|-------------|
| [Image Editing Guide](./docs/editing/README.md) | Comprehensive guide for image editing |
| [Text-to-Image Guide](./docs/t2i/README.md) | Text-to-image generation guide |
| [Optimization Guide](./docs/optimization.md) | Memory and speed optimization |
| [Prompt Enhancement](./docs/prompt_enhance.md) | LLM-based prompt enhancement |
| [Environment Variables](./docs/env_vars.md) | Configuration via environment variables |

### Backend-Specific Guides

- [Diffusers Models](./docs/editing/diffusers.md) - Qwen, FLUX, LongCat
- [Step1X-Edit](./docs/editing/step1x.md) - Step1X-Edit v1.0/v1.1
- [BAGEL](./docs/editing/bagel.md) - ByteDance BAGEL
- [Emu3.5](./docs/editing/emu35.md) - BAAI Emu3.5
- [DreamOmni2](./docs/editing/dreamomni2.md) - DreamOmni2
- [Flux Kontext](./docs/editing/flux_kontext.md) - BFL official
- [HunyuanImage-3.0](./docs/editing/hunyuan_image_3.md) - Tencent HunyuanImage
- [OpenAI](./docs/editing/openai.md) - GPT-Image API
- [Google GenAI](./docs/editing/google_genai.md) - Gemini

## Environment Variables

Model-specific parameters are configured via `DG_*` environment variables:

```bash
# Step1X-Edit
DG_STEP1X_VERSION=v1.1
DG_STEP1X_SIZE_LEVEL=512

# BAGEL
DG_BAGEL_CFG_TEXT_SCALE=3.0
DG_BAGEL_CFG_IMG_SCALE=1.5

# API backends
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
```

See [Environment Variables](./docs/env_vars.md) for the complete list.

## License

Apache-2.0
