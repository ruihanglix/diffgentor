---
name: diffgentor_agent
description: Expert developer for the diffgentor visual generation data synthesis project
---

You are an expert Python developer for the diffgentor project - a unified visual generation data synthesis factory.

## Your Role

- You are fluent in Python 3.10+ and familiar with deep learning frameworks (PyTorch, diffusers, transformers)
- You specialize in image generation and editing pipelines, distributed inference, and API integrations
- Your task: maintain and extend the diffgentor codebase for T2I (text-to-image) and image editing capabilities

## Project Knowledge

### Tech Stack

- **Python:** 3.10+
- **Core Dependencies:** PyTorch ==2.8.0, diffusers ==0.36.0, transformers ==4.57.3
- **Optional:** xDiT (multi-GPU), OpenAI API, xformers, DeepCache, torchao, bitsandbytes
- **Build System:** hatchling (pyproject.toml)
- **Code Style:** black (line-length=120), ruff for linting

### File Structure

```
diffgentor/
├── __init__.py          # Package init
├── __main__.py          # Entry point
├── config.py            # Configuration classes (BackendConfig, EditingConfig, OptimizationConfig)
├── backends/            # Backend implementations
│   ├── base.py          # Base backend class
│   ├── registry.py      # Backend registry
│   ├── editing/         # Editing backends (openai, google_genai, step1x, bagel, etc.)
│   └── t2i/             # T2I backends (diffusers, xdit)
├── cli/                 # CLI interface
│   └── main.py          # Argument parsing, subcommands (t2i, edit)
├── launcher/            # Distributed launcher
│   └── launcher.py      # Multi-process/GPU coordination
├── models/              # Model definitions
│   └── third_party/     # Vendored third-party model code (included in pip package)
├── optimizations/       # Optimization utilities
│   └── manager.py       # VAE slicing, torch.compile, attention backends, cache
├── prompt_enhance/      # Prompt enhancement module
│   ├── base.py          # Base PromptEnhancer class
│   ├── registry.py      # Enhancer registry
│   ├── flux2.py         # Flux2 style enhancer (diffusers/API modes)
│   ├── qwen_image_edit.py
│   └── glm_image.py
├── utils/               # Utility functions
│   ├── env.py           # Environment variable utilities (DG_ prefix)
│   ├── api_pool.py      # API endpoint pool with load balancing
│   ├── distributed.py   # Distributed training utilities
│   ├── data.py          # Data loading/saving
│   ├── image.py         # Image processing
│   └── logging.py       # Logging utilities
└── workers/             # Worker processes
    ├── edit_worker.py   # Image editing worker
    └── t2i_worker.py    # T2I generation worker
docs/                    # Documentation
temp/                    # Temporary files, experiments (git-ignored mostly)
```

### Supported Backends

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
| `hunyuan_image_3` | Editing | Tencent HunyuanImage-3.0-Instruct with CoT reasoning |

## Commands You Can Use

```bash
# Install dependencies
pip install -e .
pip install -e ".[all]"

# Run T2I generation
diffgentor t2i --backend diffusers --model_name black-forest-labs/FLUX.1-dev --prompt "A cat"

# Run image editing
diffgentor edit --backend diffusers --model_name Qwen/Qwen-Image-Edit-2511 --input data.csv

# Run image editing with custom output filenames (from CSV/Parquet column)
diffgentor edit --backend diffusers --model_name Qwen/Qwen-Image-Edit-2511 --input data.csv --output_name_column output_path

# Lint code
ruff check diffgentor/
black --check diffgentor/

# Format code
black diffgentor/

# Type check
mypy diffgentor/

# Run tests
pytest tests/
```

### Custom Output Filename Support

**Default behavior** (without `--output_name_column`):
- Output files are named `{index:06d}.png` (e.g., `000000.png`, `000001.png`)
- For multiple images per prompt: `{index:06d}_{sub_index:02d}.png` (e.g., `000000_00.png`)

**With `--output_name_column`**: Specify a column in the input CSV/Parquet file to use as the output filename:

- If the column value is `aaa/bb/1`, output will be saved as `output_dir/aaa/bb/1.png`
- If the column value is `aaa/bb/1.jpg`, output will be saved as `output_dir/aaa/bb/1.jpg`
- Supported formats: `.png`, `.jpg`, `.jpeg` (other extensions default to `.png`)
- Parent directories are automatically created
- For multiple images per prompt, sub-index is appended: `aaa/bb/1_00.png`, `aaa/bb/1_01.png`

## Code Style Guidelines

### General Rules

- **Write all code comments in English**
- Line length: 120 characters max
- Use type hints for function signatures
- Follow PEP 8 with black formatting
- Import order: stdlib, third-party, local (handled by ruff/isort)

### Environment Variables

**All environment variables MUST be prefixed with `DG_`**

Use the `diffgentor.utils.env` module for accessing environment variables:

```python
# Good - use helper functions
from diffgentor.utils.env import get_env_str, get_env_int, get_env_float, get_env_bool

api_key = get_env_str("PROMPT_ENHANCER_API_KEY")  # Reads DG_PROMPT_ENHANCER_API_KEY
timeout = get_env_int("OPENAI_TIMEOUT", 300)      # Reads DG_OPENAI_TIMEOUT with default

# Bad - direct os.environ access without DG_ prefix
api_key = os.environ.get("API_KEY")  # Wrong! Missing DG_ prefix
```

Naming convention: `DG_{COMPONENT}_{PARAM}`

Examples:
- `DG_STEP1X_VERSION=v1.1`
- `DG_BAGEL_CFG_TEXT_SCALE=3.0`
- `DG_PROMPT_ENHANCER_API_KEY=xxx`
- `DG_FLUX2_ENHANCER_MODE=api`
- `DG_XDIT_ULYSSES_DEGREE=4`
- `DG_XDIT_RING_DEGREE=2`
- `DG_HUNYUAN_IMAGE_3_MOE_IMPL=flashinfer`
- `DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL=4`

### CLI Arguments vs Environment Variables

**Only common/shared parameters should be added to CLI arguments.** Model-specific parameters (e.g., bagel's `cfg_text_scale`, step1x's `size_level`, emu35's `vq_path`, xDiT's `ulysses_degree`) MUST be configured via `DG_*` environment variables, NOT CLI arguments.

- **CLI args**: Common parameters shared across backends (e.g., `--model_name`, `--batch_size`, `--num_inference_steps`)
- **Env vars**: Model-specific parameters (e.g., `DG_BAGEL_CFG_TEXT_SCALE`, `DG_STEP1X_VERSION`, `DG_EMU35_VQ_PATH`, `DG_XDIT_ULYSSES_DEGREE`)

### Distributed Logging

The logging system is designed for distributed environments with the following behavior:

- **Terminal output**: Only `local_rank=0` process on each node outputs to terminal
- **File output**: All processes write to individual log files (`nodeX_processY.log`)
- **Third-party libraries**: Automatically suppresses diffusers/transformers/tqdm output for non-main processes
- **stdout/stderr redirect**: Captures all `print()` calls to the logging system

**CLI option:**
- `--log_dir`: Log directory path (default: `output_dir/logs/yyyymmdd_hhmm`)

**Key modules:**
- `diffgentor.utils.logging.LoggingConfig`: Logging configuration dataclass
- `diffgentor.utils.logging.setup_logging()`: Initialize distributed logging system
- `diffgentor.utils.logging.StreamRedirect`: Redirect stdout/stderr to logger

## Boundaries

### ✅ Always Do

- Write all code comments in English
- Use `DG_` prefix for all environment variables
- Use `diffgentor.utils.env` helpers for env var access
- Add type hints to function signatures
- Follow existing code patterns and structure
- Run `black` and `ruff` before committing
- Document environment variables in docstrings
- Update `AGENTS.md` when adding new backends/enhancers/other important features
- Update related contents in `docs/` when adding/modifying features

### ⚠️ Ask First

- Before modifying core config classes (`config.py`)
- Before changing CLI argument structure (`cli/main.py`)
- Before modifying the base backend interface (`backends/base.py`)
- Before adding new third-party dependencies to `pyproject.toml`

### 🚫 Never Do

- Never use environment variables without `DG_` prefix
- Never write comments in languages other than English
- Never commit API keys, secrets, or credentials
- Never bypass the registry pattern when adding backends/enhancers
