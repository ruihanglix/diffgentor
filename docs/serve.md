# Serve Mode (OpenAI-Compatible API)

Diffgentor can run as an HTTP server that exposes an OpenAI-compatible Images API. This allows any tool or library that speaks the OpenAI protocol to use diffgentor backends for image generation and editing — no code changes required on the client side.

## Prerequisites

Install the serve extras:

```bash
pip install diffgentor[serve]
# or include it with everything
pip install diffgentor[all]
```

This adds `fastapi`, `uvicorn`, and `python-multipart`.

## Quick Start

### Text-to-Image Server

```bash
diffgentor serve --mode t2i \
    --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --port 8000
```

### Image Editing Server

```bash
diffgentor serve --mode edit \
    --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --model_type qwen \
    --port 8000
```

Once the server is running you can use the standard OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# Text-to-Image
result = client.images.generate(
    prompt="A cat sitting on a windowsill",
    model="FLUX.1-dev",
    n=1,
    size="1024x1024",
)

# The image is returned as base64
import base64
img_bytes = base64.b64decode(result.data[0].b64_json)
with open("cat.png", "wb") as f:
    f.write(img_bytes)
```

## API Endpoints

| Method | Path | Description | Requires |
|--------|------|-------------|----------|
| `POST` | `/v1/images/generations` | Text-to-Image generation | `--mode t2i` |
| `POST` | `/v1/images/edits` | Image editing (multipart/form-data) | `--mode edit` |
| `GET` | `/v1/models` | List loaded model(s) | always |
| `GET` | `/v1/models/{model_id}` | Retrieve a specific model | always |
| `POST` | `/v1/set_lora` | Load and activate a LoRA adapter | diffusers backend |
| `POST` | `/v1/merge_lora_weights` | Fuse LoRA weights into the base model | diffusers backend |
| `POST` | `/v1/unmerge_lora_weights` | Unfuse LoRA, restore base model | diffusers backend |
| `GET` | `/v1/list_loras` | List loaded LoRA adapters | diffusers backend |

Calling an endpoint whose mode is not active returns HTTP 400 with a descriptive message.

### POST /v1/images/generations

Generate images from a text prompt. Accepts a JSON body.

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | *required* | Text description of the desired image |
| `model` | string | - | Informational; the server always uses the loaded model |
| `n` | int | `1` | Number of images to generate |
| `size` | string | model default | Image size, e.g. `"1024x1024"`, `"1536x1024"` |
| `quality` | string | - | Quality hint (passed to backend if supported) |
| `response_format` | string | `"b64_json"` | Only `b64_json` is supported |
| `output_format` | string | `"png"` | Image encoding: `png`, `jpeg`, or `webp` |
| `background` | string | - | Background mode (if backend supports it) |
| `style` | string | - | Style hint (if backend supports it) |

**Response:**

```json
{
  "created": 1740000000,
  "data": [
    {
      "b64_json": "<base64-encoded image>",
      "revised_prompt": null,
      "url": null
    }
  ]
}
```

### POST /v1/images/edits

Edit an image given a prompt. Uses `multipart/form-data` (same as the OpenAI client).

**Form fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file(s) | *required* | One or more input images |
| `prompt` | string | *required* | Editing instruction |
| `model` | string | - | Informational |
| `n` | int | `1` | Number of output images |
| `size` | string | - | Target size, e.g. `"1024x1024"` |
| `quality` | string | - | Quality hint |
| `response_format` | string | `"b64_json"` | Only `b64_json` is supported |
| `output_format` | string | `"png"` | Image encoding |
| `mask` | file | - | Optional mask image |

**Response:** Same `ImagesResponse` format as generations.

### GET /v1/models

Returns the model currently loaded by the server.

```json
{
  "object": "list",
  "data": [
    {
      "id": "black-forest-labs/FLUX.1-dev",
      "object": "model",
      "created": 0,
      "owned_by": "diffgentor"
    }
  ]
}
```

## CLI Options

The `serve` subcommand accepts all common backend and optimization options (same as `t2i` and `edit`), plus the following server-specific options:

### Server Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mode` | `t2i` / `edit` | *required* | Which mode to serve |
| `--host` | string | `0.0.0.0` | Host to bind to |
| `--port` | int | `8000` | Port to listen on |
| `--max_concurrent` | int | `1` | Max concurrent GPU inference requests |

### Backend & Optimization Options

All the standard options are available:

| Option | Description |
|--------|-------------|
| `--backend` | Backend to use (`diffusers`, `openai`, `google_genai`, etc.) |
| `--model_name` | Model name or path (required) |
| `--model_type` | Explicit pipeline type (auto-detected if omitted) |
| `--device` | Device (`cuda`, `cpu`, `cuda:N`) |
| `--torch_dtype` | Data type (`float16`, `bfloat16`, `float32`) |
| `--enable_compile` | Enable `torch.compile` |
| `--enable_cpu_offload` | Enable model CPU offload |
| `--enable_vae_slicing` | Enable VAE slicing |
| `--enable_vae_tiling` | Enable VAE tiling |
| `--attention_backend` | Attention backend (`flash`, `sage`, `xformers`) |
| `--cache_type` | Cache acceleration type |

## Environment Variables

Serve-specific parameters that apply to every request:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DG_SERVE_NUM_INFERENCE_STEPS` | int | model default | Override denoising steps |
| `DG_SERVE_GUIDANCE_SCALE` | float | model default | Override guidance scale |

All other `DG_*` environment variables (model-specific parameters) work the same as in batch mode. See [Environment Variables](env_vars.md) for the full list.

```bash
DG_SERVE_NUM_INFERENCE_STEPS=28 \
DG_SERVE_GUIDANCE_SCALE=3.5 \
diffgentor serve --mode t2i \
    --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev
```

## Client Examples

### Python (openai)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

# --- Text-to-Image ---
result = client.images.generate(
    prompt="A futuristic cityscape at sunset",
    model="FLUX.1-dev",
    n=2,
    size="1024x1024",
)
for i, img in enumerate(result.data):
    import base64
    with open(f"output_{i}.png", "wb") as f:
        f.write(base64.b64decode(img.b64_json))

# --- Image Editing ---
result = client.images.edit(
    image=open("input.png", "rb"),
    prompt="Add a rainbow in the sky",
    model="Qwen-Image-Edit-2511",
)
import base64
with open("edited.png", "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))
```

### cURL

```bash
# Text-to-Image
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat in space",
    "n": 1,
    "size": "1024x1024"
  }'

# Image Editing
curl -X POST http://localhost:8000/v1/images/edits \
  -F "image=@input.png" \
  -F "prompt=Make the sky purple"

# List Models
curl http://localhost:8000/v1/models
```

### TypeScript / JavaScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "unused",
});

const result = await client.images.generate({
  prompt: "A serene mountain landscape",
  model: "FLUX.1-dev",
  n: 1,
  size: "1024x1024",
});

console.log(result.data[0].b64_json?.substring(0, 40) + "...");
```

## Full Examples

### Serve FLUX.1-dev with Optimizations

```bash
diffgentor serve --mode t2i \
    --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_compile \
    --enable_vae_slicing \
    --attention_backend flash \
    --port 8000
```

### Serve Qwen Editing Model with CPU Offload

```bash
diffgentor serve --mode edit \
    --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --model_type qwen \
    --enable_cpu_offload \
    --port 8001
```

### Serve Step1X-Edit

```bash
DG_STEP1X_VERSION=v1.1 \
DG_STEP1X_SIZE_LEVEL=512 \
diffgentor serve --mode edit \
    --backend step1x \
    --model_name /path/to/step1x \
    --port 8000
```

### Serve HunyuanImage-3.0

```bash
diffgentor serve --mode edit \
    --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --port 8000
```

### Allow Multiple Concurrent Requests

By default only one GPU inference runs at a time. If you have enough VRAM you can increase concurrency:

```bash
diffgentor serve --mode t2i \
    --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --max_concurrent 4 \
    --port 8000
```

## Multi-GPU Serving

When multiple GPUs are available the server can load multiple model replicas and distribute incoming requests across them for higher throughput. The strategy depends on how many GPUs each model instance needs.

### Data Parallelism (1 GPU per model)

For models that fit on a single GPU (most diffusers models), the server loads N replicas in-process, each pinned to a different `cuda:i` device. Requests are dispatched round-robin via an `asyncio.Queue`.

```bash
# 4 GPUs -> 4 replicas, 4x throughput
CUDA_VISIBLE_DEVICES=0,1,2,3 \
diffgentor serve --mode t2i \
    --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --num_gpus 4
```

### Model Sharding (multiple GPUs per model)

Some large models use `device_map="auto"` to shard across multiple GPUs (e.g. HunyuanImage-3.0, Emu3.5). The server spawns separate worker subprocesses, each with its own `CUDA_VISIBLE_DEVICES` slice so that `device_map="auto"` distributes within each worker's visible GPUs.

```bash
# HunyuanImage-3.0: 4 GPUs per model, 8 GPUs total -> 2 replicas
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL=4 \
diffgentor serve --mode edit \
    --backend hunyuan_image_3 \
    --model_name ./HunyuanImage-3-Instruct-Distil \
    --num_gpus 8
```

```bash
# Emu3.5: 2 GPUs per model, 8 GPUs total -> 4 replicas
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
DG_EMU35_GPUS_PER_MODEL=2 \
DG_EMU35_VQ_PATH=/path/to/vq \
diffgentor serve --mode edit \
    --backend emu35 \
    --model_name /path/to/emu35 \
    --num_gpus 8
```

### How it works

The pool type is selected automatically:

| `gpus_per_model` | Pool type | Mechanism |
|-------------------|-----------|-----------|
| 1 (or unset) | `InProcessPool` | N replicas loaded in-process on `cuda:0..N-1` |
| >1 | `SubprocessPool` | N child processes, each with isolated `CUDA_VISIBLE_DEVICES` |
| 0 (all GPUs) | No pool | Single model using all GPUs via `device_map="auto"` |

The `gpus_per_model` value is read from backend-specific environment variables:

| Backend | Environment variable | Default |
|---------|---------------------|---------|
| `emu35` | `DG_EMU35_GPUS_PER_MODEL` | `0` (all GPUs) |
| `hunyuan_image_3` | `DG_HUNYUAN_IMAGE_3_GPUS_PER_MODEL` | `0` (all GPUs) |
| `deepgen` | `DG_DEEPGEN_GPUS_PER_MODEL` | `1` |
| all others | — | `1` |

When `--num_gpus` is not specified, the server auto-detects from `CUDA_VISIBLE_DEVICES` or `torch.cuda.device_count()`. If only 1 GPU is available (or `gpus_per_model=0`), no pool is created and the server falls back to single-model behavior.

## Architecture

### Single-GPU (default)

```
                         ┌─────────────────────────────┐
  OpenAI Python Client   │   FastAPI  (uvicorn)        │
  or any HTTP client ───►│                             │
                         │  /v1/images/generations ────►│──► BaseBackend.generate()
                         │  /v1/images/edits ──────────►│──► BaseEditingBackend.edit()
                         │  /v1/models ────────────────►│──► model info
                         │                             │
                         │  asyncio.Semaphore guards   │
                         │  concurrent GPU access       │
                         └─────────────────────────────┘
```

The server loads the model once at startup and keeps it resident in GPU memory. Incoming requests are handled asynchronously by FastAPI; actual inference is dispatched to a thread via `asyncio.to_thread()` so the event loop stays responsive. A semaphore (controlled by `--max_concurrent`) prevents overloading GPU memory with too many parallel inference calls.

### Multi-GPU with InProcessPool

```
                         ┌──────────────────────────────────────┐
  HTTP Client ──────────►│   FastAPI  (uvicorn)                 │
                         │            │                         │
                         │      asyncio.Queue                   │
                         │       ┌────┴────┐                    │
                         │  Worker 0   Worker 1   Worker 2 ...  │
                         │  (cuda:0)   (cuda:1)   (cuda:2)      │
                         │  Backend    Backend    Backend        │
                         └──────────────────────────────────────┘
```

### Multi-GPU with SubprocessPool

```
  HTTP Client ──────────► FastAPI (main process)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
          Subprocess 0   Subprocess 1   Subprocess 2
          GPU 0,1        GPU 2,3        GPU 4,5
          Backend        Backend        Backend
          (device_map=   (device_map=   (device_map=
           "auto")        "auto")        "auto")
```

## LoRA Hot-Loading

The server supports dynamic LoRA adapter management, allowing you to load, switch, and unload LoRA adapters at runtime without restarting the server. This works with all diffusers-based backends (both T2I and editing).

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/set_lora` | Load and activate a LoRA adapter |
| `POST` | `/v1/merge_lora_weights` | Fuse active LoRA weights into the base model |
| `POST` | `/v1/unmerge_lora_weights` | Unfuse LoRA weights, restoring the base model |
| `GET` | `/v1/list_loras` | List loaded adapters and their status |

### POST /v1/set_lora

Load one or more LoRA adapters and make them active for subsequent inference requests. Supports both single and multiple adapters.

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `lora_nickname` | string or list of strings | *required* | Unique identifier(s) for the adapter(s) |
| `lora_path` | string, list of strings, or null | nickname | Path(s) to `.safetensors` file, directory, or HuggingFace repo ID |
| `target` | string or list of strings | `"all"` | Which component(s) to apply the LoRA to (`"all"`, `"transformer"`, etc.) |
| `strength` | float or list of floats | `1.0` | Adapter strength(s) (scale factor). If scalar, broadcast to all adapters |

**Single LoRA:**

```bash
curl -X POST http://localhost:8000/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
    "lora_nickname": "my_style",
    "lora_path": "/path/to/lora.safetensors",
    "strength": 0.8
  }'
```

**Multiple LoRAs (activated simultaneously):**

```bash
curl -X POST http://localhost:8000/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
    "lora_nickname": ["style_lora", "character_lora"],
    "lora_path": ["/path/to/style.safetensors", "/path/to/character.safetensors"],
    "strength": [0.7, 0.9]
  }'
```

**Notes:**
- When lists are provided, all parameters (`lora_nickname`, `lora_path`, `strength`) must have the same length, or scalar values will be broadcast.
- Multiple LoRAs are applied simultaneously using diffusers' `set_adapters` — their effects stack.
- If you call `set_lora` again with the same nickname and path, the adapter weights are reused from cache — only the strength is updated.

### POST /v1/merge_lora_weights

Fuse the currently active LoRA weights directly into the base model for slightly faster inference.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target` | string | `null` | Which component to merge (e.g. `"all"`, `"transformer"`) |
| `strength` | float | `1.0` | LoRA strength for merge |

```bash
curl -X POST http://localhost:8000/v1/merge_lora_weights \
  -H "Content-Type: application/json" \
  -d '{"strength": 0.8}'
```

### POST /v1/unmerge_lora_weights

Unfuse the LoRA weights, restoring the base model. You must call this before switching to a different LoRA if you used `merge_lora_weights`.

```bash
curl -X POST http://localhost:8000/v1/unmerge_lora_weights
```

### GET /v1/list_loras

List all loaded LoRA adapters and their current status.

```bash
curl http://localhost:8000/v1/list_loras
```

**Response example:**

```json
{
  "loaded_adapters": [
    {"nickname": "my_style", "path": "/path/to/lora.safetensors"}
  ],
  "active": {
    "all": [
      {"nickname": "my_style", "path": "/path/to/lora.safetensors", "merged": false, "strength": 0.8}
    ]
  }
}
```

### Switching LoRAs

```bash
# 1. Load and activate LoRA A
curl -X POST http://localhost:8000/v1/set_lora \
  -d '{"lora_nickname": "lora_a", "lora_path": "/path/to/A.safetensors"}'

# 2. Generate with LoRA A...

# 3. Switch to LoRA B (previous adapter stays cached)
curl -X POST http://localhost:8000/v1/set_lora \
  -d '{"lora_nickname": "lora_b", "lora_path": "/path/to/B.safetensors"}'

# 4. Generate with LoRA B...

# 5. Switch back to LoRA A (instant, loaded from cache)
curl -X POST http://localhost:8000/v1/set_lora \
  -d '{"lora_nickname": "lora_a", "lora_path": "/path/to/A.safetensors"}'
```

### Combining Multiple LoRAs

You can activate multiple LoRA adapters simultaneously. Their effects are stacked.

```bash
# Load a style LoRA and a character LoRA at the same time
curl -X POST http://localhost:8000/v1/set_lora \
  -H "Content-Type: application/json" \
  -d '{
    "lora_nickname": ["style_lora", "character_lora"],
    "lora_path": ["/path/to/style.safetensors", "/path/to/character.safetensors"],
    "strength": [0.7, 0.9]
  }'

# Generate with both LoRAs active...

# Switch to a single LoRA (deactivates the others)
curl -X POST http://localhost:8000/v1/set_lora \
  -d '{"lora_nickname": "style_lora", "lora_path": "/path/to/style.safetensors"}'
```

### Multi-GPU Support

LoRA operations are automatically broadcast to all worker replicas in both `InProcessPool` and `SubprocessPool` modes, keeping all replicas in sync.

### Caveats

- **Supported backends only:** LoRA management requires diffusers-based backends (`diffusers` backend for T2I, or diffusers editing backends). API backends (OpenAI, Google GenAI) and non-diffusers backends (step1x, bagel, emu35) will return HTTP 400.
- **torch.compile:** If `--enable_compile` is used, switching LoRA may invalidate compiled graphs, causing a one-time recompilation delay.
- **xDiT backend:** Not supported for LoRA management.

## Limitations

- **Single model**: The server loads exactly one model. To serve multiple models, run multiple server instances on different ports.
- **No streaming**: Image streaming (`stream=True`) is not yet supported. The full image is returned once generation completes.
- **No URL response**: Only `b64_json` response format is supported. The `url` format (which requires file hosting) is not implemented.
- **No authentication**: The server does not validate API keys. Use a reverse proxy if you need authentication in production.
- **xDiT not supported**: The xDiT backend requires `torchrun` and is not compatible with the serve mode.

## Comparison with Batch Mode

| Feature | Batch (`t2i` / `edit`) | Serve |
|---------|------------------------|-------|
| Input | File (CSV, JSONL, prompts file) | HTTP request |
| Output | Files on disk | JSON response (base64) |
| Concurrency | Multi-GPU / multi-process | Multi-GPU worker pool or single model |
| Use case | Large-scale data synthesis | Interactive / API integration |
| Resume support | Yes (`--resume`) | N/A (stateless) |
