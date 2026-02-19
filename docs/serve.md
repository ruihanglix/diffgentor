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

## Architecture

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

## Limitations

- **Single model**: The server loads exactly one model. To serve multiple models, run multiple server instances on different ports.
- **No streaming**: Image streaming (`stream=True`) is not yet supported. The full image is returned once generation completes.
- **No URL response**: Only `b64_json` response format is supported. The `url` format (which requires file hosting) is not implemented.
- **No authentication**: The server does not validate API keys. Use a reverse proxy if you need authentication in production.

## Comparison with Batch Mode

| Feature | Batch (`t2i` / `edit`) | Serve |
|---------|------------------------|-------|
| Input | File (CSV, JSONL, prompts file) | HTTP request |
| Output | Files on disk | JSON response (base64) |
| Concurrency | Multi-GPU / multi-process | Single model, async requests |
| Use case | Large-scale data synthesis | Interactive / API integration |
| Resume support | Yes (`--resume`) | N/A (stateless) |
