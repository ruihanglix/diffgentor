# Google GenAI (Gemini) Editing Backend

Edit images using Google's Gemini native image models via multimodal input.

## Features

- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
- Multimodal input support (instruction + multiple images)

## Prerequisites

Install the Google GenAI SDK:

```bash
pip install google-genai
```

Set your API key:

```bash
export GEMINI_API_KEY=your_api_key
```

## Supported Models

| Model | Description |
|-------|-------------|
| `gemini-2.5-flash-image` | Gemini 2.5 Flash with native image generation (Nano Banana) |
| `gemini-3-pro-image-preview` | Gemini 3 Pro image preview (Nano Banana Pro) |

## Basic Usage

```bash
diffgentor edit --backend google_genai \
    --model_name gemini-2.5-flash-image \
    --input data.csv \
    --output_dir ./edited
```

## Environment Variables

### Single Endpoint

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key (required) | - |
| `DG_GEMINI_BASE_URL` | Base URL override (for proxy) | - |
| `DG_GEMINI_API_VERSION` | API version (v1, v1beta, v1alpha) | - |
| `DG_GEMINI_RATE_LIMIT` | Rate limit per minute | 0 (no limit) |
| `DG_GEMINI_ASPECT_RATIO` | Default aspect ratio for output | - |

### Multiple Endpoints

| Variable | Description | Example |
|----------|-------------|---------|
| `DG_GEMINI_ENDPOINTS` | Comma-separated base URLs | `http://proxy1.com,http://proxy2.com` |
| `DG_GEMINI_API_KEYS` | Comma-separated API keys | `key1,key2,key3` |
| `DG_GEMINI_RATE_LIMITS` | Comma-separated rate limits | `60,30,60` |
| `DG_GEMINI_WEIGHTS` | Comma-separated weights | `1,2,1` |

## CLI Arguments

Pool settings are configured via CLI arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--timeout` | Request timeout in seconds | 300 |
| `--api_max_retries` | Max retry attempts | 0 |
| `--retry_delay` | Initial retry delay in seconds | 1.0 |
| `--max_global_workers` | Total concurrent workers | 16 |
| `--num_processes` | Number of worker processes | 4 |

Threads per process = `max_global_workers / num_processes` (e.g., 16/4 = 4 threads per process)

```bash
# Using CLI arguments
diffgentor edit --backend google_genai \
    --timeout 120 \
    --api_max_retries 3 \
    --max_global_workers 32 \
    --num_processes 8 \
    --input data.csv
```

## Input Format

CSV with image URL and instruction:

```csv
image_url,instruction
https://example.com/cat.jpg,"Add a red hat to the cat"
/local/path/dog.png,"Change the background to a beach"
```

## How It Works

The backend sends multimodal content to Gemini:

```
[instruction_text, input_image1, input_image2, ...]
```

Gemini processes the instruction with the images and generates an edited result.

## Examples

### Basic Editing

```bash
GEMINI_API_KEY=your_key diffgentor edit --backend google_genai \
    --input data.csv \
    --output_dir ./edited
```

### Multiple API Keys (Load Balancing)

```bash
# Use multiple keys for higher throughput
DG_GEMINI_API_KEYS=key1,key2,key3 \
DG_GEMINI_RATE_LIMIT=60 \
diffgentor edit --backend google_genai \
    --input data.csv
```

### Multiple Endpoints

```bash
# Use multiple proxy endpoints
DG_GEMINI_ENDPOINTS=http://proxy1.com,http://proxy2.com \
DG_GEMINI_API_KEYS=key1,key2 \
DG_GEMINI_RATE_LIMITS=30,60 \
DG_GEMINI_WEIGHTS=1,2 \
diffgentor edit --backend google_genai \
    --input data.csv
```

### With Timeout and Retry

```bash
DG_GEMINI_TIMEOUT=120 \
DG_GEMINI_MAX_RETRIES=3 \
DG_GEMINI_RETRY_DELAY=2.0 \
diffgentor edit --backend google_genai \
    --input data.csv
```

### High Concurrency

```bash
DG_GEMINI_API_KEYS=key1,key2,key3,key4 \
DG_GEMINI_MAX_WORKERS=8 \
diffgentor edit --backend google_genai \
    --input large_dataset.csv
```

### Custom Aspect Ratio

```bash
DG_GEMINI_ASPECT_RATIO=16:9 diffgentor edit --backend google_genai \
    --input data.csv
```

### Process Specific Rows

```bash
diffgentor edit --backend google_genai \
    --input data.csv \
    --filter_rows 0:100
```

## Multi-Image Editing

Gemini supports multiple input images for context-aware editing:

```csv
image_url,image_url_2,instruction
img1.jpg,img2.jpg,"Combine these two images"
style.jpg,content.jpg,"Apply the style from the first image to the second"
```

The backend will send all images along with the instruction.

## Rate Limiting

Built-in rate limiter using sliding window (per endpoint):

```bash
# Limit to 30 requests per minute
DG_GEMINI_RATE_LIMIT=30 diffgentor edit --backend google_genai \
    --input data.csv
```

With multiple endpoints, each has independent rate limiting.

## Error Handling

The backend handles common errors:
- Rate limit errors (with automatic retry if configured)
- API key errors (failover to other endpoints/keys)
- Network timeouts (configurable timeout and retry)

## Load Balancing

Multiple endpoints are selected using weighted round-robin. Endpoints with recent errors receive reduced weight:

```bash
# Endpoint 2 gets 2x traffic
DG_GEMINI_ENDPOINTS=http://proxy1.com,http://proxy2.com \
DG_GEMINI_WEIGHTS=1,2 \
diffgentor edit --backend google_genai --input data.csv
```

## API Response Handling

The backend extracts images from:

1. `response.candidates[].content.parts[].inline_data` (primary)
2. `response.parts[].inline_data` (fallback)
3. Data URLs in text responses (e.g., `![image](data:image/png;base64,...)`)

This ensures compatibility with various proxy configurations.

## Comparison with OpenAI

| Feature | Google GenAI | OpenAI |
|---------|--------------|--------|
| Multi-image input | ✓ | ✗ |
| Instruction understanding | Strong (multimodal) | API-based |
| Output format | Inline data | b64_json |
| Rate limiting | Built-in | Built-in |
| Multi-endpoint | ✓ | ✓ |
| Multi-key | ✓ | ✓ |

## Notes

- **No local GPU required**: Processing happens on Google's servers
- **Multimodal input**: Send instruction + images together for best results
- **Rate limits**: Check your API quota at [Google AI Studio](https://ai.google.dev)
- **Cost**: Check [Google AI pricing](https://ai.google.dev/pricing) for current rates
- **Multi-image**: Pass multiple images for context-aware editing
- **Timeout**: Default 5 minutes; adjust for slow proxies
