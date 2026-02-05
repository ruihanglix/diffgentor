# Google GenAI (Gemini) Backend for T2I

Generate images using Google's Gemini native image models via the Google GenAI SDK.

## Features

- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
- Per-endpoint rate limiting

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
# Default model (gemini-2.5-flash-image)
diffgentor t2i --backend google_genai \
    --prompt "A futuristic city at sunset"

# Specific model
diffgentor t2i --backend google_genai \
    --model_name gemini-3-pro-image-preview \
    --prompt "A serene mountain landscape"

# With aspect ratio
DG_GEMINI_ASPECT_RATIO=16:9 diffgentor t2i --backend google_genai \
    --prompt "A panoramic ocean view"
```

## Environment Variables

### Single Endpoint

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Gemini API key (required) | - |
| `DG_GEMINI_BASE_URL` | Base URL override (for proxy) | - |
| `DG_GEMINI_API_VERSION` | API version (v1, v1beta, v1alpha) | - |
| `DG_GEMINI_RATE_LIMIT` | Rate limit per minute | 0 (no limit) |
| `DG_GEMINI_ASPECT_RATIO` | Default aspect ratio | - |

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
diffgentor t2i --backend google_genai \
    --timeout 120 \
    --api_max_retries 3 \
    --max_global_workers 32 \
    --num_processes 8 \
    --prompts_file prompts.jsonl
```

## Examples

### Basic Generation

```bash
GEMINI_API_KEY=your_key diffgentor t2i --backend google_genai \
    --prompt "A cat wearing a space suit"
```

### Multiple API Keys (Load Balancing)

```bash
# Use multiple keys for higher throughput
DG_GEMINI_API_KEYS=key1,key2,key3 \
DG_GEMINI_RATE_LIMIT=60 \
diffgentor t2i --backend google_genai \
    --prompts_file prompts.jsonl
```

### Multiple Endpoints

```bash
# Use multiple proxy endpoints
DG_GEMINI_ENDPOINTS=http://proxy1.com,http://proxy2.com \
DG_GEMINI_API_KEYS=key1,key2 \
DG_GEMINI_RATE_LIMITS=30,60 \
DG_GEMINI_WEIGHTS=1,2 \
diffgentor t2i --backend google_genai \
    --prompts_file prompts.jsonl
```

### With Timeout and Retry

```bash
DG_GEMINI_TIMEOUT=120 \
DG_GEMINI_MAX_RETRIES=3 \
DG_GEMINI_RETRY_DELAY=2.0 \
diffgentor t2i --backend google_genai \
    --prompt "A detailed artwork"
```

### High Concurrency

```bash
DG_GEMINI_API_KEYS=key1,key2,key3,key4 \
DG_GEMINI_MAX_WORKERS=8 \
diffgentor t2i --backend google_genai \
    --prompts_file large_prompts.jsonl \
    --num_images_per_prompt 4
```

### Custom Aspect Ratio

```bash
# 16:9 widescreen
DG_GEMINI_ASPECT_RATIO=16:9 diffgentor t2i --backend google_genai \
    --prompt "A cinematic landscape"

# 9:16 portrait
DG_GEMINI_ASPECT_RATIO=9:16 diffgentor t2i --backend google_genai \
    --prompt "A tall skyscraper"

# Square
DG_GEMINI_ASPECT_RATIO=1:1 diffgentor t2i --backend google_genai \
    --prompt "A profile picture"
```

## Aspect Ratios

Supported aspect ratios:
- `1:1` - Square
- `16:9` - Widescreen
- `9:16` - Portrait
- `4:3` - Standard
- `3:4` - Portrait standard
- `3:2` - Classic photo
- `2:3` - Portrait classic

The backend can also auto-convert `--width` and `--height` to appropriate aspect ratios.

## API Behavior

The Google GenAI backend:

1. Uses the `google-genai` SDK for API calls
2. Requests IMAGE response modality
3. Extracts images from `inline_data` parts
4. Falls back to parsing data URLs from text responses (for some proxy configurations)

## Rate Limiting

Built-in rate limiter using sliding window (per endpoint):

```bash
# Limit to 60 requests per minute per endpoint
DG_GEMINI_RATE_LIMIT=60 diffgentor t2i --backend google_genai \
    --prompts_file large_prompts.jsonl
```

With multiple endpoints, each has independent rate limiting:

```bash
DG_GEMINI_ENDPOINTS=,http://proxy.com \
DG_GEMINI_API_KEYS=key1,key2 \
DG_GEMINI_RATE_LIMITS=60,30 \
diffgentor t2i --backend google_genai --prompts_file prompts.jsonl
```

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
diffgentor t2i --backend google_genai --prompts_file prompts.jsonl
```

## Differences from OpenAI Backend

| Feature | Google GenAI | OpenAI |
|---------|--------------|--------|
| Response format | Multimodal (inline_data) | b64_json |
| Size control | Aspect ratio | Fixed sizes |
| Rate limiting | Built-in | Built-in |
| Multi-endpoint | ✓ | ✓ |
| Multi-key | ✓ | ✓ |

## Notes

- **No local GPU required**: All processing happens on Google's servers
- **Rate limits**: Check [Google AI Studio](https://ai.google.dev) for current limits
- **Pricing**: Check [Google AI pricing](https://ai.google.dev/pricing) for costs
- **API versions**: Use `v1beta` for latest features
- **Timeout**: Default 5 minutes; adjust for slow proxies
