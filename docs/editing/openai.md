# OpenAI Editing Backend

Use OpenAI's GPT-Image API for image editing.

## Features

- Multiple API endpoints with load balancing
- Multiple API keys per endpoint
- Configurable timeout (default: 5 minutes)
- Configurable retry with exponential backoff (default: no retry)
- Thread-safe concurrent processing
- Per-endpoint rate limiting

## Prerequisites

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_api_key_here
```

## Supported Models

| Model | Multi-Image | Description |
|-------|-------------|-------------|
| `gpt-image-1` | ✓ | GPT-Image generation/editing model |
| `dall-e-3` | ✗ | DALL-E 3 (limited editing support) |
| `dall-e-2` | ✗ | DALL-E 2 with inpainting |

## Basic Usage

```bash
diffgentor edit --backend openai \
    --model_name gpt-image-1 \
    --input data.csv \
    --output_dir ./output
```

## Environment Variables

### Single Endpoint

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_API_BASE` | Base URL override | - |
| `DG_OPENAI_RATE_LIMIT` | Rate limit per minute | 0 (no limit) |

### Multiple Endpoints

| Variable | Description | Example |
|----------|-------------|---------|
| `DG_OPENAI_ENDPOINTS` | Comma-separated base URLs | `http://proxy1.com,http://proxy2.com` |
| `DG_OPENAI_API_KEYS` | Comma-separated API keys | `key1,key2,key3` |
| `DG_OPENAI_RATE_LIMITS` | Comma-separated rate limits | `60,30,60` |
| `DG_OPENAI_WEIGHTS` | Comma-separated weights | `1,2,1` |

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
diffgentor edit --backend openai \
    --timeout 120 \
    --api_max_retries 3 \
    --max_global_workers 32 \
    --num_processes 8 \
    --input data.csv
```

## Examples

### Basic Editing

```bash
diffgentor edit --backend openai \
    --model_name gpt-image-1 \
    --input data.csv
```

### Multiple API Keys (Load Balancing)

```bash
# Use multiple keys for higher throughput
DG_OPENAI_API_KEYS=key1,key2,key3 \
DG_OPENAI_RATE_LIMIT=60 \
diffgentor edit --backend openai \
    --model_name gpt-image-1 \
    --input data.csv
```

### Multiple Endpoints

```bash
# Use multiple proxy endpoints
DG_OPENAI_ENDPOINTS=http://proxy1.com,http://proxy2.com \
DG_OPENAI_API_KEYS=key1,key2 \
DG_OPENAI_RATE_LIMITS=30,60 \
DG_OPENAI_WEIGHTS=1,2 \
diffgentor edit --backend openai \
    --input data.csv
```

### With Timeout and Retry

```bash
DG_OPENAI_TIMEOUT=120 \
DG_OPENAI_MAX_RETRIES=3 \
DG_OPENAI_RETRY_DELAY=2.0 \
diffgentor edit --backend openai \
    --input data.csv
```

### High Concurrency

```bash
DG_OPENAI_API_KEYS=key1,key2,key3,key4 \
DG_OPENAI_MAX_WORKERS=8 \
diffgentor edit --backend openai \
    --input large_dataset.csv
```

### Process Specific Rows

```bash
diffgentor edit --backend openai \
    --model_name gpt-image-1 \
    --input data.csv \
    --filter_rows 0:50
```

### Multi-Image Editing (gpt-image-1+)

`gpt-image-1` supports multiple input images for combining/compositing:

```csv
image_url,instruction
"img1.jpg,img2.jpg,img3.jpg","Combine these items into a gift basket"
"/path/to/a.png|/path/to/b.png","Merge these two product photos"
```

```bash
diffgentor edit --backend openai \
    --model_name gpt-image-1.5 \
    --input multi_image_data.csv
```

## Input Format

CSV with image URL and instruction:

```csv
image_url,instruction
https://example.com/image.jpg,"Add a red hat to the person"
/local/path/image.png,"Change the background to a beach"
```

## API Behavior

The OpenAI backend:

1. Downloads/loads input images
2. Sends image + instruction to OpenAI API
3. Downloads and saves the result
4. Handles rate limits with automatic retry (if configured)

## Rate Limiting

Built-in rate limiter using sliding window (per endpoint):

```bash
# Limit to 60 requests per minute per endpoint
DG_OPENAI_RATE_LIMIT=60 diffgentor edit --backend openai \
    --input large_dataset.csv
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
DG_OPENAI_ENDPOINTS=http://proxy1.com,http://proxy2.com \
DG_OPENAI_WEIGHTS=1,2 \
diffgentor edit --backend openai --input data.csv
```

## Cost Considerations

OpenAI API usage is billed. Check [OpenAI pricing](https://openai.com/pricing) for current rates.

Tips to manage costs:
- Test with small datasets first (`--filter_rows 0:10`)
- Use `--resume` to avoid re-processing
- Monitor your OpenAI usage dashboard
- Use multiple keys to avoid individual rate limits

## Notes

- **No Local GPU Required**: All processing happens on OpenAI's servers
- **Network Dependent**: Requires stable internet connection
- **API Key Security**: Never commit your API key to version control
- **Timeout**: Default 5 minutes; adjust for slow networks
