# Memory Optimizations

This guide covers all memory optimization techniques available in Diffgentor to reduce VRAM usage during inference.

## Overview

| Optimization | Memory Reduction | Speed Impact | Use Case |
|--------------|------------------|--------------|----------|
| VAE Slicing | ✓✓ | Slight slowdown | Batch VAE processing |
| VAE Tiling | ✓✓✓ | Moderate slowdown | Large resolution images |
| Model CPU Offload | ✓✓✓✓ | Significant slowdown | VRAM limited |
| Sequential CPU Offload | ✓✓✓✓✓ | Very significant slowdown | Extremely limited VRAM |
| Group Offloading | ✓✓✓ | Moderate slowdown | Fine-grained control |
| Layerwise Casting | ✓✓✓ | Minimal | FP8 storage optimization |
| Data Type | ✓✓ | Varies | General memory reduction |

## VAE Slicing

VAE slicing processes VAE encoding/decoding in slices to reduce peak memory. This is particularly useful when generating multiple images in a batch.

### How It Works

Instead of processing the entire batch at once, the VAE processes images one at a time, reducing peak memory usage proportionally to batch size.

### Usage

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_vae_slicing
```

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class VAESlicingOptimizer(Optimizer):
    def apply(self, pipe, config):
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        return pipe
```

### When to Use

- ✓ When generating multiple images (`--batch_size > 1`)
- ✓ When VRAM is limited
- ✗ Not needed for single image generation

## VAE Tiling

VAE tiling processes images in tiles for very large resolutions, enabling generation of images larger than VRAM would normally allow.

### How It Works

The VAE divides large images into overlapping tiles, processes each tile independently, and then stitches them together. This allows processing of arbitrarily large images.

### Usage

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_vae_tiling \
    --height 2048 --width 2048
```

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class VAETilingOptimizer(Optimizer):
    def apply(self, pipe, config):
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        return pipe
```

### When to Use

- ✓ When generating high-resolution images (> 1024x1024)
- ✓ When generating images that exceed VRAM capacity
- ✗ Minor overhead for standard resolutions

## CPU Offload

CPU offload moves model components to CPU RAM when not in use, significantly reducing VRAM usage at the cost of inference speed.

### Model CPU Offload

Offloads entire model components (transformer, text encoders) to CPU when not in use.

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_cpu_offload
```

### Sequential CPU Offload

More aggressive offloading that moves each layer to GPU only when needed.

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_sequential_cpu_offload
```

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class ModelCPUOffloadOptimizer(Optimizer):
    def should_apply(self, config):
        # Don't apply if sequential offload is also enabled
        return config.enable_cpu_offload and not config.enable_sequential_cpu_offload
    
    def apply(self, pipe, config):
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        return pipe

@register_optimizer
class SequentialCPUOffloadOptimizer(Optimizer):
    def apply(self, pipe, config):
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            pipe.enable_sequential_cpu_offload()
        return pipe
```

### Comparison

| Feature | Model Offload | Sequential Offload |
|---------|--------------|-------------------|
| Memory Savings | High | Very High |
| Speed Impact | Moderate | Significant |
| Granularity | Component-level | Layer-level |
| Best For | 12-16GB VRAM | 8-12GB VRAM |

### When to Use

- ✓ `--enable_cpu_offload`: When VRAM is 12-16GB
- ✓ `--enable_sequential_cpu_offload`: When VRAM is < 12GB
- ✗ Don't use both simultaneously

## Group Offloading

Group offloading provides fine-grained control over memory offloading using async streaming for better performance.

### Usage

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_group_offloading
```

### Configuration

The offload type can be configured via environment variable:

| Variable | Description | Values |
|----------|-------------|--------|
| `DG_GROUP_OFFLOAD_TYPE` | Offloading granularity | `leaf_level`, `block_level` |

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class GroupOffloadingOptimizer(Optimizer):
    def apply(self, pipe, config):
        if hasattr(pipe, "enable_group_offload"):
            pipe.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type=config.group_offload_type,  # leaf_level or block_level
                use_stream=True,  # Async streaming for better performance
            )
        return pipe
```

### Offload Types

- **`leaf_level`**: Most fine-grained, highest memory savings, slower
- **`block_level`**: Offloads entire transformer blocks, balance of speed/memory

## Layerwise Casting

Layerwise casting stores model weights in a lower precision format (e.g., FP8) while computing in a higher precision (e.g., BF16), reducing memory without significant quality loss.

### Usage

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_layerwise_casting
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_STORAGE_DTYPE` | Storage precision | `float8_e4m3fn` |
| `DG_COMPUTE_DTYPE` | Compute precision | `bfloat16` |

### Supported Storage Dtypes

- `float8_e4m3fn` - 8-bit floating point (recommended)
- `float8_e5m2` - 8-bit floating point (alternative)
- `float16` - 16-bit floating point
- `bfloat16` - 16-bit brain floating point

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class LayerwiseCastingOptimizer(Optimizer):
    def apply(self, pipe, config):
        dtype_map = {
            "float8_e4m3fn": torch.float8_e4m3fn,
            "float8_e5m2": torch.float8_e5m2,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        storage_dtype = dtype_map.get(config.storage_dtype)
        compute_dtype = dtype_map.get(config.compute_dtype)
        
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "enable_layerwise_casting"):
            component.enable_layerwise_casting(
                storage_dtype=storage_dtype,
                compute_dtype=compute_dtype,
            )
        return pipe
```

## Data Type Selection

Using lower precision data types reduces memory usage and can improve speed on compatible hardware.

### Available Options

| Dtype | Memory | Speed | Compatibility |
|-------|--------|-------|---------------|
| `float32` | Highest | Baseline | Universal |
| `float16` | Half | Faster | Wide |
| `bfloat16` | Half | Faster | Ampere+ GPUs |

### Usage

```bash
# bfloat16 (recommended for modern GPUs)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16

# float16 (wider compatibility)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype float16
```

### Recommendations

- **Ampere+ GPUs (RTX 30xx, A100, etc.)**: Use `bfloat16`
- **Older GPUs (RTX 20xx, V100)**: Use `float16`
- **CPU or debugging**: Use `float32`

## Combining Memory Optimizations

### Extreme Low VRAM (6-8GB)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --torch_dtype float16 \
    --enable_sequential_cpu_offload \
    --enable_vae_slicing \
    --enable_vae_tiling
```

### Low VRAM (8-12GB)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --torch_dtype float16 \
    --enable_cpu_offload \
    --enable_vae_slicing
```

### Medium VRAM (16-24GB)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_vae_slicing
```

### High Resolution Large Images

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_vae_tiling \
    --height 4096 --width 4096 \
    --enable_cpu_offload
```

## Compatibility Matrix

| Optimization | CPU Offload | Sequential Offload | Group Offload | xDiT |
|--------------|-------------|-------------------|---------------|------|
| VAE Slicing | ✓ | ✓ | ✓ | ✓ |
| VAE Tiling | ✓ | ✓ | ✓ | ✓ |
| CPU Offload | - | ✗ | ✗ | ✗ |
| Sequential Offload | ✗ | - | ✗ | ✗ |
| Group Offload | ✗ | ✗ | - | ✗ |
| Layerwise Casting | ✓ | ✓ | ✓ | ✓ |
