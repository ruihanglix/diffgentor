# Speed Optimizations

This guide covers all speed optimization techniques available in Diffgentor to accelerate inference.

## Overview

| Optimization | Speed Boost | Memory Impact | Compatibility |
|--------------|-------------|---------------|---------------|
| TF32 | ✓ | None | Ampere+ GPUs |
| torch.compile | ✓✓✓ | Slight increase | PyTorch 2.0+ |
| Flash Attention | ✓✓ | Slight reduction | Modern GPUs |
| Sage Attention | ✓✓ | Slight reduction | Modern GPUs |
| xFormers | ✓✓ | Reduction | Wide |
| Fuse QKV | ✓ | None | Most models |
| Cache Acceleration | ✓✓ | Slight increase | Model-specific |

## TF32 (TensorFloat-32)

TF32 is a math mode available on Ampere and newer NVIDIA GPUs that provides significant speedup for matrix operations with minimal precision loss.

### How It Works

TF32 uses 19-bit precision (10-bit mantissa, 8-bit exponent, 1-bit sign) instead of FP32's 32 bits, allowing faster tensor core operations while maintaining numerical stability.

### Usage

TF32 is **enabled by default** in Diffgentor. To explicitly control it:

```bash
# Enabled (default)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --enable_tf32

# Disabled
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --no-enable_tf32
```

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class TF32Optimizer(Optimizer):
    def apply(self, pipe, config):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return pipe
```

## torch.compile

PyTorch 2.0's `torch.compile` optimizes the model graph for faster execution through kernel fusion, memory optimization, and specialized kernels.

### How It Works

1. **Graph Capture**: Traces the model to create a computation graph
2. **Optimization**: Applies optimizations like operator fusion
3. **Code Generation**: Generates optimized kernels using TorchInductor

### Usage

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_compile
```

**Note:** First run will be slow due to compilation. Subsequent runs are significantly faster.

### Compile Modes

| Mode | Speed | Compile Time | Memory |
|------|-------|--------------|--------|
| `default` | Moderate | Fast | Low |
| `reduce-overhead` | High | Medium | Medium |
| `max-autotune` | Highest | Slow | Higher |
| `max-autotune-no-cudagraphs` | High | Slow | Medium |

Default mode in Diffgentor: `max-autotune-no-cudagraphs`

### Configuration

Environment variables for fine-tuning:

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_COMPILE_MODE` | Compile mode | `max-autotune-no-cudagraphs` |
| `DG_COMPILE_FULLGRAPH` | Use fullgraph mode | `false` |
| `DG_COMPILE_COMPONENTS` | Components to compile | `transformer` |

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class CompileOptimizer(Optimizer):
    def apply(self, pipe, config):
        # Configure inductor for optimal performance
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True

        for component_name in config.compile_components:
            component = getattr(pipe, component_name, None)
            if component is None:
                continue
            
            component.to(memory_format=torch.channels_last)
            compiled = torch.compile(
                component,
                mode=config.compile_mode,
                fullgraph=config.compile_fullgraph,
            )
            setattr(pipe, component_name, compiled)
        return pipe
```

### Best Practices

- **Warmup**: Run a few inference passes to complete compilation
- **Batch Size**: Keep consistent batch sizes to avoid recompilation
- **Model Changes**: Recompilation triggered by model modifications

## Attention Backends

Optimized attention implementations can significantly speed up transformer inference.

### Available Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `flash` | Flash Attention 2 | flash-attn package |
| `flash_3` | Flash Attention 3 | flash-attn 3.x |
| `sage` | Sage Attention | sageattention package |
| `xformers` | xFormers memory efficient | xformers package |

See https://huggingface.co/docs/diffusers/optimization/attention_backends for more details.

### Usage

```bash
# Flash Attention (recommended)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --attention_backend flash

# Sage Attention
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --attention_backend sage

# xFormers
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --attention_backend xformers
```

### Alternative: xFormers Flag

For backwards compatibility, xFormers can also be enabled via:

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --enable_xformers
```

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class AttentionBackendOptimizer(Optimizer):
    def apply(self, pipe, config):
        component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
        if component and hasattr(component, "set_attention_backend"):
            component.set_attention_backend(config.attention_backend)
        return pipe

@register_optimizer
class XFormersOptimizer(Optimizer):
    def apply(self, pipe, config):
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
        return pipe
```

### Comparison

| Backend | Speed | Memory | Stability |
|---------|-------|--------|-----------|
| Flash Attention | ★★★★★ | ★★★★ | ★★★★★ |
| Sage Attention | ★★★★☆ | ★★★★ | ★★★★☆ |
| xFormers | ★★★★☆ | ★★★★★ | ★★★★★ |
| Default | ★★★☆☆ | ★★★☆☆ | ★★★★★ |

## Fuse QKV Projections

Fusing Query, Key, and Value projections into a single operation reduces memory access overhead.

### Usage

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --enable_fuse_qkv
```

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class FuseQKVOptimizer(Optimizer):
    def apply(self, pipe, config):
        if hasattr(pipe, "fuse_qkv_projections"):
            pipe.fuse_qkv_projections()
        return pipe
```

## Cache Acceleration

Cache acceleration techniques skip redundant computations during the diffusion process by caching and reusing intermediate results.

### Supported Cache Types

| Cache Type | Best For | Speed Boost |
|------------|----------|-------------|
| `deep_cache` | SD/SDXL UNet models | ★★★★☆ |
| `first_block_cache` | DiT models (FLUX, SD3) | ★★★★☆ |
| `pab` | DiT models | ★★★☆☆ |
| `faster_cache` | General | ★★★★★ |
| `taylor_seer` | General | ★★★☆☆ |
| `cache_dit` | DiT models | ★★★★☆ |

### Usage

```bash
# DeepCache (for SD/SDXL)
diffgentor t2i --backend diffusers \
    --model_name stabilityai/stable-diffusion-xl-base-1.0 \
    --cache_type deep_cache

# First Block Cache (for FLUX/SD3)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --cache_type first_block_cache

# Pyramid Attention Broadcast (PAB)
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --cache_type pab

# FasterCache
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --cache_type faster_cache

# CacheDiT
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --cache_type cache_dit
```

### Cache Configuration

Fine-tune cache behavior via environment variables:

| Variable | Cache Type | Description | Default |
|----------|------------|-------------|---------|
| `DG_CACHE_INTERVAL` | deep_cache, taylor_seer | Steps between cache updates | 3 |
| `DG_CACHE_BRANCH_ID` | deep_cache | UNet branch to cache | 0 |
| `DG_FBC_THRESHOLD` | first_block_cache | Similarity threshold | 0.2 |
| `DG_PAB_SPATIAL_SKIP` | pab | Spatial attention skip range | 2 |
| `DG_PAB_TIMESTEP_SKIP` | pab | Timestep skip range | (100, 800) |

### Implementation Details

```python
# From diffgentor/optimizations/optimizers.py
@register_optimizer
class CacheOptimizer(Optimizer):
    def apply(self, pipe, config):
        cache_type = config.cache_type.lower()
        cache_handlers = {
            "deep_cache": self._apply_deep_cache,
            "first_block_cache": self._apply_first_block_cache,
            "pab": self._apply_pab_cache,
            "faster_cache": self._apply_faster_cache,
            "taylor_seer": self._apply_taylor_seer_cache,
            "cache_dit": self._apply_cache_dit,
        }
        handler = cache_handlers.get(cache_type)
        if handler:
            return handler(pipe, config.cache_config)
        return pipe
```

### DeepCache

Best for UNet-based models (SD, SDXL). Caches UNet branch outputs and reuses them.

```python
def _apply_deep_cache(self, pipe, cache_config):
    from DeepCache import DeepCacheSDHelper
    helper = DeepCacheSDHelper(pipe=pipe)
    helper.set_params(
        cache_interval=cache_config.get("cache_interval", 3),
        cache_branch_id=cache_config.get("cache_branch_id", 0),
    )
    helper.enable()
    return pipe
```

### First Block Cache

Best for DiT models (FLUX, SD3). Caches first transformer block outputs when similarity is high.

```python
def _apply_first_block_cache(self, pipe, cache_config):
    from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
    fbc_config = FirstBlockCacheConfig(
        threshold=cache_config.get("threshold", 0.2)
    )
    component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if component:
        apply_first_block_cache(component, fbc_config)
    return pipe
```

### Pyramid Attention Broadcast (PAB)

Broadcasts attention across timesteps to reduce computation.

```python
def _apply_pab_cache(self, pipe, cache_config):
    from diffusers import PyramidAttentionBroadcastConfig
    pab_config = PyramidAttentionBroadcastConfig(
        spatial_attention_block_skip_range=cache_config.get("spatial_skip_range", 2),
        spatial_attention_timestep_skip_range=cache_config.get("timestep_skip_range", (100, 800)),
        current_timestep_callback=lambda: pipe.current_timestep,
    )
    component = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    if component and hasattr(component, "enable_cache"):
        component.enable_cache(pab_config)
    return pipe
```

## Combining Speed Optimizations

### Maximum Speed Configuration

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --torch_dtype bfloat16 \
    --enable_compile \
    --attention_backend flash \
    --cache_type first_block_cache \
    --enable_fuse_qkv
```

### Balanced Speed + Compatibility

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --torch_dtype bfloat16 \
    --attention_backend flash \
    --cache_type first_block_cache
```

## Compatibility Matrix

| Optimization | torch.compile | Flash Attn | xFormers | Cache | xDiT |
|--------------|---------------|------------|----------|-------|------|
| TF32 | ✓ | ✓ | ✓ | ✓ | ✓ |
| torch.compile | - | ✓ | ✓ | ✓ | ✗ |
| Flash Attention | ✓ | - | ✗ | ✓ | ✓ |
| xFormers | ✓ | ✗ | - | ✓ | ✓ |
| Cache | ✓ | ✓ | ✓ | - | ⚠️ |
| Fuse QKV | ✓ | ✓ | ✓ | ✓ | ✓ |

⚠️ = Limited support, test with your specific model

## Benchmarking Tips

1. **Warmup**: Always run a few inference passes before timing
2. **Consistent Inputs**: Use same batch size, resolution, and steps
3. **Multiple Runs**: Average over multiple runs for reliable numbers
4. **Exclude First Run**: torch.compile first run includes compilation time

```bash
# Example benchmark script
for i in {1..10}; do
    diffgentor t2i --backend diffusers \
        --model_name black-forest-labs/FLUX.1-dev \
        --prompt "A landscape" \
        --enable_compile \
        --attention_backend flash \
        --cache_type first_block_cache
done
```
