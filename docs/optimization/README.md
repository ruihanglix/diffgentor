# Optimization Guide

Diffgentor provides a comprehensive optimization framework to improve inference speed and reduce memory usage. This guide covers all available optimization techniques.

## Quick Reference

| Optimization | Memory Reduction | Speed Impact | CLI Flag |
|--------------|------------------|--------------|----------|
| VAE Slicing | ✓✓ | Slight slowdown | `--enable_vae_slicing` |
| VAE Tiling | ✓✓✓ | Moderate slowdown | `--enable_vae_tiling` |
| CPU Offload | ✓✓✓✓ | Significant slowdown | `--enable_cpu_offload` |
| Sequential CPU Offload | ✓✓✓✓✓ | Very significant slowdown | `--enable_sequential_cpu_offload` |
| Group Offloading | ✓✓✓ | Moderate slowdown | `--enable_group_offloading` |
| Layerwise Casting | ✓✓✓ | Minimal | `--enable_layerwise_casting` |
| torch.compile | ✗ | ✓✓✓ Speedup | `--enable_compile` |
| Flash Attention | ✓ | ✓✓ Speedup | `--attention_backend flash` |
| Sage Attention | ✓ | ✓✓ Speedup | `--attention_backend sage` |
| xFormers | ✓ | ✓✓ Speedup | `--enable_xformers` |
| Fuse QKV | ✗ | ✓ Speedup | `--enable_fuse_qkv` |
| TF32 | ✗ | ✓ Speedup | `--enable_tf32` (default) |
| Cache Acceleration | ✗ | ✓✓ Speedup | `--cache_type <type>` |
| Batch Inference | ✗ | ✓✓✓ Throughput | `--batch_size N` |
| Multi-GPU (xDiT) | ✗ | ✓✓✓✓ Speedup | `--backend xdit` |

## Documentation Structure

- [Memory Optimizations](memory.md) - VAE slicing/tiling, CPU offload, group offloading, layerwise casting
- [Speed Optimizations](speed.md) - torch.compile, attention backends, cache acceleration
- [Multi-GPU (xDiT)](multi_gpu.md) - Distributed inference with xDiT parallelism
- [Batch Inference](batch_inference.md) - Batched generation for higher throughput

## Optimization Architecture

Diffgentor uses a modular optimizer framework. All optimizers are registered in `diffgentor/optimizations/optimizers.py`:

```python
from diffgentor.optimizations.base import Optimizer, register_optimizer

@register_optimizer
class MyOptimizer(Optimizer):
    @property
    def name(self) -> str:
        return "My Optimizer"
    
    def should_apply(self, config: OptimizationConfig) -> bool:
        return config.enable_my_optimization
    
    def apply(self, pipe: Any, config: OptimizationConfig) -> Any:
        # Apply optimization to pipeline
        return pipe
```

### Available Optimizers

| Optimizer Class | Description | Config Field |
|----------------|-------------|--------------|
| `TF32Optimizer` | Enable TF32 for Ampere+ GPUs | `enable_tf32` |
| `VAESlicingOptimizer` | VAE slicing for memory efficiency | `enable_vae_slicing` |
| `VAETilingOptimizer` | VAE tiling for large images | `enable_vae_tiling` |
| `SequentialCPUOffloadOptimizer` | Sequential CPU offload | `enable_sequential_cpu_offload` |
| `ModelCPUOffloadOptimizer` | Model-level CPU offload | `enable_cpu_offload` |
| `XFormersOptimizer` | xFormers memory efficient attention | `enable_xformers` |
| `FuseQKVOptimizer` | Fuse QKV projections | `enable_fuse_qkv` |
| `GroupOffloadingOptimizer` | Group offloading | `enable_group_offloading` |
| `LayerwiseCastingOptimizer` | Layerwise dtype casting | `enable_layerwise_casting` |
| `AttentionBackendOptimizer` | Attention backend selection | `attention_backend` |
| `CompileOptimizer` | torch.compile optimization | `enable_compile` |
| `CacheOptimizer` | Cache acceleration | `cache_type` |

## CLI Usage

### Using Individual Flags

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --enable_vae_slicing \
    --enable_compile \
    --attention_backend flash
```

### Using Optimization String

Combine multiple optimizations with a comma-separated string:

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --optimize "vae_slicing,compile,flash_attention"
```

## Configuration Classes

### OptimizationConfig

All optimization settings are defined in `diffgentor/config.py`:

```python
@dataclass
class OptimizationConfig:
    # Basic optimizations
    torch_dtype: str = "bfloat16"
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False
    enable_cpu_offload: bool = False
    enable_sequential_cpu_offload: bool = False
    enable_xformers: bool = False
    
    # Attention backend
    attention_backend: Optional[str] = None  # flash, sage, xformers
    
    # Compilation
    enable_compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_fullgraph: bool = False
    compile_components: List[str] = ["transformer"]
    
    # Cache acceleration
    cache_type: Optional[str] = None
    cache_config: Dict[str, Any] = {}
    
    # Memory optimization
    enable_group_offloading: bool = False
    group_offload_type: str = "leaf_level"
    enable_layerwise_casting: bool = False
    storage_dtype: str = "float8_e4m3fn"
    compute_dtype: str = "bfloat16"
    
    # Quantization
    quantization: Optional[str] = None
    quantization_components: List[str] = ["transformer"]
    
    # Other optimizations
    enable_fuse_qkv: bool = False
    enable_tf32: bool = True
```

## Recommended Configurations

### Low VRAM (8-12GB)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --torch_dtype float16 \
    --enable_cpu_offload \
    --enable_vae_slicing \
    --enable_vae_tiling
```

### Balanced (16-24GB)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_vae_slicing \
    --attention_backend flash
```

### High Performance (40GB+)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_compile \
    --attention_backend flash \
    --cache_type first_block_cache \
    --batch_size 4
```

### Maximum Throughput (Multi-GPU)

```bash
export DG_XDIT_ULYSSES_DEGREE=4
export DG_XDIT_DATA_PARALLEL_DEGREE=2
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --batch_size 8
```

## Combining Optimizations

Many optimizations can be combined for maximum effect:

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A detailed landscape" \
    --torch_dtype bfloat16 \
    --enable_compile \
    --enable_vae_slicing \
    --attention_backend flash \
    --cache_type first_block_cache \
    --batch_size 2
```

**Note:** Some combinations may not be compatible. Test with your specific setup.

### Compatibility Notes

| Optimization A | Optimization B | Compatible |
|---------------|----------------|------------|
| CPU Offload | Sequential CPU Offload | ✗ (use one) |
| torch.compile | xDiT | ✗ |
| Flash Attention | xFormers | ✗ (use one) |
| Group Offloading | CPU Offload | ✗ (use one) |
| Batch Inference | All memory opts | ✓ |
| Cache Acceleration | torch.compile | ✓ |
