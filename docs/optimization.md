# Optimization Guide

Diffgentor provides a comprehensive optimization framework to improve inference speed and reduce memory usage.

**This document has been reorganized. Please see the following detailed guides:**

## Documentation

| Document | Description |
|----------|-------------|
| [Overview](optimization/README.md) | Quick reference, architecture, and recommended configurations |
| [Memory Optimizations](optimization/memory.md) | VAE slicing/tiling, CPU offload, group offloading, layerwise casting |
| [Speed Optimizations](optimization/speed.md) | torch.compile, attention backends, cache acceleration |
| [Multi-GPU (xDiT)](optimization/multi_gpu.md) | Distributed inference with xDiT parallelism |
| [Batch Inference](optimization/batch_inference.md) | Batched generation for higher throughput |

## Quick Reference

| Optimization | Memory Reduction | Speed Impact | CLI Flag |
|--------------|------------------|--------------|----------|
| VAE Slicing | ✓✓ | Slight slowdown | `--enable_vae_slicing` |
| VAE Tiling | ✓✓✓ | Moderate slowdown | `--enable_vae_tiling` |
| CPU Offload | ✓✓✓✓ | Significant slowdown | `--enable_cpu_offload` |
| torch.compile | ✗ | ✓✓✓ Speedup | `--enable_compile` |
| Flash Attention | ✓ | ✓✓ Speedup | `--attention_backend flash` |
| Cache Acceleration | ✗ | ✓✓ Speedup | `--cache_type <type>` |
| Batch Inference | ✗ | ✓✓✓ Throughput | `--batch_size N` |
| Multi-GPU (xDiT) | ✗ | ✓✓✓✓ Speedup | `--backend xdit` |

## Quick Start

### Memory-Constrained (8-16GB VRAM)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-schnell \
    --torch_dtype float16 \
    --enable_cpu_offload \
    --enable_vae_slicing
```

### Balanced Performance (24GB VRAM)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_vae_slicing \
    --attention_backend flash
```

### Maximum Speed (40GB+ VRAM)

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --torch_dtype bfloat16 \
    --enable_compile \
    --attention_backend flash \
    --cache_type first_block_cache \
    --batch_size 4
```

### Multi-GPU High Throughput

```bash
export DG_XDIT_ULYSSES_DEGREE=4
export DG_XDIT_DATA_PARALLEL_DEGREE=2
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --batch_size 8
```

## All Available Optimizations

### Memory Optimizations
- **VAE Slicing**: Process VAE in slices to reduce peak memory
- **VAE Tiling**: Tile VAE for large resolution images
- **CPU Offload**: Offload models to CPU when not in use
- **Sequential CPU Offload**: Fine-grained layer-level offloading
- **Group Offloading**: Async stream-based offloading
- **Layerwise Casting**: FP8 storage with higher precision compute

### Speed Optimizations
- **TF32**: TensorFloat-32 for Ampere+ GPUs (enabled by default)
- **torch.compile**: PyTorch 2.0 graph optimization
- **Flash Attention**: Optimized attention kernel
- **Sage Attention**: Alternative optimized attention
- **xFormers**: Memory efficient attention
- **Fuse QKV**: Fuse Query/Key/Value projections
- **Cache Acceleration**: Skip redundant diffusion computations
  - DeepCache
  - First Block Cache
  - PAB (Pyramid Attention Broadcast)
  - FasterCache
  - TaylorSeer
  - CacheDiT

### Throughput Optimizations
- **Batch Inference**: Process multiple prompts/images per forward pass
- **Multi-GPU (xDiT)**: Distributed inference across GPUs
  - Data Parallelism
  - Ulysses Parallelism
  - Ring Parallelism
  - PipeFusion
  - CFG Parallelism

For detailed documentation on each optimization, see the linked guides above.
