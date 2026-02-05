# Multi-GPU Inference (xDiT)

Diffgentor supports distributed inference across multiple GPUs using xDiT (xfuser), enabling significant speedup for large models and high-throughput scenarios.

## Overview

xDiT provides multiple parallelism strategies that can be combined for optimal performance:

| Strategy | Description | Best For |
|----------|-------------|----------|
| Data Parallel | Process different samples on different GPUs | High throughput |
| Ulysses | Sequence parallelism (split sequence) | Large models |
| Ring | Sequence parallelism (ring communication) | Large models |
| PipeFusion | Pipeline parallelism | Very large models |
| CFG Parallel | Split CFG branches across GPUs | CFG-guided models |

## Prerequisites

Install xDiT:

```bash
pip install xfuser
```

## Basic Usage

```bash
# 4 GPUs with Ulysses parallelism
export DG_XDIT_ULYSSES_DEGREE=4
diffgentor t2i --backend xdit \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompt "A landscape" \
    --num_gpus 4
```

**Note:** xDiT requires `torchrun` for multi-GPU execution. Diffgentor handles this automatically.

## Environment Variables

xDiT parallelism parameters are configured via `DG_XDIT_*` environment variables, **not CLI arguments**.

| Variable | Description | Default |
|----------|-------------|---------|
| `DG_XDIT_DATA_PARALLEL_DEGREE` | Data parallelism degree | 1 |
| `DG_XDIT_ULYSSES_DEGREE` | Ulysses sequence parallelism degree | 1 |
| `DG_XDIT_RING_DEGREE` | Ring sequence parallelism degree | 1 |
| `DG_XDIT_PIPEFUSION_DEGREE` | PipeFusion parallelism degree | 1 |
| `DG_XDIT_USE_CFG_PARALLEL` | Enable CFG parallelism | `false` |

### Configuration via BackendConfig

These environment variables are accessed through `BackendConfig` properties:

```python
# From diffgentor/config.py
@dataclass
class BackendConfig:
    @property
    def data_parallel_degree(self) -> int:
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_DATA_PARALLEL_DEGREE", 1)

    @property
    def ulysses_degree(self) -> int:
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_ULYSSES_DEGREE", 1)

    @property
    def ring_degree(self) -> int:
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_RING_DEGREE", 1)

    @property
    def pipefusion_degree(self) -> int:
        from diffgentor.utils.env import get_env_int
        return get_env_int("XDIT_PIPEFUSION_DEGREE", 1)

    @property
    def use_cfg_parallel(self) -> bool:
        from diffgentor.utils.env import get_env_bool
        return get_env_bool("XDIT_USE_CFG_PARALLEL", False)
```

## Parallelism Strategies

### Data Parallelism

Process different prompts/samples on different GPUs simultaneously. Best for high-throughput batch processing.

```bash
export DG_XDIT_DATA_PARALLEL_DEGREE=4
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --prompts_file prompts.jsonl \
    --num_gpus 4 \
    --batch_size 4
```

**GPU Usage:** Each GPU processes `batch_size / data_parallel_degree` samples.

### Ulysses Parallelism

Split the sequence dimension across GPUs. Effective for models with long sequence lengths.

```bash
export DG_XDIT_ULYSSES_DEGREE=4
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4
```

**How it works:**
1. Splits sequence along the sequence dimension
2. All-to-all communication for attention
3. Reduces memory per GPU proportionally

### Ring Parallelism

Sequence parallelism using ring communication pattern. Lower communication overhead than Ulysses for certain configurations.

```bash
export DG_XDIT_RING_DEGREE=4
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4
```

**How it works:**
1. Splits sequence into chunks
2. Ring-based P2P communication
3. Good for high-bandwidth interconnects

### PipeFusion

Pipeline parallelism that distributes transformer layers across GPUs. Best for very large models that don't fit on a single GPU.

```bash
export DG_XDIT_PIPEFUSION_DEGREE=4
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4
```

**How it works:**
1. Distributes layers across GPUs
2. Activations flow through pipeline
3. Reduces memory per GPU significantly

### CFG Parallelism

Parallelize conditional and unconditional branches of Classifier-Free Guidance across GPUs.

```bash
export DG_XDIT_USE_CFG_PARALLEL=true
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 2
```

**How it works:**
1. One GPU handles conditional branch
2. Other GPU handles unconditional branch
3. Requires exactly 2 GPUs or combined with other strategies

## Hybrid Parallelism

Combine multiple strategies for maximum efficiency. The total GPU count must equal the product of all degrees.

### Example: 8 GPUs with Hybrid

```bash
# 8 GPUs: 2x Ulysses × 2x Ring × 2x Data
export DG_XDIT_ULYSSES_DEGREE=2
export DG_XDIT_RING_DEGREE=2
export DG_XDIT_DATA_PARALLEL_DEGREE=2

diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --batch_size 8
```

**GPU Allocation:** 2 × 2 × 2 = 8 GPUs

### Example: 4 GPUs with CFG + Ulysses

```bash
export DG_XDIT_USE_CFG_PARALLEL=true
export DG_XDIT_ULYSSES_DEGREE=2

diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4
```

**GPU Allocation:** 2 (CFG) × 2 (Ulysses) = 4 GPUs

## xDiT Backend Implementation

### Initialization

```python
# From diffgentor/backends/t2i/xdit_backend.py
class XDiTBackend(BaseBackend):
    def load_model(self, **kwargs):
        from xfuser import xFuserArgs, xDiTParallel
        from xfuser.core.distributed import (
            get_world_group,
            init_distributed_environment,
            get_runtime_state,
        )
        
        # Initialize distributed environment
        init_distributed_environment(
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
        
        # Create engine configuration
        engine_args = self._create_engine_args()
        self.engine_config, self.input_config = engine_args.create_config()
        
        # Load and wrap pipeline
        pipe = DiffusionPipeline.from_pretrained(self.model_name, ...)
        self.pipe = xDiTParallel(pipe, self.engine_config, self.input_config)
```

### Engine Arguments

```python
def _create_engine_args(self):
    from xfuser import xFuserArgs
    
    config = self.backend_config
    args_dict = {
        "model": self.model_name,
        "data_parallel_degree": config.data_parallel_degree,
        "ulysses_degree": config.ulysses_degree,
        "ring_degree": config.ring_degree,
        "pipefusion_parallel_degree": config.pipefusion_degree,
        "use_cfg_parallel": config.use_cfg_parallel,
    }
    return xFuserArgs(**args_dict)
```

### Batch Inference

xDiT backend fully supports batch inference:

```python
def generate(self, prompt, ...):
    batch_size = self.backend_config.batch_size
    
    # Update runtime state for xDiT
    if self._runtime_state is not None:
        self._runtime_state.set_input_parameters(
            batch_size=min(batch_size, len(prompts)),
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
    
    # Process in batches
    for batch_idx in range(num_batches):
        batch_prompts = prompts[start_idx:end_idx]
        output = self.pipe(prompt=batch_prompts, ...)
        all_images.extend(output.images)
```

## Recommended Configurations

### 2 GPUs - CFG Parallel

```bash
export DG_XDIT_USE_CFG_PARALLEL=true
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 2
```

### 4 GPUs - Ulysses

```bash
export DG_XDIT_ULYSSES_DEGREE=4
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4
```

### 8 GPUs - High Throughput

```bash
export DG_XDIT_ULYSSES_DEGREE=4
export DG_XDIT_DATA_PARALLEL_DEGREE=2
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 8 \
    --batch_size 8 \
    --prompts_file large_prompts.jsonl
```

### 16 GPUs - Maximum Scale

```bash
export DG_XDIT_ULYSSES_DEGREE=4
export DG_XDIT_RING_DEGREE=2
export DG_XDIT_DATA_PARALLEL_DEGREE=2
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 16 \
    --batch_size 16
```

## Optimization Compatibility

| Optimization | xDiT Compatible |
|--------------|-----------------|
| VAE Slicing | ✓ |
| VAE Tiling | ✓ |
| torch.compile | ✗ |
| Flash Attention | ✓ |
| xFormers | ✓ |
| Cache Acceleration | ⚠️ Limited |
| CPU Offload | ✗ |
| Batch Inference | ✓ |

### Pre-Wrap Optimizations

Some optimizations must be applied before xDiT wrapping:

```python
# From diffgentor/backends/t2i/xdit_backend.py
def _apply_pre_wrap_optimizations(self, pipe):
    config = self.optimization_config
    
    # TF32
    if config.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # VAE optimizations
    if config.enable_vae_slicing and hasattr(pipe, "vae"):
        pipe.vae.enable_slicing()
    
    if config.enable_vae_tiling and hasattr(pipe, "vae"):
        pipe.vae.enable_tiling()
```

## Troubleshooting

### Common Issues

1. **NCCL Errors**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=1  # If InfiniBand issues
   ```

2. **GPU Count Mismatch**
   - Ensure `num_gpus` equals product of all parallel degrees
   - Check with: `ulysses × ring × data × (2 if cfg_parallel else 1)`

3. **Out of Memory**
   - Reduce batch size
   - Increase sequence parallelism (ulysses/ring degree)
   - Use PipeFusion for very large models

4. **Slow Initialization**
   - Normal for first run due to distributed setup
   - Consider keeping model loaded for multiple inferences

### Debug Mode

```bash
export DG_DEBUG=true
export NCCL_DEBUG=INFO
diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --num_gpus 4
```

## Performance Tips

1. **Match batch size to data parallel degree** for optimal utilization
2. **Use Ulysses for FLUX/SD3** - DiT models benefit most from sequence parallelism
3. **PipeFusion for memory-constrained setups** - distributes layers across GPUs
4. **CFG parallel for low GPU counts** - easy 2x speedup with 2 GPUs
5. **Monitor GPU utilization** - use `nvidia-smi` to identify bottlenecks
