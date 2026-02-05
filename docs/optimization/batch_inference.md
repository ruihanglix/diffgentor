# Batch Inference

Batch inference allows processing multiple prompts or images in a single forward pass, significantly improving throughput compared to sequential processing.

## Overview

| Component | Batch Support | Notes |
|-----------|---------------|-------|
| T2I (diffusers) | ✓ | Full batch support |
| T2I (xDiT) | ✓ | Full batch support with multi-GPU |
| Editing (diffusers) | ✓/⚠️ | Model-dependent |
| API backends | ⚠️ | Concurrent requests instead |

## How Batch Inference Works

### Sequential vs Batch Processing

**Sequential Processing:**
```
Prompt 1 → Generate → Image 1
Prompt 2 → Generate → Image 2
Prompt 3 → Generate → Image 3
Time: 3 × single_image_time
```

**Batch Processing:**
```
[Prompt 1, Prompt 2, Prompt 3] → Generate → [Image 1, Image 2, Image 3]
Time: ~1.3 × single_image_time (amortized overhead)
```

Batch processing is more efficient because:
1. **Reduced overhead**: Model initialization, memory transfers happen once
2. **Better GPU utilization**: More parallel computation
3. **Optimized memory access**: Better cache utilization

## Configuration

### CLI Usage

```bash
# T2I with batch_size=4
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.jsonl \
    --batch_size 4

# Editing with batch_size=2
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --batch_size 2
```

### Configuration Classes

```python
# From diffgentor/config.py
@dataclass
class BackendConfig:
    batch_size: int = 1  # Batch size for batch inference

@dataclass
class T2IConfig:
    batch_size: int = 1

@dataclass
class EditingConfig:
    batch_size: int = 1
```

## T2I Batch Inference

### Diffusers Backend

The diffusers backend supports full batch inference for text-to-image generation.

```bash
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.jsonl \
    --batch_size 4 \
    --output_dir ./output
```

**Implementation flow:**
1. Worker loads prompts from file
2. Prompts are grouped into batches of `batch_size`
3. Each batch is processed in a single pipeline call
4. Images are saved with proper indexing

### xDiT Backend

xDiT backend provides enhanced batch support with multi-GPU parallelism.

```bash
export DG_XDIT_ULYSSES_DEGREE=4
export DG_XDIT_DATA_PARALLEL_DEGREE=2

diffgentor t2i --backend xdit \
    --model_name FLUX.1-dev \
    --prompts_file prompts.jsonl \
    --batch_size 8 \
    --num_gpus 8
```

**xDiT-specific features:**
- Runtime state configuration for optimal batching
- Data parallelism for distributing samples across GPUs
- Automatic batch size optimization based on parallel configuration

**Implementation details:**

```python
# From diffgentor/backends/t2i/xdit_backend.py
def generate(self, prompt, ...):
    batch_size = self.backend_config.batch_size
    
    # Configure xDiT runtime state for batch processing
    if self._runtime_state is not None:
        self._runtime_state.set_input_parameters(
            batch_size=min(batch_size, len(prompts)),
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )
    
    # Process in batches
    if batch_size >= len(prompts):
        # Single batch
        output = self.pipe(prompt=prompts, ...)
        all_images.extend(output.images)
    else:
        # Multiple batches
        for batch_idx in range(num_batches):
            batch_prompts = prompts[start_idx:end_idx]
            output = self.pipe(prompt=batch_prompts, ...)
            all_images.extend(output.images)
```

### Prompts File Format

Batch processing works best with a prompts file:

**JSONL format:**
```jsonl
{"prompt": "A serene mountain landscape at sunset"}
{"prompt": "A futuristic cityscape with flying cars"}
{"prompt": "A cozy cabin in a snowy forest"}
```

**CSV format:**
```csv
prompt
"A serene mountain landscape at sunset"
"A futuristic cityscape with flying cars"
"A cozy cabin in a snowy forest"
```

## Editing Batch Inference

### Batch Support by Model

Not all editing models support batch inference. The `ModelStrategy` system defines this:

| Model Type | Batch Support | Reason |
|------------|---------------|--------|
| qwen (Qwen-Image-Edit-Plus) | ✗ | Multi-image input (batch_disabled) |
| qwen_singleimg | ✓ | Single image input, full batch support |
| flux2 | ✗ | Multi-image, shared_image_batch mode |
| flux2_klein | ✗ | Multi-image, shared_image_batch mode |
| flux1_kontext | ✗ | shared_image_batch mode |
| longcat | ✗ | batch_disabled |
| glm_image | ✗ | Multi-image input (batch_disabled) |

**Why multi-image models don't support batch?**

For models like Qwen-Image-Edit-Plus that accept multiple input images per edit request, each sample already has a variable number of images. Batching multiple such samples would require nested batching (batch of batches), which diffusers pipelines don't natively support.

### Strategy Configuration

```python
# From diffgentor/backends/editing/strategies/base.py
@dataclass
class ModelConfig:
    batch_disabled: bool = False      # Completely disable batch
    shared_image_batch: bool = False  # Same image, multiple instructions
    multi_image: bool = False         # Multiple input images per request
```

**Batch inference logic:**

```python
# From diffgentor/backends/editing/diffusers_editing.py
def batch_edit(self, batch_images, batch_instructions, ...):
    # Check if model supports batch
    if not self.strategy.supports_batch:
        print_rank0("Pipeline does not support batch. Falling back to sequential.")
        return self._sequential_edit(...)
    
    # True batch inference
    return self._batch_edit_impl(...)
```

### Usage Example

```bash
diffgentor edit --backend diffusers \
    --model_name Qwen/Qwen-Image-Edit-2511 \
    --input data.csv \
    --batch_size 4 \
    --output_dir ./output
```

**Input CSV format:**
```csv
input_images,instruction
image1.png,"Make the sky more blue"
image2.png,"Add a rainbow"
image3.png,"Convert to oil painting style"
image4.png,"Make it look like night time"
```

### Batch Edit Implementation

```python
# From diffgentor/backends/editing/diffusers_editing.py
def _batch_edit_impl(self, batch_images, batch_instructions, ...):
    batch_size = self.backend_config.batch_size
    
    # Normalize images
    normalized_images = []
    for img in batch_images:
        if isinstance(img, Image.Image):
            normalized_images.append(img)
        else:
            normalized_images.append(img[0] if img else None)
    
    all_images = []
    num_batches = (len(batch_instructions) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(batch_instructions))
        
        batch_prompts = batch_instructions[start_idx:end_idx]
        batch_imgs = normalized_images[start_idx:end_idx]
        
        pipe_kwargs = {
            "prompt": batch_prompts,
            "image": batch_imgs,
            "num_inference_steps": num_inference_steps,
            ...
        }
        
        output = self.pipe(**pipe_kwargs)
        all_images.extend(output.images)
    
    return all_images
```

## Worker Batch Processing

Workers handle batch processing at a higher level, coordinating between data loading and backend inference.

### Edit Worker

```python
# From diffgentor/workers/edit_worker.py (conceptual)
for batch_start in range(0, total, config.batch_size):
    batch_end = min(batch_start + config.batch_size, total)
    batch_tasks = pending_tasks[batch_start:batch_end]
    
    self._log(
        f"Processing batch "
        f"{batch_start // config.batch_size + 1}/"
        f"{(total + config.batch_size - 1) // config.batch_size}..."
    )
    
    # Process batch
    results = backend.edit_batch(batch_tasks, **kwargs)
    
    # Save results
    for idx, image in results:
        save_image(image, output_path)
```

### T2I Worker

```python
# From diffgentor/workers/t2i_worker.py (conceptual)
prompts = load_prompts(config.prompts_file)

for batch_start in range(0, len(prompts), config.batch_size):
    batch_prompts = prompts[batch_start:batch_start + config.batch_size]
    
    images = backend.generate(prompt=batch_prompts, ...)
    
    for i, image in enumerate(images):
        save_image(image, get_output_path(batch_start + i))
```

## Error Handling

Batch processing includes fallback mechanisms for robustness:

```python
# From diffgentor/backends/editing/diffusers_editing.py
def edit_batch(self, batch_data, **kwargs):
    try:
        # Try batch inference
        return self.batch_edit(batch_images, batch_instructions, **kwargs)
    except Exception as e:
        log_error(EditingError("Batch inference failed", cause=e))
        print_rank0("Falling back to sequential processing")
        
        # Fallback to sequential
        results = []
        for images, instruction, idx in batch_data:
            try:
                edited = self.edit(images, instruction, **kwargs)
                results.append((idx, edited[0] if edited else None))
            except Exception as e2:
                results.append((idx, None))
        return results
```

## Memory Considerations

Batch inference increases memory usage proportionally. Consider these guidelines:

### VRAM Requirements

| Batch Size | VRAM Multiplier | Recommended VRAM |
|------------|-----------------|------------------|
| 1 | 1.0x | 16GB |
| 2 | ~1.5x | 24GB |
| 4 | ~2.5x | 40GB |
| 8 | ~4x | 80GB |

*Actual multiplier depends on model architecture and image resolution*

### Combining with Memory Optimizations

```bash
# Large batch with memory optimizations
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.jsonl \
    --batch_size 4 \
    --enable_vae_slicing \
    --torch_dtype bfloat16
```

**Note:** VAE slicing is especially helpful with batch inference as it processes VAE encoding/decoding sequentially within a batch.

## Performance Tuning

### Finding Optimal Batch Size

1. **Start small**: Begin with batch_size=2
2. **Monitor memory**: Use `nvidia-smi` to track VRAM usage
3. **Check throughput**: Measure images/second, not just time per image
4. **Find the sweet spot**: Usually where VRAM is 80-90% utilized

### Throughput Calculation

```
Throughput = batch_size × num_batches / total_time
```

Example:
- batch_size=4
- 100 images total → 25 batches
- Total time: 500 seconds
- Throughput: 100 / 500 = 0.2 images/second

### Best Practices

1. **Match batch size to VRAM** - Don't over-allocate
2. **Use VAE slicing** - Reduces peak memory during decoding
3. **Consider data parallelism** - For multi-GPU setups
4. **Profile first** - Measure before optimizing

```bash
# Optimal configuration for 40GB GPU
diffgentor t2i --backend diffusers \
    --model_name black-forest-labs/FLUX.1-dev \
    --prompts_file prompts.jsonl \
    --batch_size 4 \
    --enable_vae_slicing \
    --attention_backend flash \
    --cache_type first_block_cache
```

## API Backend Batching

API backends (OpenAI, Google GenAI) don't support true batch inference but use concurrent requests:

```python
# Conceptual - API backends use thread pool
with ThreadPoolExecutor(max_workers=num_processes) as executor:
    futures = [
        executor.submit(api_call, prompt)
        for prompt in prompts
    ]
    results = [f.result() for f in futures]
```

Configure concurrency via:
- `--num_processes`: Number of concurrent workers
- `--max_global_workers`: Maximum concurrent API requests

## Distributed Batch Processing

For very large batch jobs across multiple nodes:

```bash
# Node 0
diffgentor t2i --backend diffusers \
    --prompts_file prompts.jsonl \
    --batch_size 4 \
    --node_rank 0 \
    --num_nodes 4

# Node 1
diffgentor t2i --backend diffusers \
    --prompts_file prompts.jsonl \
    --batch_size 4 \
    --node_rank 1 \
    --num_nodes 4
```

Each node processes a shard of the prompts, enabling massive throughput across clusters.

## Summary

| Scenario | Recommended batch_size | Additional Settings |
|----------|----------------------|---------------------|
| 16GB VRAM, single GPU | 1-2 | `--enable_vae_slicing` |
| 24GB VRAM, single GPU | 2-4 | `--enable_vae_slicing` |
| 40GB VRAM, single GPU | 4-8 | `--attention_backend flash` |
| 80GB VRAM, single GPU | 8-16 | All speed optimizations |
| Multi-GPU (xDiT) | 4-8 per GPU | Data parallelism |
| API backends | N/A | Use `--num_processes` |
