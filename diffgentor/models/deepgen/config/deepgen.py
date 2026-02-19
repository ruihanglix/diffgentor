# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen model configuration.

This config file defines the model architecture and hyperparameters for DeepGen.
Model paths (diffusion_model_path, ar_model_path) should be provided via
environment variables:
    - DG_DEEPGEN_DIFFUSION_MODEL_PATH: Path to diffusion model (SD3.5)
    - DG_DEEPGEN_AR_MODEL_PATH: Path to AR model (Qwen2.5-VL)

The config is loaded by diffgentor.models.deepgen.config.load_deepgen_config().
"""

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import SD3Transformer2DModel
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration


# Tokenizer configuration
# Note: pretrained_model_name_or_path will be set from DG_DEEPGEN_AR_MODEL_PATH
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    trust_remote_code=True,
    padding_side="right",
)

# Prompt template for Qwen2.5-VL
prompt_template = dict(
    IMG_START_TOKEN="<|vision_start|>",
    IMG_END_TOKEN="<|vision_end|>",
    IMG_CONTEXT_TOKEN="<|image_pad|>",
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM="<|im_start|>system\n{system}<|im_end|>\n",
    INSTRUCTION="<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
    SUFFIX="<|im_end|>",
    SUFFIX_AS_EOS=True,
    SEP="\n",
    STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
    GENERATION="Generate an image: {input}",
    CFG="Generate an image.",
)

# Model configuration
# Note: pretrained_model_name_or_path fields will be set from environment variables
model = dict(
    # Number of query tokens for the connector
    num_queries=128,
    # Connector configuration (bridges LLM to DiT)
    connector=dict(
        hidden_size=2048,
        intermediate_size=11946,
        num_hidden_layers=6,
        num_attention_heads=32,
        _attn_implementation="flash_attention_2",
    ),
    # LMM (Language-Multimodal Model) configuration
    # pretrained_model_name_or_path will be set from DG_DEEPGEN_AR_MODEL_PATH
    lmm=dict(
        type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    # Tokenizer reference
    tokenizer=tokenizer,
    # Prompt template reference
    prompt_template=prompt_template,
    # Whether to freeze LMM weights
    freeze_lmm=True,
    # Transformer (DiT) configuration
    # pretrained_model_name_or_path will be set from DG_DEEPGEN_DIFFUSION_MODEL_PATH
    transformer=dict(
        type=SD3Transformer2DModel.from_pretrained,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ),
    # Scheduler configuration
    # pretrained_model_name_or_path will be set from DG_DEEPGEN_DIFFUSION_MODEL_PATH
    scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        subfolder="scheduler",
    ),
    # VAE configuration
    # pretrained_model_name_or_path will be set from DG_DEEPGEN_DIFFUSION_MODEL_PATH
    vae=dict(
        type=AutoencoderKL.from_pretrained,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ),
    # Pretrained checkpoint path (set via DG_DEEPGEN_CHECKPOINT)
    pretrained_pth=None,
    # Whether to use activation checkpointing
    use_activation_checkpointing=False,
    # Whether to freeze transformer weights
    freeze_transformer=True,
    # LoRA configuration
    lora_rank=64,
    lora_alpha=128,
)
