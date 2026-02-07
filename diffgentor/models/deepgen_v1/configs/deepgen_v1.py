# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""DeepGen v1 model configuration.

This module defines the default configuration for DeepGen v1 model.
Model paths should be provided via environment variables, not hardcoded here.
"""

# Prompt template for Qwen2.5-VL
prompt_template = dict(
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
    INSTRUCTION='<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n',
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.'
)

# Connector configuration
connector_config = dict(
    hidden_size=2048,
    intermediate_size=11946,
    num_hidden_layers=6,
    _attn_implementation='flash_attention_2',
    num_attention_heads=32,
)

# Model configuration
model_config = dict(
    num_queries=128,
    vit_input_size=448,
    max_length=1024,
    freeze_lmm=True,
    freeze_transformer=True,
    use_activation_checkpointing=False,
)
