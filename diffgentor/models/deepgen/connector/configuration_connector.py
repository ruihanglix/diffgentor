# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Connector configuration for DeepGen model."""

from transformers.configuration_utils import PretrainedConfig


class ConnectorConfig(PretrainedConfig):
    """Configuration class for the Connector module.

    The Connector module bridges the LLM hidden states to the DiT input space.

    Args:
        hidden_size: Hidden dimension size (default: 768)
        intermediate_size: Intermediate dimension in MLP (default: 3072)
        num_hidden_layers: Number of transformer layers (default: 12)
        num_attention_heads: Number of attention heads (default: 12)
        hidden_act: Activation function (default: "gelu_pytorch_tanh")
        layer_norm_eps: Layer normalization epsilon (default: 1e-6)
        attention_dropout: Attention dropout rate (default: 0.0)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        _attn_implementation: str = "sdpa",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self._attn_implementation = _attn_implementation
