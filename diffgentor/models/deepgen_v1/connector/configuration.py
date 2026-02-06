# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Connector configuration for DeepGen model."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ConnectorConfig(PretrainedConfig):
    """Configuration class for the Connector encoder.

    This connector bridges the LLM hidden states to the DiT model.

    Args:
        hidden_size: Hidden dimension size (default: 768)
        intermediate_size: FFN intermediate size (default: 3072)
        num_hidden_layers: Number of transformer layers (default: 12)
        num_attention_heads: Number of attention heads (default: 12)
        hidden_act: Activation function (default: "gelu_pytorch_tanh")
        layer_norm_eps: Layer norm epsilon (default: 1e-6)
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
