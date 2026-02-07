# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Connector module for DeepGen model.

This module provides the connector architecture that bridges the LLM hidden states
to the DiT (Diffusion Transformer) input space.
"""

from diffgentor.models.deepgen.connector.configuration_connector import ConnectorConfig
from diffgentor.models.deepgen.connector.modeling_connector import ConnectorEncoder

__all__ = ["ConnectorConfig", "ConnectorEncoder"]
