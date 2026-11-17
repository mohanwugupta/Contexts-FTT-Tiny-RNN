"""Frozen-model causal analysis of the v2.1 A/B representational subspace."""

from .config import CausalConfig, final_config, smoke_config

__version__ = "2.2.0"

__all__ = ["CausalConfig", "smoke_config", "final_config"]
