"""Minimal proof of concept for emergent averaging under uncertainty."""

from .environment.episodes import V1Batch, V1EpisodeConfig, generate_batch

__version__ = "1.0.0"

__all__ = ["V1Batch", "V1EpisodeConfig", "generate_batch"]
