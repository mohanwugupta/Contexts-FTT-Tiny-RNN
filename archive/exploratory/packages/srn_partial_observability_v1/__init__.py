"""Elman-SRN proof of concept for averaging under partial observability."""

from .environment.episodes import EpisodeBatch, EpisodeConfig, generate_batch

__version__ = "1.0.0"

__all__ = ["EpisodeBatch", "EpisodeConfig", "generate_batch"]

