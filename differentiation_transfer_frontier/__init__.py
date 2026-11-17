"""Differentiation--transfer frontier experiment."""

from .config import FrontierConfig, load_frontier_config
from .pipeline import run_frontier

__all__ = ["FrontierConfig", "load_frontier_config", "run_frontier"]
