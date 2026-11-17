"""Hard-gated diagnosis and conditional rescue of human meta-prior transfer."""

from .config import DiagnoseRescueConfig, load_config
from .pipeline import run_pipeline

__all__ = ["DiagnoseRescueConfig", "load_config", "run_pipeline"]
