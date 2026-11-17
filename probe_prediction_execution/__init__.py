"""Prediction-versus-execution diagnostics for time-matched frozen probes."""

from .analysis import (
    aggregate_prediction_execution,
    compute_probe_gaps,
    reconstruct_normalized_rt,
)
from .config import PredictionExecutionConfig, load_config
from .pipeline import run_probe_prediction_execution

__all__ = [
    "PredictionExecutionConfig",
    "aggregate_prediction_execution",
    "compute_probe_gaps",
    "load_config",
    "reconstruct_normalized_rt",
    "run_probe_prediction_execution",
]
