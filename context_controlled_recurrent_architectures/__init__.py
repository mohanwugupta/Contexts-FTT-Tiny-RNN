"""Context-controlled recurrent architecture experiment."""

from .config import ExperimentConfig, load_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)

__all__ = ["ExperimentConfig", "load_config", "run_pipeline"]
