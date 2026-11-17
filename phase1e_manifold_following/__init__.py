"""Phase 1E relinearized manifold-following retrofit."""

from .config import ExperimentConfig, smoke_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["ExperimentConfig", "smoke_config", "run_pipeline"]
