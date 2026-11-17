"""Phase 1L familiar-contrast bottleneck and neutral-trajectory closure."""

from .config import ExperimentConfig, confirmatory_config, smoke_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["ExperimentConfig", "confirmatory_config", "smoke_config", "run_pipeline"]
