"""Phase 1I compositional context-transformation audit."""

from .config import ExperimentConfig, confirmatory_config, smoke_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__=["ExperimentConfig","confirmatory_config","smoke_config","run_pipeline"]
