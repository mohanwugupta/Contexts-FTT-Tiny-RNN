"""Output-relevant compositionality sufficiency and emergence experiment."""

from .config import ExperimentConfig


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__=["ExperimentConfig","run_pipeline"]
