"""Analysis-only final task-space compositionality synthesis."""

from .config import SynthesisConfig


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["SynthesisConfig", "run_pipeline"]
