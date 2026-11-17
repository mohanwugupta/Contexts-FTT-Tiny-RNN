"""Frozen SRN nonlinearity and output-relevant compositionality analysis."""

from .config import MechanismConfig


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["MechanismConfig", "run_pipeline"]
