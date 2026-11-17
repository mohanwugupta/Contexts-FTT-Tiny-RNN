"""Familiar-only checkpoint selection and locked random-control probes."""

from .config import SweepConfig, load_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["SweepConfig", "load_config", "run_pipeline"]
