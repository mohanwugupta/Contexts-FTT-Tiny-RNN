"""Frozen SRN Mixed/Novel representation diagnostics."""

from .config import DiagnosticConfig, default_config, smoke_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation

    return implementation(*args, **kwargs)


__all__ = ["DiagnosticConfig", "default_config", "smoke_config", "run_pipeline"]
