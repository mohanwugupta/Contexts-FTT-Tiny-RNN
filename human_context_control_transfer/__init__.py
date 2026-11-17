"""Locked human-task transfer test for the selected context-control models."""

from .config import HumanTransferConfig, load_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["HumanTransferConfig", "load_config", "run_pipeline"]
