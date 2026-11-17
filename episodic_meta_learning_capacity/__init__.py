"""Episodic FOMAML capacity experiment for contextual SRNs."""

from .config import MetaCapacityConfig, load_meta_capacity_config

def run_meta_capacity_experiment(*args, **kwargs):
    from .pipeline import run_meta_capacity_experiment as _run
    return _run(*args, **kwargs)

__all__ = ["MetaCapacityConfig", "load_meta_capacity_config", "run_meta_capacity_experiment"]
