"""Probe retrieval from immutable corrected familiar state-space anchors."""

from .config import RetrievalConfig,load_config


def run_pipeline(*args,**kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args,**kwargs)


__all__=["RetrievalConfig","load_config","run_pipeline"]
