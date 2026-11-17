"""Human familiar-context state-space adjudication."""

from .config import AdjudicationConfig, load_config


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as implementation
    return implementation(*args, **kwargs)


__all__ = ["AdjudicationConfig", "load_config", "run_pipeline"]
