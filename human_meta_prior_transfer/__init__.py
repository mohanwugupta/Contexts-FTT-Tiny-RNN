"""Causal transfer of the locked H128 synthetic meta-prior."""

from .config import HumanMetaTransferConfig, load_human_meta_transfer_config


def run_human_meta_transfer(*args, **kwargs):
    from .pipeline import run_human_meta_transfer as _run
    return _run(*args, **kwargs)


__all__ = ["HumanMetaTransferConfig", "load_human_meta_transfer_config", "run_human_meta_transfer"]
