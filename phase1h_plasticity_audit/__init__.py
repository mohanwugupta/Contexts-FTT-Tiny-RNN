"""Phase 1H shared/private representational plasticity audit."""

from .config import AuditConfig, confirmatory_config, smoke_config


def run_audit(*args, **kwargs):
    from .pipeline import run_audit as implementation
    return implementation(*args, **kwargs)


__all__ = ["AuditConfig", "confirmatory_config", "smoke_config", "run_audit"]
