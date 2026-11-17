"""Locked configuration for the human state-space adjudication."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class AdjudicationConfig:
    analysis_version: str = "human-state-space-adjudication-1.0.0"
    profile: str = "final"
    random_seed: int = 73191
    multistarts: int = 8
    cv_multistarts: int = 2
    recovery_replicates: int = 100
    recovery_threshold: float = 0.80
    participant_limit: int | None = None
    cv_min_training_blocks: int = 2
    cv_max_folds_per_participant: int | None = None
    m3b_ridge: float = 1.0
    make_plots: bool = True
    show_progress: bool = True
    workers: int = 0

    def validate(self) -> "AdjudicationConfig":
        if self.profile not in {"smoke", "final"}:
            raise ValueError("profile must be smoke or final")
        if self.multistarts < 2:
            raise ValueError("multistarts must be at least two")
        if self.cv_multistarts < 2:
            raise ValueError("cv_multistarts must be at least two")
        if self.recovery_replicates < 1:
            raise ValueError("recovery_replicates must be positive")
        if not 0 < self.recovery_threshold <= 1:
            raise ValueError("recovery_threshold must be in (0, 1]")
        if self.cv_min_training_blocks < 2:
            raise ValueError("CV must begin after at least two familiar blocks")
        if self.workers < 0:
            raise ValueError("workers must be zero (automatic) or positive")
        return self

    @property
    def effective_workers(self) -> int:
        if self.workers:
            return self.workers
        return max(1,min(6,(os.cpu_count() or 2)-1))


def load_config(path: str | Path | None = None, *, profile: str | None = None) -> AdjudicationConfig:
    payload = {}
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    known = set(AdjudicationConfig.__dataclass_fields__)
    config = AdjudicationConfig(**{k: v for k, v in payload.items() if k in known})
    if profile is not None:
        config = replace(config, profile=profile)
    if config.profile == "smoke":
        config = replace(
            config, multistarts=min(config.multistarts, 3),
            recovery_replicates=min(config.recovery_replicates, 1),
            participant_limit=config.participant_limit or 3,
            cv_max_folds_per_participant=1,
        )
    return config.validate()
