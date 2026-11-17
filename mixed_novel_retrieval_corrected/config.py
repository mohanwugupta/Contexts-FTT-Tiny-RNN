from __future__ import annotations

from dataclasses import dataclass,replace
import json,os
from pathlib import Path


@dataclass(frozen=True)
class RetrievalConfig:
    analysis_version: str="mixed-novel-retrieval-corrected-1.1.0-absolute-fit"
    profile: str="final"
    bootstrap_replicates: int=1000
    random_seed: int=91827
    participant_cv_folds: int=5
    r2_shrinkage: float=10.0
    r3_rank: int=2
    familiar_plausibility_delta_bic: float=20.0
    posterior_predictive_profiles: int=5000
    posterior_summary_replicates: int=1000
    workers: int=0
    make_plots: bool=True
    show_progress: bool=True

    @property
    def effective_workers(self):
        return self.workers or max(1,min(6,(os.cpu_count() or 2)-1))

    def validate(self):
        if self.profile not in {"smoke","final"}: raise ValueError("profile must be smoke or final")
        if self.bootstrap_replicates<1: raise ValueError("bootstrap_replicates must be positive")
        if self.participant_cv_folds<2: raise ValueError("participant_cv_folds must be >=2")
        if self.r3_rank<1 or self.r3_rank>7: raise ValueError("r3_rank must be in 1..7")
        if self.posterior_predictive_profiles<5000: raise ValueError("posterior_predictive_profiles must be at least 5000")
        if self.posterior_summary_replicates<100: raise ValueError("posterior_summary_replicates must be at least 100")
        if self.workers<0: raise ValueError("workers must be zero or positive")
        return self


def load_config(path=None,*,profile=None):
    payload=json.loads(Path(path).read_text()) if path else {}
    known=set(RetrievalConfig.__dataclass_fields__)
    config=RetrievalConfig(**{k:v for k,v in payload.items() if k in known})
    if profile: config=replace(config,profile=profile)
    if config.profile=="smoke":
        config=replace(config,bootstrap_replicates=min(config.bootstrap_replicates,20),workers=min(config.effective_workers,2))
    return config.validate()
