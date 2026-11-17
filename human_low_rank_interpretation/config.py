"""Preregistered constants for PRD v3.1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LowRankInterpretationConfig:
    candidate_ranks: tuple[int, ...] = (1, 2, 3, 7)
    cv_folds: int = 5
    bootstrap_replicates: int = 1000
    parallel_analysis_replicates: int = 1000
    semantic_recovery_replicates: int = 200
    stability_cosine_threshold: float = 0.70
    semantic_recovery_threshold: float = 0.60
    random_false_label_threshold: float = 0.05
    association_alpha: float = 0.05
    score_shrinkage_strength: float = 8.0
    random_seed: int = 31000
    participant_limit_per_dataset: int | None = None
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "final"

    def validate(self):
        if tuple(sorted(set(self.candidate_ranks))) != self.candidate_ranks:
            raise ValueError("candidate_ranks must be unique and increasing.")
        if not self.candidate_ranks or min(self.candidate_ranks) < 1 or max(self.candidate_ranks) > 7:
            raise ValueError("candidate_ranks must lie between one and seven.")
        if self.cv_folds < 2:
            raise ValueError("Participant cross-validation needs at least two folds.")
        if min(self.bootstrap_replicates, self.parallel_analysis_replicates,
               self.semantic_recovery_replicates) < 1:
            raise ValueError("All resampling counts must be positive.")
        if not 0 < self.stability_cosine_threshold <= 1:
            raise ValueError("stability_cosine_threshold must be in (0,1].")
        if self.participant_limit_per_dataset is not None and self.participant_limit_per_dataset < 10:
            raise ValueError("Smoke limits must retain at least ten participants per dataset.")
        if self.profile not in {"smoke", "final"}:
            raise ValueError("profile must be smoke or final.")


def smoke_config() -> LowRankInterpretationConfig:
    return LowRankInterpretationConfig(
        cv_folds=3,
        bootstrap_replicates=50,
        parallel_analysis_replicates=100,
        semantic_recovery_replicates=25,
        participant_limit_per_dataset=24,
        profile="smoke",
    )


def final_config() -> LowRankInterpretationConfig:
    return LowRankInterpretationConfig()
