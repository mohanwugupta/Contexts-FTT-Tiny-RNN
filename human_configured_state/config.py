"""Preregistered constants for the human configured-state analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanConfiguredStateConfig:
    endpoint_epsilon_log_rt: float = 0.03
    srn_endpoint_epsilon_rt: float = 0.02
    profile_shrinkage_strength: float = 8.0
    token_effect_shrinkage_strength: float = 12.0
    low_rank_variance_threshold: float = 0.90
    reliability_threshold: float = 0.30
    cv_folds: int = 5
    recovery_replicates: int = 20
    recovery_effect_scale: float = 1.5
    recovery_above_chance: float = 0.20
    random_seed: int = 30000
    participant_limit_per_dataset: int | None = None
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "final"

    def validate(self):
        if self.endpoint_epsilon_log_rt <= 0 or self.srn_endpoint_epsilon_rt <= 0:
            raise ValueError("Endpoint reliability thresholds must be positive.")
        if self.profile_shrinkage_strength <= 0 or self.token_effect_shrinkage_strength <= 0:
            raise ValueError("Hierarchical shrinkage strengths must be positive.")
        if not 0 < self.low_rank_variance_threshold <= 1:
            raise ValueError("Low-rank variance threshold must be in (0,1].")
        if self.cv_folds < 2:
            raise ValueError("Participant cross-validation requires at least two folds.")
        if self.recovery_replicates < 1:
            raise ValueError("At least one recovery replicate is required.")
        if self.participant_limit_per_dataset is not None and self.participant_limit_per_dataset < 6:
            raise ValueError("A participant limit must retain at least six people per dataset.")
        if self.profile not in {"smoke", "final"}:
            raise ValueError("profile must be smoke or final.")


def smoke_config() -> HumanConfiguredStateConfig:
    return HumanConfiguredStateConfig(
        cv_folds=3, recovery_replicates=3, participant_limit_per_dataset=12,
        make_plots=True, profile="smoke"
    )


def final_config() -> HumanConfiguredStateConfig:
    return HumanConfiguredStateConfig()
