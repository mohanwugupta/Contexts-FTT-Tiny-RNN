"""Fixed optimization settings and environmental condition registry."""

from __future__ import annotations

from dataclasses import dataclass, replace


VALIDITY_LEVELS = (1.00, 0.95, 0.85, 0.75, 0.65, 0.55)
CUE_ACCURACY_LEVELS = (0.97, 0.90, 0.80, 0.70, 0.60)
STATE_SEPARATION_LEVELS = (0.20, 0.40, 0.60, 0.80)
OUTCOME_NOISE_LEVELS = (0.0, 0.01, 0.02, 0.05)
STATE_VARIANTS = ("fixed_global", "fixed_per_run", "context_learning")
DIAGNOSTIC_Q_LEVELS = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class EnvironmentalCondition:
    context_validity: float
    cue_accuracy: float
    state_separation: float
    outcome_noise_sd: float = 0.01
    state_variant: str = "fixed_global"

    def validate(self):
        if not 0.5 <= self.context_validity <= 1.0:
            raise ValueError("context_validity must be in [0.5,1].")
        if not 0.5 < self.cue_accuracy <= 1.0:
            raise ValueError("cue_accuracy must be in (0.5,1].")
        if not 0.0 < self.state_separation < 1.0:
            raise ValueError("state_separation must be in (0,1).")
        if self.outcome_noise_sd < 0:
            raise ValueError("outcome_noise_sd cannot be negative.")
        if self.state_variant not in STATE_VARIANTS:
            raise ValueError(f"Unknown state_variant {self.state_variant!r}.")

    @property
    def condition_id(self) -> str:
        variants = {
            "fixed_global": "S1",
            "fixed_per_run": "S2",
            "context_learning": "S3",
        }
        return (
            f"v{self.context_validity:.2f}_cue{self.cue_accuracy:.2f}_"
            f"d{self.state_separation:.2f}_y{self.outcome_noise_sd:.3f}_"
            f"{variants[self.state_variant]}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    hidden_dim: int = 8
    block_length: int = 10
    n_blocks: int = 16
    training_batch_size: int = 128
    learning_rate: float = 0.005
    gradient_clip: float = 1.0
    cue_mean: float = 1.0
    discovery_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    confirmation_seeds: tuple[int, ...] = tuple(range(1000, 1020))
    discovery_pass_required: int = 4
    confirmation_pass_required: int = 16
    diagnostic_histories: int = 16
    diagnostic_q_levels: tuple[float, ...] = DIAGNOSTIC_Q_LEVELS
    conditions: tuple[EnvironmentalCondition, ...] | None = None
    run_confirmation: bool = True
    run_followups: bool = True
    run_ablations: bool = True
    run_hidden_analysis: bool = True
    max_confirmation_regimes: int = 3
    force_staged_analyses: bool = False
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "discovery"

    @property
    def training_trials(self) -> int:
        return int(self.n_blocks * self.block_length)

    @property
    def weight_updates(self) -> int:
        return self.training_trials

    @property
    def n_block_pairs(self) -> int:
        return self.n_blocks // 2

    def validate(self):
        if self.hidden_dim != 8:
            raise ValueError("The primary architecture is fixed at eight hidden units.")
        if self.block_length != 10:
            raise ValueError("The FTT block length is fixed at exactly ten trials.")
        if self.n_blocks < 2 or self.n_blocks % 2:
            raise ValueError("n_blocks must be a positive even number of complete A/B pairs.")
        if self.training_batch_size < 1:
            raise ValueError("training_batch_size must be positive.")
        if not self.discovery_seeds:
            raise ValueError("At least one discovery seed is required.")
        overlap = set(self.discovery_seeds) & set(self.confirmation_seeds)
        if overlap:
            raise ValueError(f"Discovery and confirmation seeds overlap: {sorted(overlap)}")
        for condition in self.conditions or discovery_conditions():
            condition.validate()


def discovery_conditions(
    *, outcome_noise_sd: float = 0.01, state_variant: str = "fixed_global"
) -> tuple[EnvironmentalCondition, ...]:
    return tuple(
        EnvironmentalCondition(validity, accuracy, separation, outcome_noise_sd, state_variant)
        for validity in VALIDITY_LEVELS
        for accuracy in CUE_ACCURACY_LEVELS
        for separation in STATE_SEPARATION_LEVELS
    )


def smoke_config() -> ExperimentConfig:
    conditions = (
        EnvironmentalCondition(1.00, 0.90, 0.20, 0.01, "fixed_global"),
        EnvironmentalCondition(0.75, 0.80, 0.60, 0.01, "fixed_global"),
        EnvironmentalCondition(0.55, 0.60, 0.80, 0.01, "fixed_global"),
    )
    return ExperimentConfig(
        n_blocks=4,
        training_batch_size=8,
        discovery_seeds=(0,),
        confirmation_seeds=(1000,),
        discovery_pass_required=1,
        confirmation_pass_required=1,
        diagnostic_histories=2,
        conditions=conditions,
        run_confirmation=False,
        run_followups=False,
        run_ablations=False,
        run_hidden_analysis=False,
        max_confirmation_regimes=1,
        profile="smoke",
    )


def discovery_config() -> ExperimentConfig:
    return ExperimentConfig()


def with_software_stages(config: ExperimentConfig) -> ExperimentConfig:
    """Reduced forced-stage profile used only to exercise software paths."""
    return replace(
        config,
        conditions=(EnvironmentalCondition(0.75, 0.80, 0.60, 0.01, "fixed_global"),),
        run_confirmation=True,
        run_followups=True,
        run_ablations=True,
        run_hidden_analysis=True,
        force_staged_analyses=True,
    )

