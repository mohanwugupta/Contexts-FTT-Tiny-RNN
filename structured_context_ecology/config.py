"""Frozen configuration for the stepwise structured-context ecology."""

from __future__ import annotations

from dataclasses import dataclass

from confirmatory_srn.config import FIXED_SEQUENCE
from execution_objective_search.config import ObjectiveCandidate


HIGH_MONOTONIC = "High_monotonic"
LOW_MONOTONIC = "Low_monotonic"
HIGH_SEQUENCE_ONLY = "High_sequence_only"
CONDITIONS = (HIGH_MONOTONIC, LOW_MONOTONIC, HIGH_SEQUENCE_ONLY)
SALIENCE_LEVELS = (0.0, 0.05, 0.15, 0.30)


@dataclass(frozen=True)
class EcologyConfig:
    hidden_dim: int = 8
    token_dim: int = 4
    high_scale: float = 2.0
    low_scale: float = .02
    fixed_sequence: tuple[int, ...] = FIXED_SEQUENCE
    learning_rate: float = .005
    gradient_clip: float = 1.0
    block_length: int = 10
    n_blocks: int = 17
    conditions: tuple[str, ...] = CONDITIONS
    screen_seeds: tuple[int, ...] = tuple(range(5))
    confirmatory_seeds: tuple[int, ...] = tuple(range(20))
    gate_fraction: float = .80
    minimum_stage0_separation_retention: float = .50
    normative_c_time: float = .50
    normative_c_risk: float = 1.0
    normative_gamma: float = 2.0
    lambda_execution: float = .20
    responsibility_min_abs_denominator: float = .01
    salience_sigmas: tuple[float, ...] = SALIENCE_LEVELS
    salience_selection_rule: str = (
        "smallest_gate_passing_nonzero_sigma_improving_both_RT_between_fraction_"
        "and_hidden_segment_distance_vs_sigma0"
    )
    stage2_position_groups: tuple[int, ...] = (1, 1, 2, 2, 3, 3, 3)
    trained_structured_context_dim: int = 7
    reserved_novel_context_dim: int = 3
    noise_seed_offset: int = 41000
    baseline_output: str = "outputs/confirmatory_srn_final"
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "ecology"

    @property
    def training_sequences(self) -> int:
        return self.block_length * self.n_blocks

    @property
    def transitions_per_sequence(self) -> int:
        return len(self.fixed_sequence) - 1

    @property
    def structured_model_context_dim(self) -> int:
        return self.trained_structured_context_dim + self.reserved_novel_context_dim

    @property
    def execution_candidate(self) -> ObjectiveCandidate:
        return ObjectiveCandidate(
            "A", self.normative_c_time, self.normative_c_risk,
            self.normative_gamma, self.lambda_execution,
        )

    def condition_scale(self, condition: str) -> float:
        if condition in {HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY}:
            return self.high_scale
        if condition == LOW_MONOTONIC:
            return self.low_scale
        raise ValueError(f"Unknown ecology condition {condition!r}.")

    def has_execution(self, condition: str) -> bool:
        if condition not in self.conditions:
            raise ValueError(f"Unknown ecology condition {condition!r}.")
        return condition != HIGH_SEQUENCE_ONLY

    def validate(self):
        if (self.hidden_dim, self.token_dim) != (8, 4):
            raise ValueError("The ecology freezes the eight-unit, four-token SRN.")
        if (self.high_scale, self.low_scale) != (2.0, .02):
            raise ValueError("High/Low initialization scales are frozen at 2/.02.")
        if self.fixed_sequence != FIXED_SEQUENCE:
            raise ValueError("The registered repository sequence is frozen.")
        if (self.learning_rate, self.gradient_clip) != (.005, 1.0):
            raise ValueError("The established optimizer schedule is frozen.")
        if self.block_length != 10 or self.n_blocks < 2:
            raise ValueError("The ecology requires alternating ten-sequence blocks.")
        if self.conditions != CONDITIONS:
            raise ValueError("Exactly the three frozen conditions are required.")
        if self.salience_sigmas != SALIENCE_LEVELS:
            raise ValueError("The salience screen is frozen at 0,.05,.15,.30.")
        if self.stage2_position_groups != (1, 1, 2, 2, 3, 3, 3):
            raise ValueError("Ordinal transition grouping is frozen at 2/2/3.")
        if (self.trained_structured_context_dim, self.reserved_novel_context_dim) != (7, 3):
            raise ValueError("Structured context uses seven trained plus three Novel-only axes.")
        if not 0 < self.gate_fraction <= 1:
            raise ValueError("gate_fraction must lie in (0,1].")
        if not 0 < self.minimum_stage0_separation_retention <= 1:
            raise ValueError("Separation retention must lie in (0,1].")
        self.execution_candidate.validate()
        if self.profile == "ecology":
            if self.training_sequences != 170:
                raise ValueError("The confirmatory ecology has exactly 170 sequences.")
            if self.screen_seeds != tuple(range(5)):
                raise ValueError("Screening seeds are frozen at 0--4.")
            if self.confirmatory_seeds != tuple(range(20)):
                raise ValueError("Confirmatory seeds are frozen at 0--19.")


def ecology_config() -> EcologyConfig:
    return EcologyConfig()


def smoke_config() -> EcologyConfig:
    return EcologyConfig(
        n_blocks=2,
        screen_seeds=(0, 1),
        confirmatory_seeds=(0, 1),
        gate_fraction=.50,
        make_plots=False,
        profile="smoke",
    )
