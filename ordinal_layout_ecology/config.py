"""Frozen configuration for the ordinal/layout ecology."""

from __future__ import annotations

from dataclasses import dataclass

from confirmatory_srn.config import FIXED_SEQUENCE
from execution_objective_search.config import ObjectiveCandidate


HIGH_MONOTONIC = "High_monotonic"
LOW_MONOTONIC = "Low_monotonic"
HIGH_SEQUENCE_ONLY = "High_sequence_only"
CONDITIONS = (HIGH_MONOTONIC, LOW_MONOTONIC, HIGH_SEQUENCE_ONLY)


@dataclass(frozen=True)
class LayoutEcologyConfig:
    hidden_dim: int = 8
    token_dim: int = 4
    ordinal_dim: int = 8
    layout_dim: int = 16
    global_context_dim: int = 2
    layout_banks: int = 2
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
    responsibility_min_abs_denominator: float = .01
    normative_c_time: float = .50
    normative_c_risk: float = 1.0
    normative_gamma: float = 2.0
    lambda_execution: float = .20
    overlap_levels: tuple[int, ...] = (0, 1, 2, 4)
    overlap_channel_count: int = 4
    mixed_half_layout_scale: float = .8660254037844386
    mixed_success_rule: str = (
        "higher_usable_between_endpoint_fraction_and_lower_AB_segment_distance_"
        "than_frozen_categorical_reference"
    )
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
        raise ValueError(f"Unknown layout-ecology condition {condition!r}.")

    def has_execution(self, condition: str) -> bool:
        if condition not in self.conditions:
            raise ValueError(f"Unknown layout-ecology condition {condition!r}.")
        return condition != HIGH_SEQUENCE_ONLY

    def validate(self):
        if (self.hidden_dim, self.token_dim) != (8, 4):
            raise ValueError("The frozen SRN has eight hidden units and four tokens.")
        if (self.ordinal_dim, self.layout_dim, self.global_context_dim) != (8, 16, 2):
            raise ValueError("Ordinal/layout/global dimensions are frozen at 8/16/2.")
        if self.layout_banks != 2:
            raise ValueError("Two familiar layout banks must exist from training onset.")
        if (self.high_scale, self.low_scale) != (2.0, .02):
            raise ValueError("High/Low initialization scales are frozen at 2/.02.")
        if self.fixed_sequence != FIXED_SEQUENCE or len(self.fixed_sequence) != 8:
            raise ValueError("The registered eight-item sequence is frozen.")
        if (self.learning_rate, self.gradient_clip) != (.005, 1.0):
            raise ValueError("The established optimizer schedule is frozen.")
        if self.block_length != 10 or self.n_blocks < 2:
            raise ValueError("Training requires alternating ten-sequence blocks.")
        if self.conditions != CONDITIONS:
            raise ValueError("Exactly the three registered conditions are required.")
        if self.overlap_levels != (0, 1, 2, 4) or self.overlap_channel_count != 4:
            raise ValueError("The sparse shared-object follow-up is frozen at 0/1/2/4.")
        if not 0 < self.gate_fraction <= 1:
            raise ValueError("gate_fraction must lie in (0,1].")
        self.execution_candidate.validate()
        if self.profile == "ecology":
            if self.training_sequences != 170:
                raise ValueError("The full ecology has exactly 170 sequences.")
            if self.screen_seeds != tuple(range(5)):
                raise ValueError("Screen seeds are frozen at 0--4.")
            if self.confirmatory_seeds != tuple(range(20)):
                raise ValueError("Confirmatory seeds are frozen at 0--19.")


def ecology_config() -> LayoutEcologyConfig:
    return LayoutEcologyConfig()


def smoke_config() -> LayoutEcologyConfig:
    return LayoutEcologyConfig(
        n_blocks=2, screen_seeds=(0, 1), confirmatory_seeds=(0, 1),
        gate_fraction=.50, make_plots=False, profile="smoke",
    )
