"""Frozen scientific settings for the execution-readout diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

from execution_objective_search.config import ObjectiveCandidate, SearchConfig
from ftt_srn_execution_v2.config import FIXED_SEQUENCE


READOUT_CONDITIONS = (
    "deterministic_normative",
    "monotonic_two_parameter",
    "unrestricted_zero_initialized",
)

READOUT_LABELS = {
    "deterministic_normative": "Deterministic normative policy",
    "monotonic_two_parameter": "Two-parameter monotonic policy",
    "unrestricted_zero_initialized": "Unrestricted hidden-state head (zero init)",
}


@dataclass(frozen=True)
class ReadoutDiagnosticConfig:
    hidden_dim: int = 8
    context_alpha: float = 1.0
    high_init_scale: float = 2.0
    learning_rate: float = 0.005
    gradient_clip: float = 1.0
    fixed_sequence: tuple[int, ...] = FIXED_SEQUENCE
    block_length: int = 10
    n_blocks: int = 17
    seeds: tuple[int, ...] = tuple(range(5))
    readout_conditions: tuple[str, ...] = READOUT_CONDITIONS
    normative_c_time: float = 0.50
    normative_c_risk: float = 1.00
    normative_gamma: float = 2.00
    lambda_exec: float = 0.20
    direction_window_sequences: int = 20
    minimum_direction_seed_fraction: float = 0.80
    saturation_epsilon: float = 0.02
    maximum_trained_saturation_fraction: float = 0.80
    minimum_trained_speed_range: float = 0.05
    minimum_switch_jump: float = 0.002
    minimum_context_coefficient: float = 0.002
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "screen"

    @property
    def training_sequences(self) -> int:
        return self.n_blocks * self.block_length

    @property
    def normative_candidate(self) -> ObjectiveCandidate:
        return ObjectiveCandidate(
            "A", self.normative_c_time, self.normative_c_risk,
            self.normative_gamma, self.lambda_exec,
        )

    def validate(self):
        if self.hidden_dim != 8:
            raise ValueError("The readout diagnostic fixes the Elman hidden layer at 8 units.")
        if self.context_alpha != 1.0:
            raise ValueError("Context strength is fixed at 1.")
        if self.high_init_scale != 2.0:
            raise ValueError("The diagnostic fixes High initialization scale at 2.")
        if self.learning_rate != .005:
            raise ValueError("The established learning rate is fixed at .005.")
        if self.block_length != 10:
            raise ValueError("The frozen curriculum uses A10/B10 blocks.")
        if self.n_blocks < 2:
            raise ValueError("At least one A and one B block are required.")
        if self.readout_conditions != READOUT_CONDITIONS:
            raise ValueError("All three prespecified readout conditions are required in order.")
        if len(self.fixed_sequence) != 8 or set(self.fixed_sequence) - set(range(4)):
            raise ValueError("The fixed eight-item/four-key sequence must be preserved.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        self.normative_candidate.validate()
        if self.profile == "screen":
            if self.n_blocks != 17 or self.training_sequences != 170:
                raise ValueError("The scientific diagnostic requires all 170 sequences.")
            if self.seeds != tuple(range(5)):
                raise ValueError("The scientific diagnostic requires seeds 0--4.")

    def gate_config(self) -> SearchConfig:
        """Return the exact frozen A--E gate settings from objective search."""
        return SearchConfig(
            hidden_dim=self.hidden_dim,
            context_alpha=self.context_alpha,
            high_init_scale=self.high_init_scale,
            learning_rate=self.learning_rate,
            gradient_clip=self.gradient_clip,
            fixed_sequence=self.fixed_sequence,
            block_length=self.block_length,
            n_blocks=self.n_blocks,
            seeds=self.seeds,
            lambda_exec=self.lambda_exec,
            direction_window_sequences=self.direction_window_sequences,
            minimum_direction_seed_fraction=self.minimum_direction_seed_fraction,
            saturation_epsilon=self.saturation_epsilon,
            maximum_trained_saturation_fraction=self.maximum_trained_saturation_fraction,
            minimum_trained_speed_range=self.minimum_trained_speed_range,
            minimum_switch_jump=self.minimum_switch_jump,
            minimum_context_coefficient=self.minimum_context_coefficient,
            make_plots=False,
            show_progress=False,
            profile="screen" if self.profile == "screen" else "unit",
        )


def smoke_config() -> ReadoutDiagnosticConfig:
    return ReadoutDiagnosticConfig(
        n_blocks=2, seeds=(0,), make_plots=False, profile="smoke"
    )


def screen_config() -> ReadoutDiagnosticConfig:
    return ReadoutDiagnosticConfig()
