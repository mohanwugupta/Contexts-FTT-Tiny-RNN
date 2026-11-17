"""Frozen Phase-1 configuration for recurrent preparation and commitment."""

from __future__ import annotations

from dataclasses import dataclass

from ftt_srn_execution_v2.config import FIXED_SEQUENCE


@dataclass(frozen=True)
class CommitmentConfig:
    hidden_dim: int = 8
    context_dim: int = 3
    token_dim: int = 4
    high_init_scale: float = 2.0
    low_init_scale: float = 0.02
    context_alpha: float = 1.0
    learning_rate: float = 0.005
    gradient_clip: float = 1.0
    fixed_sequence: tuple[int, ...] = FIXED_SEQUENCE
    block_length: int = 10
    n_blocks: int = 17
    seeds: tuple[int, ...] = tuple(range(5))
    max_preparation_cycles: int = 8
    lambda_commit: float = 1.0
    c_error: float = 1.0
    c_time_values: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10)
    confidence_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80, 0.90)
    direction_window_sequences: int = 20
    minimum_seed_fraction: float = 0.80
    minimum_preparation_improvement_fraction: float = 0.50
    maximum_boundary_commit_fraction: float = 0.90
    minimum_context_effect: float = 0.002
    switch_preceding_window: int = 3
    switch_recovery_window: int = 3
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "screen"

    @property
    def training_sequences(self) -> int:
        return self.n_blocks * self.block_length

    @property
    def transitions_per_sequence(self) -> int:
        return len(self.fixed_sequence) - 1

    def validate(self):
        if self.hidden_dim != 8:
            raise ValueError("The commitment SRN fixes hidden_dim at 8.")
        if self.token_dim != 4 or self.context_dim != 3:
            raise ValueError("The commitment SRN requires 4 token and 3 context inputs.")
        if self.high_init_scale != 2.0:
            raise ValueError("High initialization scale is fixed at 2.")
        if self.low_init_scale != .02:
            raise ValueError("The existing Low initialization scale is fixed at .02.")
        if self.context_alpha != 1.0:
            raise ValueError("Context strength is fixed at 1.")
        if self.learning_rate != .005:
            raise ValueError("The established SRN learning rate is fixed at .005.")
        if self.block_length != 10:
            raise ValueError("The frozen curriculum requires 10 sequences per block.")
        if self.n_blocks < 2:
            raise ValueError("At least an A and B block are required.")
        if self.max_preparation_cycles != 8:
            raise ValueError("Phase 1 fixes M_max at 8; robustness horizons are later analyses.")
        if self.lambda_commit != 1 or self.c_error != 1:
            raise ValueError("Phase 1 fixes lambda_commit=c_error=1.")
        if self.c_time_values != (.01, .025, .05, .10):
            raise ValueError("Phase 1 uses only the prespecified four-value time-cost grid.")
        if self.confidence_thresholds != (.60, .70, .80, .90):
            raise ValueError("Threshold controls are fixed at .60/.70/.80/.90.")
        if len(self.fixed_sequence) != 8 or set(self.fixed_sequence) - set(range(4)):
            raise ValueError("The fixed eight-item/four-key motor sequence must be preserved.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if self.profile == "screen":
            if self.n_blocks != 17 or self.training_sequences != 170:
                raise ValueError("The scientific screen requires exactly 170 sequences.")
            if self.seeds != tuple(range(5)):
                raise ValueError("The scientific screen requires seeds 0--4.")
        if self.profile == "confirmatory":
            if self.n_blocks != 17 or self.training_sequences != 170:
                raise ValueError("The confirmatory run requires exactly 170 sequences.")
            if self.seeds != tuple(range(20)):
                raise ValueError("The confirmatory run requires seeds 0--19.")


def smoke_config() -> CommitmentConfig:
    return CommitmentConfig(
        n_blocks=2,
        seeds=(0,),
        c_time_values=(.01, .025, .05, .10),
        make_plots=False,
        profile="smoke",
    )


def screen_config() -> CommitmentConfig:
    return CommitmentConfig()


def confirmatory_config() -> CommitmentConfig:
    return CommitmentConfig(seeds=tuple(range(20)), profile="confirmatory")


def confirmatory_smoke_config() -> CommitmentConfig:
    return CommitmentConfig(
        n_blocks=2, seeds=(0,), make_plots=False, profile="confirmatory_smoke"
    )
