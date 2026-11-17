"""Frozen specification for the confirmatory SRN pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from execution_objective_search.config import ObjectiveCandidate
from ftt_srn_execution_v2.config import FIXED_SEQUENCE as REGISTERED_FIXED_SEQUENCE


# The prose PRD lists a different physical-key sequence, but simultaneously
# freezes the existing working run. This registered tuple is enforced throughout
# the existing configs, Methods, outputs, and regression tests.
FIXED_SEQUENCE = REGISTERED_FIXED_SEQUENCE

HIGH_MONOTONIC = "High_monotonic"
LOW_MONOTONIC = "Low_monotonic"
HIGH_SEQUENCE_ONLY = "High_sequence_only"
CONDITIONS = (HIGH_MONOTONIC, LOW_MONOTONIC, HIGH_SEQUENCE_ONLY)


@dataclass(frozen=True)
class ConfirmatoryConfig:
    hidden_dim: int = 8
    token_dim: int = 4
    context_dim: int = 3
    high_scale: float = 2.0
    low_scale: float = .02
    context_alpha: float = 1.0
    fixed_sequence: tuple[int, ...] = FIXED_SEQUENCE
    learning_rate: float = .005
    gradient_clip: float = 1.0
    block_length: int = 10
    n_blocks: int = 17
    seeds: tuple[int, ...] = tuple(range(20))
    conditions: tuple[str, ...] = CONDITIONS
    normative_c_time: float = .50
    normative_c_risk: float = 1.0
    normative_gamma: float = 2.0
    lambda_execution: float = .20
    responsibility_min_abs_denominator: float = .01
    low_rank_candidates: tuple[int, ...] = (1, 2, 3, 7)
    low_rank_selection_rule: str = "minimum_leave_seed_and_transition_out_AB_MSE"
    sequence_preservation_threshold: float = .90
    pca_components: int = 2
    random_control_seed: int = 20260904
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "confirmatory"

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
        raise ValueError(f"Unknown confirmatory condition {condition!r}.")

    def has_execution(self, condition: str) -> bool:
        if condition not in self.conditions:
            raise ValueError(f"Unknown confirmatory condition {condition!r}.")
        return condition != HIGH_SEQUENCE_ONLY

    def validate(self):
        if self.hidden_dim != 8 or self.token_dim != 4 or self.context_dim != 3:
            raise ValueError("Confirmatory architecture is fixed at 4+3 inputs and 8 hidden units.")
        if self.high_scale != 2.0:
            raise ValueError("Confirmatory High scale is fixed at 2.0.")
        if self.low_scale != .02:
            raise ValueError("Confirmatory Low scale is the existing .02 value.")
        if self.context_alpha != 1.0:
            raise ValueError("Context input strength is fixed at 1.0.")
        if self.fixed_sequence != FIXED_SEQUENCE:
            raise ValueError("The sequence must match the repository's registered working run.")
        if self.learning_rate != .005 or self.gradient_clip != 1.0:
            raise ValueError("The established optimizer schedule is frozen.")
        if self.block_length != 10 or self.n_blocks < 2:
            raise ValueError("Training requires alternating ten-sequence blocks.")
        if self.conditions != CONDITIONS:
            raise ValueError("Exactly the three prespecified primary conditions are required.")
        if self.low_rank_candidates != (1, 2, 3, 7):
            raise ValueError("Low-rank candidates are frozen at 1,2,3,full(7).")
        if self.responsibility_min_abs_denominator <= 0:
            raise ValueError("Responsibility denominators require a positive threshold.")
        if self.sequence_preservation_threshold != .90:
            raise ValueError(
                "The established causal sequence-preservation threshold is fixed at .90."
            )
        self.execution_candidate.validate()
        if self.profile == "confirmatory":
            if self.training_sequences != 170:
                raise ValueError("Confirmatory exposure is exactly 170 sequences.")
            if self.seeds != tuple(range(20)):
                raise ValueError("Canonical confirmatory seeds are frozen at 0--19.")


def confirmatory_config() -> ConfirmatoryConfig:
    return ConfirmatoryConfig()


def smoke_config() -> ConfirmatoryConfig:
    return ConfirmatoryConfig(
        n_blocks=2, seeds=(0, 1), make_plots=False, profile="smoke"
    )
