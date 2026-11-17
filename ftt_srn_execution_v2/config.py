"""Prespecified architecture, execution ecology, and fixed FTT exposure."""

from __future__ import annotations

from dataclasses import dataclass


FIXED_SEQUENCE = (0, 1, 2, 0, 3, 1, 2, 3)


@dataclass(frozen=True)
class ModelCondition:
    name: str
    init_scale: float
    execution_objective: bool

    def validate(self):
        registered = {
            "High": (0.5, True),
            "Low": (0.02, True),
            "High_seq_only": (0.5, False),
        }
        if self.name not in registered:
            raise ValueError(f"Unknown model condition {self.name!r}.")
        if (float(self.init_scale), bool(self.execution_objective)) != registered[self.name]:
            raise ValueError(f"Condition {self.name} does not match its preregistered settings.")


HIGH_EXECUTION = ModelCondition("High", 0.5, True)
LOW_EXECUTION = ModelCondition("Low", 0.02, True)
HIGH_SEQUENCE_ONLY = ModelCondition("High_seq_only", 0.5, False)
MODEL_CONDITIONS = (HIGH_EXECUTION, LOW_EXECUTION, HIGH_SEQUENCE_ONLY)


@dataclass(frozen=True)
class ExecutionConfig:
    kappa: float = 12.0
    delta: float = 0.05
    c_error: float = 1.0
    # Calibrated once, before any network probes.  At .25 the optimal policy is
    # monotone in confidence and includes interior speeds without making the
    # execution term larger than the sequence-learning objective.
    c_time: float = 0.25
    lambda_exec: float = 0.20
    confidence_grid_size: int = 21
    speed_grid_size: int = 1001

    def validate(self):
        if self.kappa <= 0 or self.c_error <= 0 or self.c_time <= 0:
            raise ValueError("Execution environment costs and sharpness must be positive.")
        if not 0 < self.lambda_exec <= 1:
            raise ValueError("lambda_exec must be in (0,1].")


@dataclass(frozen=True)
class ExperimentConfig:
    hidden_dim: int = 8
    context_alpha: float = 1.0
    learning_rate: float = 0.005
    block_length: int = 10
    n_blocks: int = 17
    fixed_sequence: tuple[int, ...] = FIXED_SEQUENCE
    checkpoint_sequences: tuple[int, ...] = (20, 40, 80, 170)
    checkpoint_labels: tuple[str, ...] = (
        "early_after_initial_ab", "human_matched", "several_switches", "final"
    )
    seeds: tuple[int, ...] = tuple(range(20))
    conditions: tuple[ModelCondition, ...] = MODEL_CONDITIONS
    execution: ExecutionConfig = ExecutionConfig()
    gradient_clip: float = 1.0
    reliable_rt_separation: float = 0.02
    mastery_threshold: float = 0.80
    grounding_window_blocks: int = 8
    force_probe_interpretation: bool = False
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "final"

    @property
    def training_sequences(self) -> int:
        return int(self.block_length * self.n_blocks)

    def resolved_checkpoint_labels(self) -> tuple[str, ...]:
        """Return labels for requested checkpoints, including short test runs.

        The preregistered run supplies all four semantic labels.  Small smoke
        configurations often change only the checkpoint sequence numbers, so
        give their terminal snapshot the invariant ``final`` label and use
        stable early labels for any preceding snapshots.
        """
        if len(self.checkpoint_sequences) == len(self.checkpoint_labels):
            return self.checkpoint_labels
        if not self.checkpoint_sequences or self.checkpoint_sequences[-1] != self.training_sequences:
            raise ValueError("Custom checkpoints require labels unless the last checkpoint is final.")
        early = ("early_after_initial_ab", "human_matched", "several_switches")
        count = len(self.checkpoint_sequences) - 1
        if count > len(early):
            raise ValueError("Every nonstandard checkpoint requires a label.")
        return early[:count] + ("final",)

    def validate(self):
        if self.hidden_dim != 8:
            raise ValueError("The v2.1 architecture is fixed at eight hidden units.")
        if self.context_alpha != 1.0:
            raise ValueError("Context strength is fixed at 1.0.")
        if self.learning_rate != 0.005:
            raise ValueError("The validated FTT learning rate is fixed at .005.")
        if self.block_length != 10:
            raise ValueError("The FTT block length is fixed at ten sequences.")
        if self.n_blocks < 2:
            raise ValueError("At least one complete A/B exposure is required.")
        if self.grounding_window_blocks < 2:
            raise ValueError("Execution grounding needs at least one late A and B block.")
        if len(self.fixed_sequence) != 8 or set(self.fixed_sequence) - set(range(4)):
            raise ValueError("The validated eight-item/four-key sequence must be preserved.")
        self.resolved_checkpoint_labels()
        if max(self.checkpoint_sequences) > self.training_sequences:
            raise ValueError("A checkpoint cannot exceed fixed training exposure.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        for condition in self.conditions:
            condition.validate()
        self.execution.validate()


def smoke_config() -> ExperimentConfig:
    return ExperimentConfig(
        n_blocks=4,
        checkpoint_sequences=(20, 40),
        checkpoint_labels=("early_after_initial_ab", "final"),
        seeds=(0,),
        profile="smoke",
    )


def pilot_config() -> ExperimentConfig:
    return ExperimentConfig(seeds=tuple(range(5)), profile="pilot")


def final_config() -> ExperimentConfig:
    return ExperimentConfig()
