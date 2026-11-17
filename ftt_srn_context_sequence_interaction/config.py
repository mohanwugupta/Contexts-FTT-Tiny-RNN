"""Prespecified configuration for the PRD v2.3 frozen analysis."""

from __future__ import annotations

from dataclasses import dataclass


CHECKPOINTS = (
    "early_after_initial_ab", "human_matched", "several_switches", "final"
)


@dataclass(frozen=True)
class InteractionConfig:
    checkpoints: tuple[str, ...] = CHECKPOINTS
    primary_checkpoint: str = "human_matched"
    endpoint_reliability_threshold: float = 0.02
    sequence_preservation_threshold: float = 0.90
    seed_success_fraction: float = 0.80
    manifold_ridge: float = 1e-5
    regression_ridge: float = 1e-8
    random_seed_offset: int = 23000
    seed_limit: int | None = None
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "final"

    def validate(self):
        if self.primary_checkpoint != "human_matched":
            raise ValueError("The human-matched checkpoint is fixed as primary in v2.3.")
        if not set(self.checkpoints).issubset(CHECKPOINTS):
            raise ValueError("Unknown v2.1 checkpoint requested.")
        if self.primary_checkpoint not in self.checkpoints:
            raise ValueError("Primary checkpoint must be included.")
        if not 0 < self.seed_success_fraction <= 1:
            raise ValueError("seed_success_fraction must be in (0,1].")
        if not 0 < self.sequence_preservation_threshold <= 1:
            raise ValueError("sequence_preservation_threshold must be in (0,1].")
        if self.manifold_ridge <= 0 or self.regression_ridge < 0:
            raise ValueError("Ridge constants must be nonnegative and manifold ridge positive.")
        if self.seed_limit is not None and self.seed_limit < 2:
            raise ValueError("Leave-one-seed-out analysis requires at least two seeds.")


def smoke_config() -> InteractionConfig:
    return InteractionConfig(seed_limit=2, profile="smoke")


def final_config() -> InteractionConfig:
    return InteractionConfig()

