"""Prespecified causal-intervention configuration."""

from __future__ import annotations

from dataclasses import dataclass


CHECKPOINTS = (
    "early_after_initial_ab", "human_matched", "several_switches", "final"
)


@dataclass(frozen=True)
class CausalConfig:
    alpha_grid: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0,
                                         0.5, 1.0, 1.5, 2.0)
    recurrent_alpha_grid: tuple[float, ...] = (-1.0, 0.0, 1.0)
    primary_alpha_max: float = 1.0
    checkpoints: tuple[str, ...] = CHECKPOINTS
    primary_checkpoint: str = "human_matched"
    variance_threshold: float = 0.90
    endpoint_reliability_threshold: float = 0.02
    sequence_preservation_threshold: float = 0.90
    execution_collapse_threshold: float = 0.50
    seed_success_fraction: float = 0.80
    manifold_ridge: float = 1e-5
    random_seed_offset: int = 22000
    seed_limit: int | None = None
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "final"

    def validate(self):
        if self.primary_checkpoint != "human_matched":
            raise ValueError("The human-matched checkpoint is fixed as primary in v2.2.")
        if not set(self.checkpoints).issubset(CHECKPOINTS):
            raise ValueError("Unknown v2.1 checkpoint requested.")
        if self.primary_checkpoint not in self.checkpoints:
            raise ValueError("Primary checkpoint must be included.")
        if tuple(sorted(self.alpha_grid)) != self.alpha_grid or 0.0 not in self.alpha_grid:
            raise ValueError("Steering alpha grid must be ordered and contain zero.")
        if self.primary_alpha_max <= 0 or self.primary_alpha_max > max(abs(x) for x in self.alpha_grid):
            raise ValueError("Primary alpha range must be positive and inside the sweep.")
        if not 0 < self.variance_threshold <= 1:
            raise ValueError("Subspace variance threshold must be in (0,1].")
        if not 0 < self.seed_success_fraction <= 1:
            raise ValueError("Seed success fraction must be in (0,1].")
        if self.seed_limit is not None and self.seed_limit < 1:
            raise ValueError("seed_limit must be positive when provided.")


def smoke_config() -> CausalConfig:
    return CausalConfig(seed_limit=1, profile="smoke")


def final_config() -> CausalConfig:
    return CausalConfig()
