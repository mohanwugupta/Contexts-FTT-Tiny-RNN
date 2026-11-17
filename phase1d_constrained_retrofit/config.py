from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


FROZEN_LAMBDAS = (.10, .20, .30, .40, .60, .70, .80, .90)
FROZEN_GATES = (.10, .90, 1.10, 1.10, .35, .75, .35)


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/phase1d_constrained_retrofit")
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    phase1c_dir: Path = Path("outputs/phase1c_pareto_path_sufficiency")
    sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    seeds: tuple[int, ...] = tuple(range(20))
    calibration_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    lambda_train: tuple[float, ...] = FROZEN_LAMBDAS
    cue_lambdas: tuple[float, ...] = tuple(round(.05 * i, 2) for i in range(21))
    learning_rate_candidates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    rho_relative_candidates: tuple[float, ...] = (1e-8, 1e-6, 1e-4)
    n_blocks: int = 17
    block_length: int = 10
    checkpoint_fractions: tuple[float, ...] = tuple(i / 10 for i in range(11))
    trust_region_fraction: float = 1e-3
    line_search_retries: int = 6
    constraint_absolute_tolerance: float = 5e-3
    svd_relative_tolerance: float = 1e-7
    endpoint_drift_max: float = .10
    separation_ratio_min: float = .90
    separation_ratio_max: float = 1.10
    familiar_loss_ratio_max: float = 1.10
    path_residual_ratio_max: float = .35
    rescue_fraction_min: float = .75
    remaining_patch_gap_ratio_max: float = .35
    bootstrap_samples: int = 2000
    random_seed: int = 20260920
    wrong_path_seed_offset: int = 1_970_321
    phase1c_plot_cell: str = "P1_correct_logit_alpha_10_anchor_30"
    rescue_denominator_tolerance: float = 1e-8
    stage2_enabled: bool = False
    stage2_local_lambdas: tuple[float, ...] = (.10, .20, .80, .90)
    stage2_composition_alpha: float = .10
    stage2_learning_rate: float = .005
    make_plots: bool = True
    profile: str = "final"

    @property
    def updates(self):
        return self.n_blocks * self.block_length

    @property
    def checkpoint_updates(self):
        return tuple(sorted({int(round(x * self.updates)) for x in self.checkpoint_fractions}))

    @property
    def gates(self):
        return (self.endpoint_drift_max, self.separation_ratio_min,
                self.separation_ratio_max, self.familiar_loss_ratio_max,
                self.path_residual_ratio_max, self.rescue_fraction_min,
                self.remaining_patch_gap_ratio_max)

    def validate(self):
        if self.sequence != (0, 1, 2, 0, 3, 1, 2, 3):
            raise ValueError("Phase 1D must use the registered motor sequence.")
        if self.lambda_train != FROZEN_LAMBDAS or .5 in self.lambda_train:
            raise ValueError("T1: Phase 1D must reuse the exact non-midpoint Phase 1C cues.")
        if self.cue_lambdas != tuple(round(.05 * i, 2) for i in range(21)):
            raise ValueError("Cue-sweep values are frozen at 0:.05:1.")
        if self.gates != FROZEN_GATES:
            raise ValueError("T13: Phase 1C-comparable thresholds cannot change.")
        if not self.learning_rate_candidates or not self.rho_relative_candidates:
            raise ValueError("Global numerical candidate sets cannot be empty.")
        if self.profile == "final":
            if self.seeds != tuple(range(20)) or self.updates != 170:
                raise ValueError("Final Phase 1D requires seeds 0–19 and 170 updates.")
            if self.calibration_seeds != (0, 1, 2, 3, 4):
                raise ValueError("Final calibration seeds are frozen at 0–4.")
        if self.bootstrap_samples < 20:
            raise ValueError("At least 20 bootstrap draws are required.")
        if .5 in self.stage2_local_lambdas:
            raise ValueError("Optional Stage 2 must also hold out lambda=.5.")

    def manifest(self):
        result = asdict(self)
        for name in ("output_dir", "checkpoint_dir", "phase1c_dir"):
            result[name] = str(result[name])
        return result


def smoke_config(output_dir: Path):
    return ExperimentConfig(
        output_dir=output_dir, seeds=(0,), calibration_seeds=(0,),
        learning_rate_candidates=(3e-4,), rho_relative_candidates=(1e-6,),
        n_blocks=1, block_length=10, checkpoint_fractions=(0., .5, 1.),
        bootstrap_samples=20, make_plots=True, profile="smoke",
    )
