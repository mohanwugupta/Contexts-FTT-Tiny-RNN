from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/phase1b_direct_task_space_path")
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    seeds: tuple[int, ...] = tuple(range(20))
    selection_seeds: tuple[int, ...] = (0, 1, 2)
    lambda_train: tuple[float, ...] = (.10, .20, .30, .40, .60, .70, .80, .90)
    cue_lambdas: tuple[float, ...] = tuple(round(.05 * i, 2) for i in range(21))
    alpha_grid: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.)
    beta_grid: tuple[float, ...] = (.1, 1., 10.)
    gamma_grid: tuple[float, ...] = (.1, 1., 10.)
    prior_curvature_alpha: float = 1e-4
    learning_rate: float = .005
    gradient_clip: float = 1.
    n_blocks: int = 17
    block_length: int = 10
    curvature_centers: tuple[float, ...] = (.25, .50, .75)
    curvature_delta: float = .10
    checkpoint_fractions: tuple[float, ...] = tuple(i / 10 for i in range(11))
    familiar_loss_ratio_max: float = 1.10
    separation_ratio_min: float = .90
    separation_ratio_max: float = 1.10
    endpoint_hidden_drift_ratio_max: float = .10
    endpoint_logit_drift_ratio_max: float = .10
    path_residual_ratio_max: float = .35
    rescue_fraction_min: float = .50
    remaining_patch_gap_ratio_max: float = .50
    mixed_q_tolerance: float = .15
    selection_path_ratio_max: float = .35
    make_plots: bool = True
    save_checkpoints: bool = True
    profile: str = "final"

    @property
    def updates(self) -> int:
        return self.n_blocks * self.block_length

    @property
    def checkpoint_updates(self) -> tuple[int, ...]:
        return tuple(sorted({int(round(x * self.updates)) for x in self.checkpoint_fractions}))

    def validate(self) -> None:
        if self.sequence != (0, 1, 2, 0, 3, 1, 2, 3):
            raise ValueError("Phase 1B must use the registered eight-token sequence.")
        if .50 in self.lambda_train:
            raise ValueError("T6 failed: held-out midpoint is present in Lambda_train.")
        if self.lambda_train != (.10, .20, .30, .40, .60, .70, .80, .90):
            raise ValueError("The primary non-midpoint training cue set is frozen.")
        if self.cue_lambdas != tuple(round(.05 * i, 2) for i in range(21)):
            raise ValueError("Dense evaluation must be 0:.05:1.")
        if not set(self.selection_seeds).issubset(self.seeds):
            raise ValueError("Selection seeds must be included in final seeds.")
        if self.block_length != 10 or self.updates < 1:
            raise ValueError("Fine-tuning uses the registered ten-sequence block schedule.")
        if self.profile == "final" and (self.seeds != tuple(range(20)) or self.updates != 170):
            raise ValueError("Final Phase 1B is frozen at 20 seeds and 170 updates.")
        frozen = (self.familiar_loss_ratio_max, self.separation_ratio_min,
                  self.separation_ratio_max, self.path_residual_ratio_max,
                  self.rescue_fraction_min, self.remaining_patch_gap_ratio_max,
                  self.mixed_q_tolerance)
        if frozen != (1.10, .90, 1.10, .35, .50, .50, .15):
            raise ValueError("Prespecified Phase 1B gate thresholds cannot be changed.")

    def manifest(self) -> dict:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["checkpoint_dir"] = str(self.checkpoint_dir)
        return values


def smoke_config(output_dir: Path) -> ExperimentConfig:
    return ExperimentConfig(
        output_dir=output_dir, seeds=(0, 1), selection_seeds=(0,), n_blocks=2,
        alpha_grid=(1e-3,), beta_grid=(1.,), gamma_grid=(1.,),
        make_plots=True, save_checkpoints=False, profile="smoke",
    )
