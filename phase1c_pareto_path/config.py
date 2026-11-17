from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/phase1c_pareto_path_sufficiency")
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    seeds: tuple[int, ...] = tuple(range(20))
    lambda_train: tuple[float, ...] = (.10, .20, .30, .40, .60, .70, .80, .90)
    cue_lambdas: tuple[float, ...] = tuple(round(.05 * i, 2) for i in range(21))
    alphas: tuple[float, ...] = (0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1., 3., 10., 30.)
    anchor_strengths: tuple[float, ...] = (.1, 1., 10., 30.)
    learning_rate: float = .005
    gradient_clip: float = 1.
    n_blocks: int = 17
    block_length: int = 10
    checkpoint_fractions: tuple[float, ...] = tuple(i / 10 for i in range(11))
    endpoint_drift_max: float = .10
    separation_ratio_min: float = .90
    separation_ratio_max: float = 1.10
    familiar_loss_ratio_max: float = 1.10
    path_residual_ratio_max: float = .35
    rescue_fraction_min: float = .75
    remaining_patch_gap_ratio_max: float = .35
    substantial_region_fraction: float = .10
    feasible_seed_fraction_min: float = .50
    specificity_rescue_advantage_min: float = .20
    specificity_pair_fraction_min: float = .60
    bootstrap_samples: int = 2000
    wrong_path_seed_offset: int = 1_970_321
    representative_anchor: float = 10.
    make_plots: bool = True
    profile: str = "final"

    @property
    def updates(self):
        return self.n_blocks * self.block_length

    @property
    def checkpoint_updates(self):
        return tuple(sorted({int(round(x * self.updates)) for x in self.checkpoint_fractions}))

    def validate(self):
        if self.sequence != (0, 1, 2, 0, 3, 1, 2, 3):
            raise ValueError("Phase 1C must use the registered sequence.")
        if .5 in self.lambda_train or self.lambda_train != (.10, .20, .30, .40, .60, .70, .80, .90):
            raise ValueError("T1 failed: the primary training cue set must exclude .50.")
        if self.cue_lambdas != tuple(round(.05 * i, 2) for i in range(21)):
            raise ValueError("Dense cue evaluation is frozen at 0:.05:1.")
        if self.profile == "final" and self.alphas != (0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1., 3., 10., 30.):
            raise ValueError("The final Pareto alpha grid is frozen.")
        if self.profile == "final" and self.anchor_strengths != (.1, 1., 10., 30.):
            raise ValueError("The final equal beta=gamma anchor grid is frozen.")
        if self.profile == "final" and (self.seeds != tuple(range(20)) or self.updates != 170):
            raise ValueError("Final Phase 1C requires 20 seeds and 170 updates.")
        gates = (self.endpoint_drift_max, self.separation_ratio_min, self.separation_ratio_max,
                 self.familiar_loss_ratio_max, self.path_residual_ratio_max,
                 self.rescue_fraction_min, self.remaining_patch_gap_ratio_max)
        if gates != (.10, .90, 1.10, 1.10, .35, .75, .35):
            raise ValueError("The strong-sufficiency criteria are frozen.")

    def manifest(self):
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["checkpoint_dir"] = str(self.checkpoint_dir)
        return values


def smoke_config(output_dir: Path):
    return ExperimentConfig(
        output_dir=output_dir, seeds=(0, 1), alphas=(0., .1), anchor_strengths=(1.,),
        n_blocks=2, bootstrap_samples=100, representative_anchor=1.,
        make_plots=True, profile="smoke",
    )
