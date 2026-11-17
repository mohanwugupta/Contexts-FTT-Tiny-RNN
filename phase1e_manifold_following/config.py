from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


FROZEN_LAMBDAS = (.10, .20, .30, .40, .60, .70, .80, .90)
FROZEN_GATES = (.10, .90, 1.10, 1.10, .35, .75, .35)


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/phase1e_manifold_following")
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    phase1c_dir: Path = Path("outputs/phase1c_pareto_path_sufficiency")
    phase1d_dir: Path = Path("outputs/phase1d_constrained_retrofit")
    sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    seeds: tuple[int, ...] = tuple(range(20))
    lambda_train: tuple[float, ...] = FROZEN_LAMBDAS
    cue_lambdas: tuple[float, ...] = tuple(round(.05 * i, 2) for i in range(21))
    n_blocks: int = 17
    block_length: int = 10
    checkpoint_fractions: tuple[float, ...] = tuple(i / 10 for i in range(11))
    initial_trust_fraction: float = 1e-3
    minimum_trust_fraction: float = 1e-8
    maximum_trust_fraction: float = 5e-2
    shrink_factor: float = .5
    enlarge_factor: float = 1.5
    accept_ratio: float = .25
    enlarge_ratio: float = .75
    gamma_relative: float = 1e-6
    tangent_regularization_relative: float = 1e-4
    restoration_iterations: int = 4
    constraint_tolerance: float = 5e-3
    svd_relative_tolerance: float = 1e-7
    endpoint_drift_max: float = .10
    separation_ratio_min: float = .90
    separation_ratio_max: float = 1.10
    familiar_loss_ratio_max: float = 1.10
    path_residual_ratio_max: float = .35
    rescue_fraction_min: float = .75
    remaining_patch_gap_ratio_max: float = .35
    q_local_access_threshold: float = .50
    stagnation_reduction_threshold: float = .10
    constraint_limited_rejection_fraction: float = .50
    bootstrap_samples: int = 2000
    random_seed: int = 20260920
    wrong_path_seed_offset: int = 1_970_321
    rescue_denominator_tolerance: float = 1e-8
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
            raise ValueError("Phase 1E must use the registered motor sequence.")
        if self.lambda_train != FROZEN_LAMBDAS or .5 in self.lambda_train:
            raise ValueError("Phase 1E held-out lambda=.5 cannot enter training.")
        if self.cue_lambdas != tuple(round(.05 * i, 2) for i in range(21)):
            raise ValueError("Evaluation cue sweep is frozen at 0:.05:1.")
        if self.gates != FROZEN_GATES:
            raise ValueError("Phase 1C/1D success gates are frozen.")
        if not (0 < self.accept_ratio < self.enlarge_ratio < 1):
            raise ValueError("Invalid fixed trust-region thresholds.")
        if self.restoration_iterations < 1:
            raise ValueError("At least one nonlinear restoration iteration is required.")
        if self.profile == "final" and (self.seeds != tuple(range(20)) or self.updates != 170):
            raise ValueError("Final Phase 1E requires seeds 0-19 and 170 attempted updates.")
        if self.bootstrap_samples < 20:
            raise ValueError("At least 20 bootstrap draws are required.")

    def manifest(self):
        result = asdict(self)
        for name in ("output_dir", "checkpoint_dir", "phase1c_dir", "phase1d_dir"):
            result[name] = str(result[name])
        return result


def smoke_config(output_dir: Path):
    return ExperimentConfig(
        output_dir=output_dir, seeds=(0,), n_blocks=1, block_length=10,
        checkpoint_fractions=(0., .5, 1.), restoration_iterations=2,
        bootstrap_samples=20, make_plots=True, profile="smoke",
    )
