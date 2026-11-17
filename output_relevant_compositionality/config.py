from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path=Path("outputs/output_relevant_compositionality")
    baseline_checkpoint_dir: Path=Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    sequence: tuple[int,...]=(0,1,2,0,3,1,2,3)
    seeds: tuple[int,...]=tuple(range(20))
    selection_seeds: tuple[int,...]=(0,1,2)
    high_scale: float=2.0
    low_scale: float=.02
    learning_rate: float=.005
    gradient_clip: float=1.0
    block_length: int=10
    n_blocks: int=17
    curvature_centers: tuple[float,...]=(.25,.5,.75)
    curvature_delta: float=.10
    regularizer_grid: tuple[float,...]=(0.,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1)
    separation_grid: tuple[float,...]=(.1,1.,10.)
    cue_lambdas: tuple[float,...]=tuple(round(.05*i,2) for i in range(21))
    separation_preservation: float=.90
    familiar_loss_tolerance: float=.10
    mixed_loss_ratio_gate: float=.75
    curvature_ratio_gate: float=.50
    low_separation_multiplier: float=2.0
    phase2_sequences: int=8
    phase2_contexts: int=6
    phase2_pretraining_epochs: int=30
    phase2_transform_magnitude: float=.25
    phase2_multiplicative_rank: int=2
    random_seed: int=20260920
    make_plots: bool=True
    profile: str="final"

    @property
    def training_sequences(self): return self.block_length*self.n_blocks

    def validate(self):
        if len(self.seeds)<10 and self.profile=="final":
            raise ValueError("Final Phase 1 requires at least ten seeds.")
        if not set(self.selection_seeds).issubset(self.seeds):
            raise ValueError("Hyperparameter selection seeds must be final experiment seeds.")
        if self.cue_lambdas != tuple(round(.05*i,2) for i in range(21)):
            raise ValueError("Evaluation cue grid is frozen at 0:.05:1.")
        if not 0<self.curvature_delta<.25:
            raise ValueError("Curvature delta must keep all selected triplets inside [0,1].")
        if self.mixed_loss_ratio_gate!=.75 or self.curvature_ratio_gate!=.50:
            raise ValueError("Prespecified Phase 1 substantial-improvement gates are frozen.")
