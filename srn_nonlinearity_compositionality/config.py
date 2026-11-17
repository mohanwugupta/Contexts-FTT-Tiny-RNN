from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MechanismConfig:
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    reproduction_dir: Path = Path("outputs/srn_mixed_novel_diagnostics")
    output_dir: Path = Path("outputs/srn_nonlinearity_compositionality")
    conditions: tuple[str, ...] = ("High_monotonic", "Low_monotonic")
    seeds: tuple[int, ...] = tuple(range(20))
    sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    cue_lambdas: tuple[float, ...] = tuple(round(.05 * i, 2) for i in range(21))
    bilinear_ranks: tuple[int, ...] = (0, 1, 2, 3)
    decoder_c: float = 1.0
    bootstrap_samples: int = 1000
    random_seed: int = 20260919
    make_plots: bool = True
    profile: str = "final"

    def validate(self):
        if len(self.seeds) < 2:
            raise ValueError("LOSO decoding requires at least two seeds per condition.")
        if self.cue_lambdas != tuple(round(.05 * i, 2) for i in range(21)):
            raise ValueError("The frozen cue grid is 0:.05:1.")
        if self.bilinear_ranks != (0, 1, 2, 3):
            raise ValueError("F2 candidate ranks are frozen at 0,1,2,3.")
        if len(self.sequence) != 8:
            raise ValueError("The registered sequence contains eight tokens and seven transitions.")
