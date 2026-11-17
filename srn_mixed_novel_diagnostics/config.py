"""Frozen choices for the representation-diagnostic analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagnosticConfig:
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    output_dir: Path = Path("outputs/srn_mixed_novel_diagnostics")
    conditions: tuple[str, ...] = ("High_monotonic", "Low_monotonic")
    seeds: tuple[int, ...] = tuple(range(20))
    fixed_sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    contexts: tuple[str, ...] = ("A", "B", "Mixed", "Novel", "NoContextProbe")
    cue_lambdas: tuple[float, ...] = tuple(round(i * .05, 2) for i in range(21))
    history_repetitions: int = 10
    decoder_c: float = 1.0
    random_seed: int = 20260919
    make_plots: bool = True
    profile: str = "final"

    def validate(self) -> None:
        if not self.seeds or not self.conditions:
            raise ValueError("At least one frozen condition and seed are required.")
        if len(self.fixed_sequence) != 8:
            raise ValueError("The registered FTT sequence must contain eight tokens.")
        if self.cue_lambdas[0] != 0 or self.cue_lambdas[-1] != 1:
            raise ValueError("The cue sweep must contain the exact A and B endpoints.")
        if tuple(sorted(self.cue_lambdas)) != self.cue_lambdas:
            raise ValueError("Cue sweep values must be ordered.")
        if self.history_repetitions < 2:
            raise ValueError("Matched terminal histories require at least two repetitions.")


def default_config(**changes) -> DiagnosticConfig:
    return DiagnosticConfig(**changes)


def smoke_config(**changes) -> DiagnosticConfig:
    values = dict(seeds=(0, 1), make_plots=True, profile="smoke")
    values.update(changes)
    return DiagnosticConfig(**values)
