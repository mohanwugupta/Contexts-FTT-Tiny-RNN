from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: Path = Path("outputs/phase1f_shared_sequence_backbone")
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    timeline_audit_dir: Path = Path("analysis/srn_training_timeline_audit")
    sequence: tuple[int, ...] = (0, 1, 2, 0, 3, 1, 2, 3)
    context_alpha: float = 1.0
    seeds: tuple[int, ...] = tuple(range(20))
    conditions: tuple[str, ...] = ("High_monotonic", "Low_monotonic")
    probes: tuple[str, ...] = ("Mixed", "Novel", "NoContextProbe")
    epsilon: float = 1e-12
    free_rollout_entry_transition: int = 1
    bootstrap_samples: int = 2000
    random_seed: int = 20260920
    stage2_enabled: bool = False
    make_plots: bool = True
    profile: str = "final"

    def validate(self):
        if self.sequence != (0, 1, 2, 0, 3, 1, 2, 3):
            raise ValueError("Phase 1F sequence is frozen.")
        if self.conditions != ("High_monotonic", "Low_monotonic"):
            raise ValueError("Phase 1F requires canonical High and Low conditions.")
        if self.probes != ("Mixed", "Novel", "NoContextProbe"):
            raise ValueError("Phase 1F probe labels are frozen and distinct.")
        if self.free_rollout_entry_transition != 1:
            raise ValueError("Free rollout entry is preregistered at transition 1.")
        if self.stage2_enabled:
            raise ValueError("Optional Stage 2 training is disabled in Phase 1F.")
        if self.profile == "final" and self.seeds != tuple(range(20)):
            raise ValueError("Final Phase 1F requires canonical seeds 0-19.")
        if self.bootstrap_samples < 20:
            raise ValueError("At least 20 bootstrap draws are required.")

    def manifest(self):
        result = asdict(self)
        for key in ("output_dir", "checkpoint_dir", "timeline_audit_dir"):
            result[key] = str(result[key])
        return result


def smoke_config(output_dir: Path):
    return ExperimentConfig(output_dir=output_dir, seeds=(0,), bootstrap_samples=20,
                            profile="smoke", make_plots=True)
