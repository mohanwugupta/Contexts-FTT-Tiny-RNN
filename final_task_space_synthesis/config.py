from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SynthesisConfig:
    source_dir: Path = Path("outputs/phase1c_pareto_path_sufficiency")
    output_dir: Path = Path("final_mechanistic_synthesis")
    checkpoint_dir: Path = Path("outputs/confirmatory_srn_final/confirmatory_checkpoints")
    bootstrap_samples: int = 2000
    random_seed: int = 20260920
    endpoint_drift_max: float = .10
    separation_ratio_min: float = .90
    separation_ratio_max: float = 1.10
    familiar_loss_ratio_max: float = 1.10
    path_residual_max: float = .35
    rescue_fraction_min: float = .75
    remaining_gap_ratio_max: float = .35
    transitions_per_sequence: int = 7
    profile: str = "final"

    def validate(self):
        frozen = (self.endpoint_drift_max, self.separation_ratio_min,
                  self.separation_ratio_max, self.familiar_loss_ratio_max,
                  self.path_residual_max, self.rescue_fraction_min,
                  self.remaining_gap_ratio_max)
        if frozen != (.10, .90, 1.10, 1.10, .35, .75, .35):
            raise ValueError("Phase 1C preregistered thresholds cannot change in synthesis.")
        if self.transitions_per_sequence != 7:
            raise ValueError("Every final time-series point must aggregate seven transitions.")
        if self.bootstrap_samples < 100:
            raise ValueError("Synthesis requires at least 100 seed-bootstrap draws.")

    def manifest(self):
        result = asdict(self)
        for key in ("source_dir", "output_dir", "checkpoint_dir"):
            result[key] = str(result[key])
        return result


REQUIRED_SOURCE_CSVS = (
    "phase1c_pareto_grid.csv", "phase1c_seed_metrics.csv",
    "phase1c_endpoint_metrics.csv", "phase1c_path_metrics.csv",
    "phase1c_specificity_controls.csv", "phase1c_rescue_metrics.csv",
    "phase1c_feasible_frontier.csv", "phase1c_timeseries_finetuning.csv",
    "phase1c_timeseries_midpoint_checkpoints.csv",
    "phase1c_timeseries_familiar_schedule.csv",
    "phase1c_timeseries_full_probe_schedule.csv",
    "phase1c_timeseries_cue_sweep.csv",
)
