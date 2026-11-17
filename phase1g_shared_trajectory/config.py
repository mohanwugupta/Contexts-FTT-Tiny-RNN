"""Frozen Phase 1G specification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from confirmatory_srn.config import FIXED_SEQUENCE


G0 = "G0_baseline_high"
G1 = "G1_context_dropout"
G2 = "G2_shared_prediction"
G3 = "G3_shared_trajectory"
G4 = "G4_prediction_plus_trajectory"
G5 = "G5_dropout_plus_trajectory"
LOW = "Low_reference"
INTERVENTIONS = (G0, G1, G2, G3, G4, G5)
ALL_CONDITIONS = INTERVENTIONS + (LOW,)
PROBES = ("A", "B", "Mixed", "NoContext", "Novel1", "Novel2", "Novel3", "Novel4")


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str = "outputs/phase1g_shared_trajectory"
    hidden_dim: int = 8
    token_dim: int = 4
    context_dim: int = 3
    high_scale: float = 2.0
    low_scale: float = .02
    context_alpha: float = 1.0
    sequence: tuple[int, ...] = FIXED_SEQUENCE
    learning_rate: float = .005
    gradient_clip: float = 1.0
    block_length: int = 10
    n_blocks: int = 17
    seeds: tuple[int, ...] = tuple(range(20))
    calibration_seeds: tuple[int, ...] = tuple(range(5))
    dropout_candidates: tuple[float, ...] = (.025, .05, .10)
    prediction_candidates: tuple[float, ...] = (.01, .05, .10, .25)
    trajectory_candidates: tuple[float, ...] = (.01, .05, .10, .25)
    lambda_execution: float = .20
    normative_c_time: float = .50
    normative_c_risk: float = 1.0
    normative_gamma: float = 2.0
    familiar_loss_ratio_limit: float = 1.10
    separation_ratio_limit: float = .75
    dropout_rng_offset: int = 1_071_003
    include_g5: bool = True
    include_low_reference: bool = True
    make_plots: bool = True
    profile: str = "confirmatory"

    @property
    def transitions(self): return len(self.sequence) - 1

    @property
    def training_sequences(self): return self.block_length * self.n_blocks

    @property
    def conditions(self):
        values = list(INTERVENTIONS if self.include_g5 else INTERVENTIONS[:-1])
        if self.include_low_reference: values.append(LOW)
        return tuple(values)

    @property
    def config_sha256(self):
        payload = asdict(self).copy(); payload.pop("output_dir", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def validate(self):
        if (self.hidden_dim, self.token_dim, self.context_dim) != (8, 4, 3):
            raise AssertionError("FAIL: NONCANONICAL_SRN")
        if self.high_scale != 2.0 or self.low_scale != .02 or self.context_alpha != 1.0:
            raise AssertionError("FAIL: NONCANONICAL_SRN")
        if self.sequence != FIXED_SEQUENCE or self.learning_rate != .005 or self.gradient_clip != 1.0:
            raise AssertionError("FAIL: NONCANONICAL_SRN")
        if self.block_length != 10 or self.n_blocks < 2:
            raise AssertionError("FAIL: NONCANONICAL_SCHEDULE")
        if not set(self.calibration_seeds).issubset(self.seeds):
            raise ValueError("Calibration seeds must be a subset of experiment seeds.")
        if any(x <= 0 for values in (self.dropout_candidates, self.prediction_candidates,
                                      self.trajectory_candidates) for x in values):
            raise ValueError("Auxiliary candidates must be positive.")
        if self.profile == "confirmatory" and (self.n_blocks != 17 or self.seeds != tuple(range(20))):
            raise ValueError("Confirmatory exposure is 170 sequences for seeds 0--19.")


def confirmatory_config(output=None):
    return ExperimentConfig(output_dir=str(output or "outputs/phase1g_shared_trajectory"))


def smoke_config(output=None):
    return ExperimentConfig(
        output_dir=str(output or "outputs/phase1g_shared_trajectory_smoke"),
        n_blocks=2, seeds=(0,), calibration_seeds=(0,),
        dropout_candidates=(.05,), prediction_candidates=(.05,),
        trajectory_candidates=(.05,), include_low_reference=True,
        make_plots=True, profile="smoke",
    )


def load_config(path=None, *, output=None, profile="confirmatory"):
    if path is None:
        return smoke_config(output) if profile == "smoke" else confirmatory_config(output)
    payload = json.loads(Path(path).read_text())
    if output is not None: payload["output_dir"] = str(output)
    for key in ("sequence", "seeds", "calibration_seeds", "dropout_candidates",
                "prediction_candidates", "trajectory_candidates"):
        if key in payload: payload[key] = tuple(payload[key])
    config = ExperimentConfig(**payload); config.validate(); return config
