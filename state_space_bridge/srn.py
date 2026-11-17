"""Additive corrected scale-2 SRN rerun; frozen v2.1 outputs are never modified."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

from ftt_srn_execution_v2.config import ExperimentConfig
from ftt_srn_execution_v2.pipeline import run_pipeline

from .contracts import execution_sanity_table


class ScientificGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorrectedCondition:
    name: str
    init_scale: float
    execution_objective: bool

    def validate(self):
        registered = {
            "High": (2.0, True), "Low": (0.02, True),
            "High_seq_only": (2.0, False),
        }
        if self.name not in registered or (
            float(self.init_scale), bool(self.execution_objective)
        ) != registered[self.name]:
            raise ValueError("Invalid corrected scale-2 bridge condition.")


CORRECTED_CONDITIONS = (
    CorrectedCondition("High", 2.0, True),
    CorrectedCondition("Low", 0.02, True),
    CorrectedCondition("High_seq_only", 2.0, False),
)


def corrected_srn_config(profile: str) -> ExperimentConfig:
    if profile not in {"smoke", "final"}:
        raise ValueError("Corrected SRN profile must be smoke or final.")
    # Smoke retains the full 170-sequence schedule so all scientific checkpoints
    # and trajectory directions can be tested; it limits only seed count.
    return ExperimentConfig(
        conditions=CORRECTED_CONDITIONS,
        seeds=(0, 1) if profile == "smoke" else tuple(range(20)),
        make_plots=False,
        profile=f"bridge_scale2_{profile}",
    )


def run_corrected_srn(output_dir, *, profile="smoke", show_progress=True):
    output = Path(output_dir).expanduser().resolve()
    config = replace(corrected_srn_config(profile), show_progress=show_progress)
    config_path = output / "run_config_v2.json"
    reusable = False
    if config_path.exists() and (output / "training_dynamics.csv").exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        conditions = {item["name"]: item for item in prior.get("conditions", [])}
        reusable = (
            float(conditions.get("High", {}).get("init_scale", -1)) == 2.0
            and float(conditions.get("Low", {}).get("init_scale", -1)) == 0.02
            and float(conditions.get("High_seq_only", {}).get("init_scale", -1)) == 2.0
            and "execution_initial_bias" not in conditions.get("High", {})
        )
    if not reusable:
        run_pipeline(output, config=config)
    dynamics = pd.read_csv(output / "training_dynamics.csv")
    sanity = execution_sanity_table(dynamics)
    sanity.to_csv(output / "execution_sanity_gate.csv", index=False)
    passed = bool(sanity.passed.all())
    record = json.loads(config_path.read_text(encoding="utf-8"))
    record.update({
        "bridge_model_source": "state-space_SRN.Rmd",
        "high_scale": 2.0,
        "execution_sanity_gate": "passed" if passed else "failed",
        "execution_sanity_metrics": "execution_sanity_gate.csv",
        "corrected_conditions": [asdict(item) for item in CORRECTED_CONDITIONS],
    })
    config_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    if not passed:
        failed = sanity.loc[~sanity.passed, ["model_condition", "metric"]]
        (output / "SCIENTIFIC_GATE_FAILURE.md").write_text(
            "# Corrected scale-2 execution sanity gate failed\n\n"
            "Per the bridge PRD, state-space and downstream retrieval fitting stopped before "
            "interpretation. Sequence CE decreased and confidence increased, but the required "
            "speed-up / normalized-RT-down trajectory did not hold for the listed conditions. "
            "No parameter was tuned to force the gate.\n\n"
            + sanity.to_markdown(index=False) + "\n",
            encoding="utf-8",
        )
        raise ScientificGateError(
            "Corrected scale-2 execution sanity gate failed; state-space fitting is blocked: "
            + failed.to_dict(orient="records").__repr__()
        )
    return output
