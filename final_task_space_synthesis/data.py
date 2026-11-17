from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import REQUIRED_SOURCE_CSVS, SynthesisConfig


def load_phase1c_sources(config: SynthesisConfig):
    source = Path(config.source_dir)
    report = source / "PHASE1C_PARETO_PATH_SUFFICIENCY_REPORT.md"
    manifest = source / "experiment_manifest.json"
    required = [source / name for name in REQUIRED_SOURCE_CSVS] + [report, manifest]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required Phase 1C artifacts:\n- " + "\n- ".join(missing))
    tables = {path.stem: pd.read_csv(path, low_memory=False)
              for path in required if path.suffix == ".csv"}
    metadata = json.loads(manifest.read_text())
    if sorted(tables["phase1c_seed_metrics"].seed.unique()) != list(range(20)) and config.profile == "final":
        raise ValueError("Final synthesis requires all 20 Phase 1C seeds.")
    return tables, report.read_text(), metadata


def audit_sequence_tables(config: SynthesisConfig, tables):
    names = (
        "phase1c_timeseries_finetuning", "phase1c_timeseries_midpoint_checkpoints",
        "phase1c_timeseries_familiar_schedule", "phase1c_timeseries_full_probe_schedule",
        "phase1c_timeseries_cue_sweep",
    )
    rows = []
    for name in names:
        frame = tables[name]
        if "transition_count" not in frame:
            raise AssertionError(f"T11/T12 failed: {name} lacks sequence transition counts.")
        values = sorted(frame.transition_count.dropna().unique().tolist())
        passed = values == [config.transitions_per_sequence]
        rows.append({"table": name, "rows": len(frame), "transition_counts": str(values),
                     "sequence_level_only": passed})
        if not passed:
            raise AssertionError(f"T11/T12 failed: {name} contains non-sequence aggregates {values}.")
    return pd.DataFrame(rows)
