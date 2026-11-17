"""Scientific gates shared by the bridge stages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def filter_pure_ab(frame: pd.DataFrame) -> pd.DataFrame:
    if "context" not in frame:
        raise ValueError("Missing context column.")
    return frame.loc[frame.context.isin(["A", "B"])].copy().reset_index(drop=True)


def assert_rt_identity(frame: pd.DataFrame, *, tolerance=1e-6):
    difference = np.abs(
        frame.normalized_rt.to_numpy(dtype=float)
        - (1.0 - frame.execution_speed.to_numpy(dtype=float))
    )
    if not np.all(difference <= tolerance):
        raise ValueError("SRN RT identity failed: normalized_RT must equal 1 - execution_speed.")


def execution_sanity_table(frame: pd.DataFrame, *, window=10) -> pd.DataFrame:
    """Require the four prespecified learning directions for each execution condition."""
    required = {
        "seed", "model_condition", "global_sequence", "sequence_ce",
        "correct_token_confidence", "execution_speed", "normalized_rt",
    }
    if required - set(frame):
        raise ValueError(f"Training dynamics missing {sorted(required - set(frame))}.")
    assert_rt_identity(frame)
    rows = []
    directions = {
        "sequence_ce_decreases": ("sequence_ce", -1),
        "correct_token_confidence_increases": ("correct_token_confidence", 1),
        "execution_speed_increases": ("execution_speed", 1),
        "normalized_rt_decreases": ("normalized_rt", -1),
    }
    execution = frame[frame.model_condition.isin(["High", "Low"])].copy()
    for condition, group in execution.groupby("model_condition", sort=False):
        per_seed = []
        for seed, seed_frame in group.groupby("seed", sort=False):
            ordered = seed_frame.sort_values("global_sequence")
            width = min(int(window), max(1, len(ordered) // 2))
            record = {"seed": seed}
            for metric, (column, direction) in directions.items():
                change = float(ordered[column].tail(width).mean() - ordered[column].head(width).mean())
                record[metric] = direction * change > 0
                record[f"{metric}_change"] = change
            per_seed.append(record)
        seed_table = pd.DataFrame(per_seed)
        for metric in directions:
            fraction = float(seed_table[metric].mean())
            rows.append({
                "model_condition": condition, "metric": metric,
                "passing_seed_fraction": fraction, "required_seed_fraction": 0.5,
                "mean_late_minus_early": float(seed_table[f"{metric}_change"].mean()),
                "passed": bool(fraction >= 0.5),
            })
        rows.append({
            "model_condition": condition, "metric": "rt_identity",
            "passing_seed_fraction": 1.0, "required_seed_fraction": 1.0,
            "mean_late_minus_early": 0.0,
            "passed": True,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No High/Low execution rows were available for the sanity gate.")
    return result


def assert_corrected_srn_source(source_dir) -> dict:
    root = Path(source_dir).expanduser().resolve()
    path = root / "run_config_v2.json"
    if not path.exists():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    conditions = {item["name"]: item for item in record.get("conditions", [])}
    expected = {
        "High": (2.0, True), "Low": (0.02, True),
        "High_seq_only": (2.0, False),
    }
    for name, (scale, execution) in expected.items():
        item = conditions.get(name)
        if item is None or float(item.get("init_scale", -1)) != scale or bool(
            item.get("execution_objective")) != execution:
            raise ValueError(
                "Bridge requires High scale 2.0 + execution, Low scale + execution, "
                "and High scale 2.0 sequence-only; the legacy High scale .5 run is rejected."
            )
    if record.get("execution_sanity_gate") != "passed":
        raise ValueError("Corrected SRN execution_sanity_gate must be passed.")
    return record
