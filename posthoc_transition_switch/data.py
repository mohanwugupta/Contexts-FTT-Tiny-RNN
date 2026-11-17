"""Strict loaders for historical transitions and saved frozen probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PosthocSwitchConfig


TRANSITION_COLUMNS = (
    "stage", "cohort", "overlap_count", "condition", "seed",
    "training_sequence", "block_index", "context", "within_block_sequence",
    "is_context_switch", "transition_position", "normalized_RT",
)


def _require_columns(frame: pd.DataFrame, required, label: str):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _boolean_values(series: pd.Series) -> pd.Series:
    """Parse saved boolean values without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin({"true", "false", "1", "0"})
    if unknown.any():
        raise ValueError(
            "Saved switch flags contain non-boolean values: "
            f"{sorted(series[unknown].astype(str).unique())}"
        )
    return normalized.isin({"true", "1"})


def validate_transition_history(
    frame: pd.DataFrame, config: PosthocSwitchConfig,
) -> pd.DataFrame:
    config.validate()
    _require_columns(frame, TRANSITION_COLUMNS, "Historical transition table")
    selected = frame[
        frame.stage.eq("stage2_layout")
        & frame.cohort.eq("confirmatory")
        & frame.overlap_count.eq(0)
        & frame.condition.isin(config.conditions)
    ].copy()
    if len(selected) != config.expected_transition_rows:
        raise ValueError(
            "Historical transition row count mismatch: "
            f"expected {config.expected_transition_rows}, found {len(selected)}."
        )
    expected_seeds = set(config.expected_seeds)
    for condition in config.conditions:
        observed = set(selected[selected.condition.eq(condition)].seed.astype(int))
        if observed != expected_seeds:
            raise ValueError(
                f"{condition} seed set mismatch: expected {sorted(expected_seeds)}, "
                f"found {sorted(observed)}."
            )
    if set(selected.block_index.astype(int)) != set(range(1, config.n_blocks + 1)):
        raise ValueError("Historical transitions do not contain every registered block.")
    if set(selected.within_block_sequence.astype(int)) != set(
        range(1, config.block_length + 1)
    ):
        raise ValueError("Historical transitions do not contain within-block 1--10.")
    if set(selected.transition_position.astype(int)) != set(config.transition_positions):
        raise ValueError("Historical transitions do not contain positions 1--7.")
    keys = [
        "condition", "seed", "block_index", "within_block_sequence",
        "transition_position",
    ]
    if selected.duplicated(keys).any():
        raise ValueError("Historical transition keys are not unique.")
    expected_context = selected.block_index.map(lambda block: "A" if block % 2 else "B")
    if not selected.context.eq(expected_context).all():
        raise ValueError("Context labels do not follow the frozen A10/B10 schedule.")
    expected_switch = (
        selected.within_block_sequence.eq(config.switch_sequence)
        & selected.block_index.ge(config.first_switch_block)
    )
    selected["is_context_switch"] = _boolean_values(selected.is_context_switch)
    if not selected.is_context_switch.eq(expected_switch).all():
        raise ValueError("Saved switch flags disagree with the frozen switch definition.")
    if not np.isfinite(selected.normalized_RT.to_numpy(float)).all():
        raise ValueError("Historical High/Low transition RT contains missing/nonfinite values.")
    selected["seed"] = selected.seed.astype(int)
    selected["block_index"] = selected.block_index.astype(int)
    selected["within_block_sequence"] = selected.within_block_sequence.astype(int)
    selected["transition_position"] = selected.transition_position.astype(int)
    return selected.sort_values(keys).reset_index(drop=True)


def load_transition_history(
    path: str | Path, config: PosthocSwitchConfig,
) -> pd.DataFrame:
    return validate_transition_history(pd.read_csv(path), config)


def load_frozen_probes(
    path: str | Path, config: PosthocSwitchConfig,
) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source)
    required = (
        "stage", "cohort", "condition", "seed", "probe_context",
        "transition_position", "normalized_RT",
    )
    _require_columns(frame, required, "Frozen probe table")
    selected = frame[
        frame.stage.eq("stage2_layout")
        & frame.cohort.eq("confirmatory")
        & frame.condition.isin(config.conditions)
        & frame.probe_context.isin(["A", "B"])
    ].copy()
    expected = (
        len(config.conditions) * len(config.expected_seeds)
        * len(config.transition_positions) * 2
    )
    if len(selected) != expected:
        raise ValueError(
            f"Frozen A/B probe rows mismatch: expected {expected}, found {len(selected)}."
        )
    keys = ["condition", "seed", "probe_context", "transition_position"]
    if selected.duplicated(keys).any():
        raise ValueError("Frozen A/B probe keys are not unique.")
    if not np.isfinite(selected.normalized_RT.to_numpy(float)).all():
        raise ValueError("Frozen A/B probe RT contains missing/nonfinite values.")
    return selected.sort_values(keys).reset_index(drop=True)
