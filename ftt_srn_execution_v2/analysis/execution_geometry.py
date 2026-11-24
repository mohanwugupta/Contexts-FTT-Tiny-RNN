"""Alignment of the speed head with the A/B representational axis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _correlation(x, y):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def execution_geometry(hidden: pd.DataFrame, model, *, checkpoint: str) -> pd.DataFrame:
    hidden_columns = sorted(
        [column for column in hidden if column.startswith("h") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    a = hidden[hidden.context == "A"][hidden_columns].mean().to_numpy(dtype=float)
    b = hidden[hidden.context == "B"][hidden_columns].mean().to_numpy(dtype=float)
    axis = a - b
    squared_length = float(axis @ axis)
    head = model.execution_head.weight.detach().cpu().numpy().reshape(-1)
    denominator = float(np.linalg.norm(head) * np.linalg.norm(axis))
    cosine = float(head @ axis / denominator) if denominator > 1e-12 else np.nan
    matrix = hidden[hidden_columns].to_numpy(dtype=float)
    projection = ((matrix - b) @ axis / squared_length) if squared_length > 1e-12 else np.full(len(hidden), np.nan)
    work = hidden[["context", "execution_speed"]].copy()
    work["projection"] = projection
    rows = []
    for context, group in work.groupby("context", sort=False):
        rows.append({
            "checkpoint": checkpoint,
            "context": context,
            "execution_head_ab_axis_cosine": cosine,
            "projection_speed_correlation_all_contexts": _correlation(
                work.projection, work.execution_speed
            ),
            "projection_speed_correlation_within_context": _correlation(
                group.projection, group.execution_speed
            ),
            "mean_ab_axis_projection": float(group.projection.mean()),
            "mean_execution_speed": float(group.execution_speed.mean()),
            "n_token_states": len(group),
        })
    return pd.DataFrame(rows)
