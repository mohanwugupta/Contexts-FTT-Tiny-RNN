"""Shared-coordinate A/B trajectory-centroid geometry."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hidden_state_geometry(hidden: pd.DataFrame) -> pd.DataFrame:
    hidden_columns = sorted(
        [column for column in hidden if column.startswith("h") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )
    keys = ["seed", "model_condition", "checkpoint"]
    rows = []
    for values, group in hidden.groupby(keys, sort=False):
        centroids = {
            context: subset[hidden_columns].mean().to_numpy(dtype=float)
            for context, subset in group.groupby("context")
        }
        if "A" not in centroids or "B" not in centroids:
            continue
        a, b = centroids["A"], centroids["B"]
        axis = a - b
        squared_length = float(axis @ axis)
        distance = float(np.sqrt(squared_length))
        for context, centroid in centroids.items():
            projection = float((centroid - b) @ axis / squared_length) if squared_length > 1e-12 else np.nan
            projected_point = b + projection * axis if np.isfinite(projection) else np.full_like(b, np.nan)
            rows.append({
                **dict(zip(keys, values)),
                "context": context,
                "ab_centroid_distance": distance,
                "distance_to_a": float(np.linalg.norm(centroid - a)),
                "distance_to_b": float(np.linalg.norm(centroid - b)),
                "projection_on_ab_axis_b0_a1": projection,
                "off_axis_distance": float(np.linalg.norm(centroid - projected_point)),
                **{f"centroid_{name}": float(value) for name, value in zip(hidden_columns, centroid)},
            })
    return pd.DataFrame(rows)
