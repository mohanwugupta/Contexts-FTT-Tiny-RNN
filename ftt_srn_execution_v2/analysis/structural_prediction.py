"""Seed-level structural prediction relating endpoints to probe displacement."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit(group: pd.DataFrame) -> dict:
    reliable = group[group.endpoint_reliable].dropna(
        subset=["rt_endpoint_signed_separation", "midpoint_deviation"]
    )
    n = len(reliable)
    if n < 2 or reliable.rt_endpoint_signed_separation.std(ddof=0) < 1e-12:
        return {"n_networks": n, "slope": np.nan, "intercept": np.nan,
                "correlation": np.nan, "r_squared": np.nan,
                "mean_interpolation_index": reliable.interpolation_index.mean(),
                "sd_interpolation_index": reliable.interpolation_index.std(ddof=0)}
    x = reliable.rt_endpoint_signed_separation.to_numpy(dtype=float)
    y = reliable.midpoint_deviation.to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    ss_total = float(np.square(y - y.mean()).sum())
    r_squared = 1.0 - float(np.square(y - predicted).sum()) / ss_total if ss_total > 0 else np.nan
    return {
        "n_networks": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "r_squared": r_squared,
        "mean_interpolation_index": float(reliable.interpolation_index.mean()),
        "sd_interpolation_index": float(reliable.interpolation_index.std(ddof=0)),
    }


def structural_prediction(interpolation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_condition", "checkpoint", "probe_type"]
    for values, group in interpolation.groupby(keys, sort=False):
        rows.append({**dict(zip(keys, values)), **_fit(group)})
    return pd.DataFrame(rows)
