"""Within-network interpolation relative to frozen pure-context endpoints."""

from __future__ import annotations

import numpy as np
import pandas as pd


def probe_interpolation(probes: pd.DataFrame, *, reliable_separation: float) -> pd.DataFrame:
    keys = ["seed", "model_condition", "checkpoint"]
    pure = probes[probes.context.isin(["A", "B"])].pivot_table(
        index=keys, columns="context", values="normalized_rt", aggfunc="first"
    ).reset_index()
    pure = pure.rename(columns={"A": "rt_a", "B": "rt_b"})
    selected = probes[~probes.context.isin(["A", "B"])].drop(
        columns=["probe_type"], errors="ignore"
    ).copy()
    result = selected.merge(pure, on=keys, how="left", validate="many_to_one")
    result = result.rename(columns={"context": "probe_type", "normalized_rt": "rt_probe"})
    result["rt_endpoint_signed_separation"] = result.rt_a - result.rt_b
    result["rt_endpoint_absolute_separation"] = result.rt_endpoint_signed_separation.abs()
    result["rt_endpoint_midpoint"] = (result.rt_a + result.rt_b) / 2.0
    result["midpoint_deviation"] = result.rt_probe - result.rt_endpoint_midpoint
    result["endpoint_reliable"] = (
        result.rt_endpoint_absolute_separation >= float(reliable_separation)
    )
    result["interpolation_index"] = np.where(
        result.endpoint_reliable,
        (result.rt_probe - result.rt_b) / result.rt_endpoint_signed_separation,
        np.nan,
    )
    lower = result[["rt_a", "rt_b"]].min(axis=1)
    upper = result[["rt_a", "rt_b"]].max(axis=1)
    result["between_endpoints"] = result.rt_probe.between(lower, upper, inclusive="both")
    result["interpolation_status"] = np.select(
        [
            ~result.endpoint_reliable,
            result.between_endpoints,
            result.rt_probe < lower,
            result.rt_probe > upper,
        ],
        ["endpoint_too_close", "between", "below_both", "above_both"],
        default="undefined",
    )
    keep = keys + [
        "probe_type", "rt_a", "rt_b", "rt_probe", "rt_endpoint_signed_separation",
        "rt_endpoint_absolute_separation", "rt_endpoint_midpoint", "midpoint_deviation",
        "endpoint_reliable", "interpolation_index", "between_endpoints",
        "interpolation_status",
    ]
    return result[keep]
