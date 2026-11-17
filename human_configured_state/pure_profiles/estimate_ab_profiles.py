"""Estimate participant A/B transition profiles without probe leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_ab_profiles(data: pd.DataFrame, *, shrinkage_strength: float = 8.0):
    pure = data[data.context.isin(["A", "B"])].copy()
    if pure.empty:
        raise ValueError("Pure-profile estimation requires A/B observations.")
    keys = ["participant_uid", "dataset", "counterbalance", "context", "transition_position"]
    cells = pure.groupby(keys, as_index=False).agg(
        raw_profile_log_rt=("log_rt", "mean"),
        profile_observations=("log_rt", "size"),
        within_cell_sd=("log_rt", "std"),
    )
    group = cells.groupby(
        ["dataset", "context", "transition_position"]
    ).raw_profile_log_rt.transform("mean")
    weight = cells.profile_observations / (cells.profile_observations + float(shrinkage_strength))
    cells["shrunk_profile_log_rt"] = weight * cells.raw_profile_log_rt + (1 - weight) * group
    index = ["participant_uid", "dataset", "counterbalance", "transition_position"]
    wide = cells.pivot_table(
        index=index, columns="context", values="shrunk_profile_log_rt", aggfunc="first"
    ).reset_index()
    counts = cells.pivot_table(
        index=index, columns="context", values="profile_observations", aggfunc="first"
    ).reset_index().rename(columns={"A": "n_a", "B": "n_b"})
    result = wide.rename(columns={"A": "r_a", "B": "r_b"}).merge(counts, on=index)
    result["midpoint"] = (result.r_a + result.r_b) / 2.0
    result["delta"] = result.r_a - result.r_b
    result["endpoint_separation"] = result.delta.abs()
    result["pure_source_contexts"] = "A|B"
    result["probe_rows_used_for_profile"] = 0
    result["shrinkage_strength"] = float(shrinkage_strength)
    return result.sort_values(
        ["dataset", "participant_uid", "transition_position"], kind="stable"
    ).reset_index(drop=True)


def compute_midpoint_deviation(
    probe: pd.DataFrame, profiles: pd.DataFrame, *, endpoint_epsilon: float
):
    profile_columns = [
        "participant_uid", "dataset", "counterbalance", "transition_position",
        "r_a", "r_b", "midpoint", "delta",
    ]
    merged = probe.merge(
        profiles[profile_columns],
        on=["participant_uid", "dataset", "transition_position"],
        how="inner", suffixes=("", "_profile"), validate="many_to_one",
    )
    merged["midpoint_deviation"] = merged.log_rt - merged.midpoint
    merged["endpoint_separation"] = merged.delta.abs()
    merged["ratio_reliable"] = merged.endpoint_separation >= float(endpoint_epsilon)
    raw = np.divide(
        merged.log_rt - merged.r_b,
        merged.r_a - merged.r_b,
        out=np.full(len(merged), np.nan),
        where=merged.delta.abs().to_numpy() > 1e-15,
    )
    merged["responsibility_q_unclipped"] = raw
    merged["responsibility_q"] = np.where(merged.ratio_reliable, raw, np.nan)
    merged["endpoint_epsilon"] = float(endpoint_epsilon)
    merged["ab_profiles_frozen_before_probe_fit"] = True
    merged["probe_rows_used_for_profile"] = 0
    return merged

