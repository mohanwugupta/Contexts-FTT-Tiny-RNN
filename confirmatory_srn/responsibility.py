"""Endpoint, midpoint, responsibility, and standardized bridge tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ConfirmatoryConfig


def probe_responsibility_tables(
    probes: pd.DataFrame, config: ConfirmatoryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    execution = probes[probes.execution_active.eq(True)].copy()
    index = ["condition", "seed", "transition_position"]
    pivot = execution.pivot_table(
        index=index, columns="probe_context", values="normalized_RT", aggfunc="first"
    ).reset_index()
    required = {"A", "B", "Mixed_norm", "Novel"}
    if not required.issubset(pivot.columns):
        raise ValueError("Frozen probes are missing an A/B/Mixed/Novel endpoint.")
    probe_contexts = ["Mixed_norm", "Novel"]
    if "Mixed_additive" in pivot:
        probe_contexts.append("Mixed_additive")
    rows = []
    for source in pivot.itertuples(index=False):
        a_rt, b_rt = float(source.A), float(source.B)
        difference = a_rt - b_rt
        midpoint = (a_rt + b_rt) / 2
        usable = abs(difference) >= config.responsibility_min_abs_denominator
        for context in probe_contexts:
            probe_rt = float(getattr(source, context))
            rows.append({
                "condition": source.condition,
                "seed": int(source.seed),
                "probe_context": context,
                "transition_position": int(source.transition_position),
                "A_RT": a_rt,
                "B_RT": b_rt,
                "AB_midpoint": midpoint,
                "AB_difference": difference,
                "probe_RT": probe_rt,
                "probe_midpoint_deviation": probe_rt - midpoint,
                "responsibility_denominator_usable": usable,
                "responsibility_min_abs_denominator": (
                    config.responsibility_min_abs_denominator
                ),
                "q_unclipped": (
                    (probe_rt - b_rt) / difference if usable else np.nan
                ),
            })
    midpoint = pd.DataFrame(rows)
    global_rows = []
    for (condition, seed, context), group in midpoint.groupby(
        ["condition", "seed", "probe_context"], sort=True
    ):
        usable = group[group.responsibility_denominator_usable]
        x = usable.AB_difference.to_numpy(dtype=float)
        y = (usable.probe_RT - usable.B_RT).to_numpy(dtype=float)
        denominator = float(x @ x)
        q = float(x @ y / denominator) if denominator > 1e-14 else np.nan
        global_rows.append({
            "condition": condition,
            "seed": int(seed),
            "probe_context": context,
            "q_global_unclipped": q,
            "usable_transition_count": len(usable),
            "global_fit_SSE": (
                float(np.sum((y - q * x) ** 2)) if np.isfinite(q) else np.nan
            ),
            "rank_selected_using_probe": False,
        })
    globals_frame = pd.DataFrame(global_rows)
    responsibility = midpoint.merge(
        globals_frame, on=["condition", "seed", "probe_context"], how="left"
    )
    responsibility["global_scalar_predicted_RT"] = (
        responsibility.B_RT
        + responsibility.q_global_unclipped * responsibility.AB_difference
    )
    responsibility["global_scalar_predicted_midpoint_deviation"] = (
        responsibility.global_scalar_predicted_RT - responsibility.AB_midpoint
    )
    responsibility["global_scalar_residual"] = (
        responsibility.probe_RT - responsibility.global_scalar_predicted_RT
    )

    endpoint_lookup = pivot.set_index(index)
    bridge_rows = []
    for row in execution.itertuples(index=False):
        endpoint = endpoint_lookup.loc[(row.condition, row.seed, row.transition_position)]
        a_rt, b_rt = float(endpoint.A), float(endpoint.B)
        difference = a_rt - b_rt
        rt = float(row.normalized_RT)
        midpoint_value = (a_rt + b_rt) / 2
        if row.probe_context == "A":
            q = 1.0
        elif row.probe_context == "B":
            q = 0.0
        elif abs(difference) >= config.responsibility_min_abs_denominator:
            q = (rt - b_rt) / difference
        else:
            q = np.nan
        bridge_rows.append({
            "source": "SRN",
            "dataset": "SRN",
            "condition": row.condition,
            "seed": int(row.seed),
            "context": row.probe_context,
            "transition_position": int(row.transition_position),
            "RT": rt,
            "A_RT": a_rt,
            "B_RT": b_rt,
            "AB_midpoint": midpoint_value,
            "AB_difference": difference,
            "q_unclipped": q,
            "probe_midpoint_deviation": rt - midpoint_value,
        })
    return midpoint, responsibility, pd.DataFrame(bridge_rows)
