"""Participant-level structural predictions of common responsibility."""

from __future__ import annotations

import numpy as np
import pandas as pd


def participant_ab_separation_test(data: pd.DataFrame, common_q: float) -> pd.DataFrame:
    probes = data[~data.retrieval_context.isin(["A", "B"])]
    rows = []
    for (dataset, participant, context), group in probes.groupby(
        ["dataset", "participant_id", "retrieval_context"], sort=False
    ):
        a = float(group.pred_A_log_rt.mean())
        b = float(group.pred_B_log_rt.mean())
        midpoint = 0.5 * (a + b)
        separation = a - b
        observed = float(group.log_rt.mean()) - midpoint
        predicted = (float(common_q) - 0.5) * separation
        rows.append(
            {
                "dataset": dataset, "participant_id": participant, "retrieval_context": context,
                "probe_exposure_cohort": (
                    group.probe_exposure_cohort.iloc[0]
                    if "probe_exposure_cohort" in group else None
                ),
                "is_probe_control": (
                    bool(group.is_probe_control.iloc[0]) if "is_probe_control" in group else None
                ),
                "A_log_rt": a, "B_log_rt": b, "AB_skill_separation": separation,
                "observed_probe_displacement": observed,
                "predicted_probe_displacement": predicted,
                "common_q_A": float(common_q), "residual": observed - predicted,
            }
        )
    return pd.DataFrame(rows)


def structural_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (dataset, context), group in rows.groupby(["dataset", "retrieval_context"], sort=False):
        if len(group) > 1 and np.std(group.AB_skill_separation) > 0:
            slope, intercept = np.polyfit(group.AB_skill_separation, group.observed_probe_displacement, 1)
            correlation = float(np.corrcoef(group.AB_skill_separation, group.observed_probe_displacement)[0, 1])
        else:
            slope = intercept = correlation = np.nan
        summaries.append(
            {
                "dataset": dataset, "retrieval_context": context,
                "n_participants": int(len(group)), "observed_slope": slope,
                "observed_intercept": intercept, "correlation": correlation,
                "predicted_slope": float(group.common_q_A.mean() - 0.5),
            }
        )
    return pd.DataFrame(summaries)
