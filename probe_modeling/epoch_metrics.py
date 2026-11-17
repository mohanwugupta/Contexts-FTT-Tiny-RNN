"""Stable epoch-level behavioral quantities for PRD2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import add_mixture_indices


def add_epoch_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Add primary midpoint deviation D and secondary mixture indices."""
    result = add_mixture_indices(data)
    result["counterfactual_midpoint_log_rt"] = 0.5 * (
        result.pred_A_log_rt + result.pred_B_log_rt
    )
    result["midpoint_deviation_D"] = result.log_rt - result.counterfactual_midpoint_log_rt
    result["counterbalance"] = result.counterbalance_history
    return result


def epoch_behavioral_summary(data: pd.DataFrame) -> pd.DataFrame:
    enriched = add_epoch_metrics(data)
    group_columns = [
        "dataset", "epoch", "counterbalance_history", "analysis_sequence_in_block",
    ]
    rows = []
    for keys, group in enriched.groupby(group_columns, observed=True, dropna=False, sort=False):
        values = dict(zip(group_columns, keys))
        values.update(
            {
                "n_participants": int(group.participant_id.nunique()),
                "n_observations": int(len(group)),
                "mean_log_rt": float(group.log_rt.mean()),
                "sem_log_rt": float(group.log_rt.sem()),
                "mean_rt": float(group.mean_rt.mean()),
                "sem_rt": float(group.mean_rt.sem()),
                "mean_midpoint_deviation_D": float(group.midpoint_deviation_D.mean()),
                "sem_midpoint_deviation_D": float(group.midpoint_deviation_D.sem()),
                "mean_mixture_index": float(group.mixture_index.mean()),
                "sem_mixture_index": float(group.mixture_index.sem()),
            }
        )
        rows.append(values)
    return pd.DataFrame(rows)


def epoch_metric_rows(data: pd.DataFrame, epoch: str) -> pd.DataFrame:
    return add_epoch_metrics(data)[data.epoch.astype("object").eq(epoch)].reset_index(drop=True)


def post_mixed_consequences(data: pd.DataFrame) -> pd.DataFrame:
    """Return all nine observations in the one immediate post-Mixed A/B block."""
    rows = data[data.epoch.astype("object").eq("Post_Mixed_Return")].copy()
    rows["return_context"] = rows.context
    rows["counterfactual_log_rt"] = np.where(
        rows.context.eq("A"), rows.pred_A_log_rt, rows.pred_B_log_rt
    )
    rows["counterfactual_rt"] = np.exp(rows.counterfactual_log_rt)
    rows["aftereffect_log_rt"] = rows.log_rt - rows.counterfactual_log_rt
    rows["aftereffect_rt"] = rows.mean_rt - rows.counterfactual_rt
    rows["measurement_qualification"] = "First A/B block after 18 Mixed sequences"
    return rows.reset_index(drop=True)
