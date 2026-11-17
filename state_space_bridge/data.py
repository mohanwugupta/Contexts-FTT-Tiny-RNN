"""Validated sequence-level human and corrected-SRN adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retrieval_modeling.data import load_retrieval_data

from .contracts import assert_corrected_srn_source, assert_rt_identity


def _counterbalance_from_first_context(frame: pd.DataFrame) -> pd.Series:
    first = (
        frame[frame.retrieval_context.isin(["A", "B"])]
        .sort_values(["participant_id", "global_sequence"], kind="stable")
        .groupby("participant_id", sort=False).retrieval_context.first()
    )
    return frame.participant_id.map(first).map({"A": "AFirst", "B": "BFirst"})


def load_human_sequence_data(path, dataset: str) -> pd.DataFrame:
    """Return the validated old-style sequence mean RT without changing its DV."""
    frame = load_retrieval_data(path, dataset)
    frame = frame.copy()
    frame["counterbalance_group"] = frame.counterbalance.astype(str)
    frame["counterbalance_order"] = _counterbalance_from_first_context(frame)
    frame["observed"] = frame.mean_rt.astype(float)
    frame["source_dv"] = "mean correct-keypress RT per sequence (ms)"
    first_probe = (
        frame[~frame.retrieval_context.isin(["A", "B"])]
        .groupby("participant_id", sort=False).global_sequence.min()
    )
    frame["first_probe_sequence"] = frame.participant_id.map(first_probe)
    frame["before_first_probe"] = (
        frame.global_sequence < frame.first_probe_sequence.fillna(np.inf)
    )
    frame["context"] = frame.retrieval_context
    return frame


def make_human_average_traces(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("pre_probe_AB", "all_AB_robustness"):
        pure = frame[frame.context.isin(["A", "B"])].copy()
        if scope == "pre_probe_AB":
            pure = pure[pure.before_first_probe]
        grouped = pure.groupby(
            ["dataset", "counterbalance_group", "counterbalance_order", "block_id",
             "analysis_sequence_in_block", "context"],
            as_index=False, dropna=False,
        ).agg(
            observed=("observed", "mean"),
            observed_sem=("observed", "sem"),
            participants=("participant_id", "nunique"),
        )
        for (_, cb), trace in grouped.groupby(["dataset", "counterbalance_group"], sort=False):
            trace = trace.sort_values(
                ["block_id", "analysis_sequence_in_block"], kind="stable"
            ).copy()
            if trace.context.nunique() < 2 or len(trace) < 4:
                continue
            trace["global_step"] = np.arange(1, len(trace) + 1)
            trace["fit_scope"] = scope
            trace["agent_type"] = "human"
            trace["condition"] = trace.dataset
            trace["seed_or_group"] = "AVERAGE"
            rows.append(trace)
    if not rows:
        raise ValueError("No human A/B traces satisfied the state-space contract.")
    return pd.concat(rows, ignore_index=True)


def load_corrected_srn_traces(source_dir) -> pd.DataFrame:
    root = Path(source_dir).expanduser().resolve()
    assert_corrected_srn_source(root)
    frame = pd.read_csv(root / "training_dynamics.csv")
    assert_rt_identity(frame)
    frame = frame[frame.model_condition.isin(["High", "Low"])].copy()
    frame["observed"] = frame.normalized_rt.astype(float)
    grouped = frame.groupby(
        ["model_condition", "block_index", "sequence_in_block", "context"], as_index=False
    ).agg(
        observed=("observed", "mean"), observed_sem=("observed", "sem"),
        participants=("seed", "nunique"), sequence_ce=("sequence_ce", "mean"),
        correct_token_confidence=("correct_token_confidence", "mean"),
        execution_speed=("execution_speed", "mean"),
    )
    grouped = grouped.sort_values(
        ["model_condition", "block_index", "sequence_in_block"], kind="stable"
    )
    grouped["global_step"] = grouped.groupby("model_condition").cumcount() + 1
    grouped["dataset"] = "FTT_SRN"
    grouped["condition"] = grouped.model_condition.map(
        {"High": "High_scale_2_execution", "Low": "Low_scale_execution"}
    )
    grouped["counterbalance_group"] = "AFirst"
    grouped["counterbalance_order"] = "AFirst"
    grouped["block_id"] = grouped.block_index
    grouped["analysis_sequence_in_block"] = grouped.sequence_in_block
    grouped["fit_scope"] = "all_AB_training"
    grouped["agent_type"] = "SRN"
    grouped["seed_or_group"] = "ALL_SEED_AVERAGE"
    return grouped

