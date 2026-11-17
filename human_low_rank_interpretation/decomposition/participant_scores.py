"""Participant folds and shrunken pure-profile component scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .estimate_basis import LowRankBasis, profile_matrix


def participant_folds(frame: pd.DataFrame, *, n_folds: int):
    participants = sorted(frame.participant_uid.astype(str).unique())
    folds = min(int(n_folds), len(participants))
    assignment = {participant: index % folds for index, participant in enumerate(participants)}
    for fold in range(folds):
        held = {participant for participant, value in assignment.items() if value == fold}
        training = frame[~frame.participant_uid.astype(str).isin(held)].copy()
        heldout = frame[frame.participant_uid.astype(str).isin(held)].copy()
        yield fold, training, heldout


def participant_component_scores(
    profiles: pd.DataFrame, basis: LowRankBasis, *, shrinkage_strength: float
) -> pd.DataFrame:
    matrix = profile_matrix(profiles)
    raw = basis.transform(matrix.to_numpy(dtype=float))
    reliability_n = profiles.groupby("participant_uid")[[
        column for column in ("n_a", "n_b") if column in profiles
    ]].mean().mean(axis=1) if {"n_a", "n_b"}.issubset(profiles) else pd.Series(
        1.0, index=matrix.index
    )
    weight = reliability_n.reindex(matrix.index).to_numpy(dtype=float)
    weight = weight / (weight + float(shrinkage_strength))
    rows = []
    dataset_map = profiles.groupby("participant_uid").dataset.first()
    for row_index, participant in enumerate(matrix.index):
        for component in range(basis.rank):
            rows.append({
                "participant_uid": participant,
                "dataset": dataset_map.loc[participant],
                "component": component + 1,
                "raw_component_score": float(raw[row_index, component]),
                "shrunken_component_score": float(weight[row_index] * raw[row_index, component]),
                "shrinkage_weight": float(weight[row_index]),
                "hierarchical_shrinkage": True,
                "basis_rank": basis.rank,
                "probe_rows_used_for_basis": basis.probe_rows_used,
            })
    return pd.DataFrame(rows)
