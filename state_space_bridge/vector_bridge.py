"""Contracts for the frozen human vector/low-rank extension."""

from __future__ import annotations

import pandas as pd


def validate_transition_positions(frame: pd.DataFrame):
    positions = set(pd.to_numeric(frame.transition_position, errors="raise").astype(int))
    if positions != set(range(1, 8)):
        raise ValueError(f"Expected seven ordinal transition positions; found {sorted(positions)}.")


def selected_rank(dimension_selection: pd.DataFrame) -> int:
    selected = dimension_selection[
        dimension_selection.selected_primary_rank.fillna(False).astype(bool)
    ]
    ranks = selected["rank"].dropna().astype(int).unique()
    if len(ranks) != 1:
        raise ValueError("Frozen dimension-selection table must identify exactly one rank.")
    return int(ranks[0])


def assert_rank_regression(dimension_selection: pd.DataFrame):
    rank = selected_rank(dimension_selection)
    if rank != 2:
        raise AssertionError(f"Frozen data-driven rank regression changed: expected 2, found {rank}.")

