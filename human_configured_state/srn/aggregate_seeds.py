"""Preregistered all-seed aggregation; best-seed selection is forbidden."""

from __future__ import annotations

import pandas as pd


def aggregate_seed_profiles(frame: pd.DataFrame, *, expected_seeds=None):
    seeds = set(int(value) for value in frame.seed.unique())
    if expected_seeds is not None and seeds != set(int(value) for value in expected_seeds):
        raise ValueError("SRN aggregation must contain every registered seed exactly as supplied.")
    value_columns = [
        column for column in (
            "execution_q", "full_component_q", "shared_component_q", "specific_component_q",
            "a_rt", "b_rt", "probe_rt", "endpoint_separation",
        ) if column in frame
    ]
    reliable = frame[frame.ratio_reliable.astype(bool)].copy()
    grouped = reliable.groupby(
        ["probe_context", "transition_position"], as_index=False
    )[value_columns].mean()
    counts = reliable.groupby(
        ["probe_context", "transition_position"], as_index=False
    ).seed.nunique().rename(columns={"seed": "reliable_seed_count"})
    result = grouped.merge(counts, on=["probe_context", "transition_position"])
    seed_text = "|".join(str(value) for value in sorted(seeds))
    result["registered_seed_count"] = len(seeds)
    result["seeds_used"] = seed_text
    result["aggregation_method"] = "all_registered_seed_mean"
    result["best_seed_selection"] = False
    return result

