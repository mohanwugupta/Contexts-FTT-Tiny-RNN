"""Sequence aggregation, matched-endpoint classification, and uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluation_schedule(*, n_blocks: int = 17, block_length: int = 10) -> tuple[int, ...]:
    """Boundaries after both contexts have been observed, including final."""
    if n_blocks < 2 or block_length <= 0:
        raise ValueError("At least two positive-length A/B blocks are required.")
    return tuple(range(2 * block_length, n_blocks * block_length + 1, block_length))


def aggregate_sequence_rt(
    transitions: pd.DataFrame, *, group_cols: list[str],
    transition_col: str = "transition_position", rt_col: str = "normalized_RT",
) -> pd.DataFrame:
    """Average transitions 1--7 within a sequence before any seed averaging."""
    required = set(group_cols) | {transition_col, rt_col}
    missing = sorted(required.difference(transitions))
    if missing:
        raise ValueError(f"Transition source is missing columns: {missing}")
    if transitions.duplicated(group_cols + [transition_col]).any():
        raise ValueError("Transition rows are duplicated within a sequence.")
    expected = tuple(range(1, 8))
    observed = transitions.groupby(group_cols, dropna=False)[transition_col].agg(
        lambda values: tuple(sorted(int(x) for x in values))
    )
    if not observed.map(lambda x: x == expected).all():
        raise ValueError("Every sequence must contain exactly transitions 1--7; initiation/wrap are excluded.")
    result = transitions.groupby(group_cols, as_index=False, dropna=False).agg(
        sequence_mean_RT=(rt_col, "mean"),
        scored_transition_count=(transition_col, "size"),
    )
    result["aggregation_rule"] = "mean(normalized_RT over transitions 1..7)"
    return result


def classify_probe_position(
    rt_a: float, rt_b: float, rt_probe: float, *, gap_tolerance: float,
) -> dict[str, float | str | bool]:
    """Classify one paired same-snapshot sequence mean using original A/B identities."""
    values = np.asarray([rt_a, rt_b, rt_probe], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Classification requires finite A, B, and probe means.")
    gap = abs(float(rt_b - rt_a))
    if gap <= gap_tolerance:
        classification = "endpoints_too_close"
        q = np.nan
    elif min(rt_a, rt_b) <= rt_probe <= max(rt_a, rt_b):
        classification = "between"
        q = float((rt_probe - rt_a) / (rt_b - rt_a))
    elif rt_probe > max(rt_a, rt_b):
        classification = "slower_than_both"
        q = float((rt_probe - rt_a) / (rt_b - rt_a))
    else:
        classification = "faster_than_both"
        q = float((rt_probe - rt_a) / (rt_b - rt_a))
    return {
        "RT_A": float(rt_a), "RT_B": float(rt_b), "RT_probe": float(rt_probe),
        "signed_AB_gap": float(rt_b - rt_a), "absolute_AB_gap": gap,
        "endpoint_gap_tolerance": float(gap_tolerance),
        "classification": classification,
        "between_same_snapshot_endpoints": classification == "between",
        "q_sequence_descriptive": q,
        "q_defined": bool(np.isfinite(q)),
    }


def classification_table(source: pd.DataFrame, *, gap_tolerance: float) -> pd.DataFrame:
    keys = ["condition", "seed", "completed_training_sequences", "incoming_context"]
    pivot = source.pivot(index=keys, columns="probe_context", values="sequence_mean_RT")
    if not {"A", "B", "Mixed", "Novel"}.issubset(pivot.columns):
        raise ValueError("Every same-snapshot event requires A/B/Mixed/Novel branches.")
    rows = []
    for key, values in pivot.iterrows():
        base = dict(zip(keys, key))
        checkpoint_hashes = source[
            source.condition.eq(base["condition"])
            & source.seed.eq(base["seed"])
            & source.completed_training_sequences.eq(base["completed_training_sequences"])
        ].snapshot_state_sha256.unique()
        if len(checkpoint_hashes) != 1:
            raise ValueError("A/B/probe branches do not share one checkpoint hash.")
        for probe in ("Mixed", "Novel"):
            rows.append({
                **base, "probe_context": probe, "checkpoint_identity": checkpoint_hashes[0],
                **classify_probe_position(values.A, values.B, values[probe],
                                          gap_tolerance=gap_tolerance),
            })
    return pd.DataFrame(rows)


def summarize_classifications(classifications: pd.DataFrame) -> pd.DataFrame:
    keys = ["condition", "probe_context", "completed_training_sequences", "incoming_context"]
    counts = classifications.groupby(keys + ["classification"], as_index=False).size()
    totals = classifications.groupby(keys, as_index=False).agg(
        seed_count=("seed", "nunique"), mean_RT_A=("RT_A", "mean"),
        mean_RT_B=("RT_B", "mean"), mean_RT_probe=("RT_probe", "mean"),
        mean_absolute_AB_gap=("absolute_AB_gap", "mean"),
    )
    counts = counts.merge(totals, on=keys, validate="many_to_one")
    counts["proportion"] = counts["size"] / counts["seed_count"]
    grand_rows = []
    for key, one in classifications.groupby(keys, sort=False):
        values = classify_probe_position(
            one.RT_A.mean(), one.RT_B.mean(), one.RT_probe.mean(),
            gap_tolerance=float(one.endpoint_gap_tolerance.iloc[0]),
        )
        grand_rows.append({**dict(zip(keys, key)), "aggregation_level": "grand_mean", **values})
    grand = pd.DataFrame(grand_rows)
    counts["aggregation_level"] = "within_seed_fraction"
    return counts, grand


def bootstrap_classification_proportions(
    classifications: pd.DataFrame, *, draws: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample seeds and preserve all times from each selected seed."""
    rng = np.random.default_rng(seed)
    categories = ["between", "slower_than_both", "faster_than_both", "endpoints_too_close"]
    draw_rows = []
    for (condition, probe), one in classifications.groupby(["condition", "probe_context"], sort=False):
        seeds = one.seed.drop_duplicates().to_numpy()
        indexed = {s: one[one.seed.eq(s)] for s in seeds}
        for draw in range(draws):
            sampled = rng.choice(seeds, len(seeds), replace=True)
            sample = pd.concat([indexed[s].assign(bootstrap_copy=i) for i, s in enumerate(sampled)],
                               ignore_index=True)
            for (time, incoming), event in sample.groupby(["completed_training_sequences", "incoming_context"]):
                for category in categories:
                    draw_rows.append({
                        "condition": condition, "probe_context": probe, "draw": draw,
                        "completed_training_sequences": int(time), "incoming_context": incoming,
                        "classification": category,
                        "proportion": float(event.classification.eq(category).mean()),
                    })
    draws_frame = pd.DataFrame(draw_rows)
    summary = draws_frame.groupby(
        ["condition", "probe_context", "completed_training_sequences", "incoming_context", "classification"],
        as_index=False,
    ).proportion.agg(
        estimate="mean", ci_lower=lambda x: float(np.quantile(x, .025)),
        ci_upper=lambda x: float(np.quantile(x, .975)),
    )
    return summary, draws_frame
