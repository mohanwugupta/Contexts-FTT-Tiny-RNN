"""Frozen transition definitions and pure-behavior semantic features."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ftt_srn_execution_v2.config import FIXED_SEQUENCE


FEATURE_SPEC_VERSION = "human-low-rank-v3.1-transition-features-1.0"


def _last_distance(sequence, index, key):
    prior = [position for position in range(index) if sequence[position] == key]
    return float(index - prior[-1]) if prior else np.nan


def formal_transition_features() -> pd.DataFrame:
    sequence = tuple(int(value) for value in FIXED_SEQUENCE)
    rows = []
    for transition in range(1, 8):
        current_index = transition - 1
        next_index = transition
        current = sequence[current_index]
        nxt = sequence[next_index]
        rows.append({
            "transition_position": transition,
            "current_key": current,
            "next_key": nxt,
            "distance_since_current_key": _last_distance(sequence, current_index, current),
            "distance_since_next_key": _last_distance(sequence, next_index, nxt),
            "cyclic_boundary": False,
            "first_half_vs_second_half": "first" if transition <= 4 else "second",
            "second_half_indicator": int(transition >= 5),
            "formal_feature_spec_version": FEATURE_SPEC_VERSION,
        })
    return pd.DataFrame(rows)


def _slope(frame: pd.DataFrame) -> float:
    if len(frame) < 2 or frame.exposure_index.nunique() < 2:
        return np.nan
    x = frame.exposure_index.to_numpy(dtype=float)
    x = (x - x.mean()) / max(x.std(ddof=0), 1e-12)
    return float(np.polyfit(x, frame.log_rt.to_numpy(dtype=float), 1)[0])


def _participant_learning_features(pure: pd.DataFrame) -> pd.DataFrame:
    ordered = pure.sort_values(
        ["participant_uid", "context", "transition_position", "block_order",
         "sequence_repetition"], kind="stable"
    ).copy()
    ordered["exposure_index"] = ordered.groupby(
        ["participant_uid", "context", "transition_position"], sort=False
    ).cumcount()
    rows = []
    keys = ["participant_uid", "context", "transition_position"]
    for values, group in ordered.groupby(keys, sort=False):
        midpoint = (group.exposure_index.min() + group.exposure_index.max()) / 2
        early = group[group.exposure_index <= midpoint]
        late = group[group.exposure_index > midpoint]
        width = max(int(np.ceil(len(group) * .20)), 1)
        rows.append({
            **dict(zip(keys, values)),
            "early_learning_slope": _slope(early),
            "late_learning_slope": _slope(late),
            "initial_log_rt": float(group.head(width).log_rt.mean()),
            "asymptotic_log_rt": float(group.tail(width).log_rt.mean()),
            "learning_observations": int(len(group)),
        })
    return pd.DataFrame(rows)


def _participant_switch_features(pure: pd.DataFrame) -> pd.DataFrame:
    switched = pure[
        pure.previous_context.isin(["A", "B"])
        & pure.context.isin(["A", "B"])
        & pure.previous_context.ne(pure.context)
    ].copy()
    cells = switched.groupby(
        ["participant_uid", "dataset", "block_id", "transition_position",
         "sequence_repetition"], as_index=False
    ).log_rt.mean()
    rows = []
    keys = ["participant_uid", "dataset", "block_id", "transition_position"]
    for values, group in cells.groupby(keys, sort=False):
        profile = group.set_index("sequence_repetition").log_rt
        first = profile.get(1, np.nan)
        early_reference = profile.reindex([2, 3]).mean()
        late_reference = profile.reindex([7, 8, 9]).mean()
        rows.append({
            **dict(zip(keys, values)),
            "switch_cost": float(first - early_reference),
            "post_switch_recovery": float(first - late_reference),
        })
    result = pd.DataFrame(rows)
    return result.groupby(
        ["participant_uid", "dataset", "transition_position"], as_index=False
    ).agg(
        switch_cost=("switch_cost", "mean"),
        post_switch_recovery=("post_switch_recovery", "mean"),
        switch_blocks=("block_id", "nunique"),
    )


def _raw_error_rates(raw_paths, tokens: pd.DataFrame) -> pd.DataFrame:
    valid = tokens[[
        "participant_id", "dataset", "block_id", "original_sequence_in_block",
        "transition_position",
    ]].drop_duplicates().copy()
    valid["participant_id"] = valid.participant_id.astype(str)
    valid["block_key"] = valid.block_id.astype(str)
    frames = []
    for dataset, path in raw_paths.items():
        raw = pd.read_csv(path, usecols=[
            "ID", "new_trialNumber", "sequenceNumber", "keypressNumber", "trialType", "correct"
        ])
        raw["participant_id"] = raw.ID.astype(str)
        raw["block_key"] = raw.new_trialNumber.astype(str)
        raw["original_sequence_in_block"] = (
            pd.to_numeric(raw.sequenceNumber, errors="coerce") + 1
        ).astype("Int64")
        raw["keypress_numeric"] = pd.to_numeric(raw.keypressNumber, errors="coerce")
        doubled = raw.groupby("trialType").keypress_numeric.transform("max").gt(7)
        raw["ordinal_keypress"] = np.where(
            doubled, raw.keypress_numeric / 2.0, raw.keypress_numeric
        )
        raw = raw[
            raw.ordinal_keypress.between(1, 7, inclusive="both")
            & np.isclose(raw.ordinal_keypress, np.round(raw.ordinal_keypress))
        ].copy()
        raw["transition_position"] = raw.ordinal_keypress.astype(int)
        raw["dataset"] = dataset
        truth = raw.correct if pd.api.types.is_bool_dtype(raw.correct) else (
            raw.correct.astype(str).str.lower().isin({"true", "1", "yes", "correct"})
        )
        raw["is_error"] = ~truth.astype(bool)
        keep = valid[valid.dataset.eq(dataset)].copy()
        joined = raw.merge(
            keep[["participant_id", "block_key", "original_sequence_in_block",
                  "transition_position"]],
            on=["participant_id", "block_key", "original_sequence_in_block",
                "transition_position"],
            how="inner",
        )
        frames.append(joined)
    attempts = pd.concat(frames, ignore_index=True)
    return attempts.groupby("transition_position", as_index=False).agg(
        error_rate=("is_error", "mean"),
        error_attempts=("is_error", "size"),
    )


def build_behavioral_transition_features(
    tokens: pd.DataFrame, *, raw_paths: dict[str, object] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pure = tokens[tokens.context.isin(["A", "B"])].copy()
    learning = _participant_learning_features(pure)
    learning_summary = learning.groupby("transition_position", as_index=False).agg(
        early_learning_slope=("early_learning_slope", "mean"),
        late_learning_slope=("late_learning_slope", "mean"),
        initial_log_rt=("initial_log_rt", "mean"),
        asymptotic_log_rt=("asymptotic_log_rt", "mean"),
        learning_participants=("participant_uid", "nunique"),
    )
    switch = _participant_switch_features(pure)
    switch_summary = switch.groupby("transition_position", as_index=False).agg(
        switch_cost=("switch_cost", "mean"),
        post_switch_recovery=("post_switch_recovery", "mean"),
        switch_participants=("participant_uid", "nunique"),
        switch_blocks=("switch_blocks", "sum"),
    )
    summary = learning_summary.merge(switch_summary, on="transition_position", how="outer")
    if raw_paths:
        summary = summary.merge(
            _raw_error_rates(raw_paths, tokens), on="transition_position", how="left"
        )
    else:
        summary["error_rate"] = np.nan
        summary["error_attempts"] = 0
    return summary, learning, switch


def _table_hash(frame: pd.DataFrame) -> str:
    serial = frame.sort_values("transition_position").to_csv(
        index=False, float_format="%.12g", na_rep="NA"
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def build_transition_feature_table(
    pure_profiles: pd.DataFrame, behavioral_features: pd.DataFrame
) -> pd.DataFrame:
    pure = pure_profiles.groupby("transition_position", as_index=False).agg(
        mean_pure_logRT=("midpoint", "mean"),
        A_minus_B=("delta", "mean"),
        pure_participants=("transition_position", "size"),
    )
    result = formal_transition_features().merge(pure, on="transition_position", how="left")
    aggregate = behavioral_features.groupby("transition_position", as_index=False).mean(
        numeric_only=True
    )
    result = result.merge(aggregate, on="transition_position", how="left")
    required = [
        "error_rate", "early_learning_slope", "late_learning_slope", "switch_cost",
        "post_switch_recovery", "initial_log_rt", "asymptotic_log_rt",
    ]
    for column in required:
        if column not in result:
            result[column] = np.nan
    result["feature_table_frozen"] = True
    result["feature_table_spec_version"] = FEATURE_SPEC_VERSION
    result["feature_table_sha256"] = _table_hash(result)
    return result
