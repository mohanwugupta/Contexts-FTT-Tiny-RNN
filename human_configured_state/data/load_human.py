"""Reuse the validated PRD3 exclusions, then recover seven SRN-aligned transitions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retrieval_modeling.data import load_retrieval_data


RAW_COLUMNS = (
    "ID", "new_trialNumber", "sequenceNumber", "keypressNumber", "rt",
    "correct", "trialType", "counterBalance",
)
CONTRACT_COLUMNS = (
    "participant_id", "dataset", "block_id", "original_sequence_in_block",
    "analysis_sequence_in_block", "retrieval_context", "raw_context",
    "counterbalance", "probe_phase", "probe_exposure_cohort",
    "probe_entry_context", "previous_context", "sequence_correct",
)


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes"})


def _probe_epochs(frame: pd.DataFrame, target: str) -> pd.Series:
    result = pd.Series(None, index=frame.index, dtype=object)
    prefix = "Mixed" if target == "Both" else "Novel"
    probe = frame.retrieval_context.eq(target)
    training = probe & frame.raw_context.astype(str).str.lower().str.contains("training")
    for _, indices in frame[training].groupby("participant_id", sort=False).groups.items():
        blocks = frame.loc[indices].sort_values("block_id", kind="stable").block_id.drop_duplicates()
        mapping = {block: f"{prefix}{number + 1}" for number, block in enumerate(blocks)}
        result.loc[indices] = frame.loc[indices, "block_id"].map(mapping)
    test = probe & frame.raw_context.astype(str).str.lower().str.contains("test")
    result.loc[test] = f"{prefix}Test"
    return result


def build_token_rows(raw: pd.DataFrame, sequence_contract: pd.DataFrame, *, dataset: str):
    """Join correct keypresses to already-validated sequence units.

    The retrieval contract has already removed raw sequence 0, incomplete
    participants/blocks, invalid sequences, and duplicate sessions. Response
    position 1 is excluded so positions 1..7 mean transitions 1->2 through
    7->8, exactly matching the frozen SRN's next-token states.
    """
    missing_raw = set(RAW_COLUMNS) - set(raw)
    missing_contract = set(CONTRACT_COLUMNS) - set(sequence_contract)
    if missing_raw or missing_contract:
        raise ValueError(
            f"Missing raw columns {sorted(missing_raw)} or contract columns {sorted(missing_contract)}."
        )
    contract = sequence_contract[list(CONTRACT_COLUMNS)].copy()
    contract["participant_id"] = contract.participant_id.astype(str)
    contract["block_id_key"] = contract.block_id.astype(str)
    contract["original_sequence_key"] = pd.to_numeric(
        contract.original_sequence_in_block, errors="raise"
    ).astype(int)
    contract = contract.drop_duplicates(
        ["participant_id", "block_id_key", "original_sequence_key"]
    )

    source = raw[list(RAW_COLUMNS)].copy()
    source["participant_id"] = source.ID.astype(str)
    source["block_id_key"] = source.new_trialNumber.astype(str)
    source["original_sequence_key"] = (
        pd.to_numeric(source.sequenceNumber, errors="coerce") + 1
    ).astype("Int64")
    source["keypress_number"] = pd.to_numeric(source.keypressNumber, errors="coerce")
    source["rt_numeric"] = pd.to_numeric(source.rt, errors="coerce")
    # The spatial Maze task stores correct target slots as 0,2,...,14,
    # whereas Melody/Mixed/Novel store ordinal positions 0,...,7. Infer the
    # declared block-label encoding from its complete observed range, then map
    # both tasks to the same ordinal response position.
    doubled_encoding = source.groupby("trialType").keypress_number.transform("max").gt(7)
    source["ordinal_keypress"] = np.where(
        doubled_encoding, source.keypress_number / 2.0, source.keypress_number
    )
    source = source[
        _truthy(source.correct) & source.rt_numeric.gt(0) &
        source.ordinal_keypress.between(1, 7, inclusive="both") &
        np.isclose(source.ordinal_keypress, np.round(source.ordinal_keypress))
    ].copy()
    joined = source.merge(
        contract,
        on=["participant_id", "block_id_key", "original_sequence_key"],
        how="inner", validate="many_to_one",
    )
    if joined.empty:
        raise ValueError("No correct SRN-aligned keypresses matched the preprocessing contract.")
    joined["response_token_position"] = joined.ordinal_keypress.astype(int) + 1
    joined["transition_position"] = joined.response_token_position - 1
    joined["transition_label"] = joined.transition_position.map(
        lambda value: f"position_{value}_to_{value + 1}"
    )
    group_columns = [
        "participant_id", "dataset", "block_id", "analysis_sequence_in_block",
        "original_sequence_in_block", "retrieval_context", "raw_context",
        "counterbalance", "probe_phase", "probe_exposure_cohort",
        "probe_entry_context", "previous_context", "response_token_position",
        "transition_position", "transition_label",
    ]
    result = joined.groupby(group_columns, dropna=False, sort=False).agg(
        mean_rt=("rt_numeric", "mean"),
        source_attempt_count=("rt_numeric", "size"),
    ).reset_index()
    expected = set(range(1, 8))
    unit_columns = ["participant_id", "block_id", "original_sequence_in_block"]
    valid_units = result.groupby(unit_columns).transition_position.agg(
        lambda values: set(values) == expected
    )
    valid_keys = set(valid_units[valid_units].index)
    keep = [tuple(row) in valid_keys for row in result[unit_columns].itertuples(index=False, name=None)]
    result = result[keep].copy()
    result["participant_uid"] = str(dataset) + ":" + result.participant_id.astype(str)
    result["context"] = result.retrieval_context
    result["sequence_repetition"] = result.analysis_sequence_in_block.astype(int)
    result["probe_split"] = np.where(
        result.sequence_repetition.mod(2).eq(1), "train", "test"
    )
    result["block_order"] = pd.to_numeric(result.block_id, errors="coerce")
    result["is_probe"] = result.context.isin(["Both", "Novel"])
    result["probe_epoch"] = _probe_epochs(result, "Both" if dataset == "Combined" else "Novel")
    result["probe_order"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for _, indices in result[result.is_probe].groupby("participant_uid", sort=False).groups.items():
        blocks = result.loc[indices].sort_values("block_order").block_id.drop_duplicates().tolist()
        mapping = {block: number + 1 for number, block in enumerate(blocks)}
        result.loc[indices, "probe_order"] = result.loc[indices, "block_id"].map(mapping).astype("Int64")
    result["log_rt"] = np.log(result.mean_rt)
    result["unit_id"] = (
        result.participant_uid + "|b" + result.block_id.astype(str) +
        "|s" + result.original_sequence_in_block.astype(str) +
        "|t" + result.transition_position.astype(str)
    )
    result["position_definition"] = "SRN-aligned transitions 1->2 through 7->8"
    if result.unit_id.duplicated().any():
        raise AssertionError("Token preprocessing produced duplicate behavioral units.")
    return result.sort_values(
        ["participant_uid", "block_order", "sequence_repetition", "transition_position"],
        kind="stable",
    ).reset_index(drop=True)


def load_human_token_data(path: str | Path, dataset: str, *, target_probe: str):
    contract = load_retrieval_data(path, dataset)
    participants = contract.loc[
        contract.retrieval_context.eq(target_probe), "participant_id"
    ].unique()
    contract = contract[contract.participant_id.isin(participants)].copy()
    raw = pd.read_csv(path, usecols=list(RAW_COLUMNS))
    result = build_token_rows(raw, contract, dataset=dataset)
    result.attrs["preprocessing_contract"] = "retrieval_v1"
    result.attrs["target_probe"] = target_probe
    return result
