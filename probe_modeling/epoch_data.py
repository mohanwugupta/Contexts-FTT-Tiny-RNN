"""PRD2 canonical nine-sequence preprocessing and epoch structure."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .data import DataContractError, EmptyDatasetError, normalize_probe_data


EPOCH_ORDER = [
    "A_or_B",
    "Mixed_1",
    "Mixed_2",
    "Post_Mixed_Return",
    "Mixed_Test",
]


def _original_sequence_numbers(data: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(data.sequence_in_block, errors="coerce")
    if raw.isna().any():
        raise DataContractError("PRD2 requires numeric sequence positions.")
    unique = set(raw.astype(int).unique())
    if 0 in unique:
        return raw.astype(int) + 1
    # The starter sequence-level export already removed raw sequence zero and
    # stores the remaining zero-based positions as 1..9.
    if unique == set(range(1, 10)):
        return raw.astype(int) + 1
    if unique.issubset(set(range(1, 11))):
        return raw.astype(int)
    raise DataContractError(f"Unexpected sequence positions: {sorted(unique)}")


def _tag_epochs(participant: pd.DataFrame) -> pd.DataFrame:
    result = participant.copy()
    result["epoch"] = "A_or_B"
    blocks = (
        result.groupby("block_id", sort=False, as_index=False)
        .agg(
            context=("context", "first"),
            probe_phase=("probe_phase", "first"),
            order=("global_sequence", "min"),
        )
        .sort_values("order", kind="stable")
    )
    training = blocks[(blocks.context == "Probe") & (blocks.probe_phase == "training")]
    tests = blocks[(blocks.context == "Probe") & (blocks.probe_phase == "test")]
    if len(training) != 2 or len(tests) < 1:
        raise DataContractError(
            f"Participant {participant.participant_id.iloc[0]!r} needs two Mixed training blocks and a Mixed test."
        )
    mixed_1, mixed_2 = training.block_id.iloc[:2]
    mixed_test = tests.block_id.iloc[-1]
    ordered_ids = blocks.block_id.tolist()
    mixed_2_position = ordered_ids.index(mixed_2)
    return_block: Optional[object] = None
    for block in ordered_ids[mixed_2_position + 1 :]:
        context = blocks.loc[blocks.block_id.eq(block), "context"].iloc[0]
        if context in {"A", "B"}:
            return_block = block
            break
    if return_block is None:
        raise DataContractError("No established-context return block follows Mixed 2.")
    mapping = {
        mixed_1: "Mixed_1",
        mixed_2: "Mixed_2",
        return_block: "Post_Mixed_Return",
        mixed_test: "Mixed_Test",
    }
    result["epoch"] = result.block_id.map(mapping).fillna("A_or_B")

    preceding = blocks.loc[blocks.order < blocks.loc[blocks.block_id.eq(mixed_1), "order"].iloc[0]]
    preceding = preceding[preceding.context.isin(["A", "B"])]
    if preceding.empty:
        raise DataContractError("Mixed 1 has no preceding established context.")
    history = preceding.context.iloc[-1]
    result["counterbalance_history"] = f"{history}_before_Mixed"
    result["counterbalance_sign"] = 1.0 if history == "A" else -1.0
    result["label_mirror"] = 1.0
    return result


def normalize_epoch_data(
    frame: pd.DataFrame,
    dataset: str = "Combined",
    *,
    cohort_only: bool = True,
) -> pd.DataFrame:
    """Normalize PRD2 data before any metric or model can see sequence 1."""
    base = normalize_probe_data(
        frame,
        dataset=dataset,
        cohort_only=cohort_only,
        complete_cases=False,
    )
    base["original_sequence_in_block"] = _original_sequence_numbers(base)
    base = base[base.original_sequence_in_block.ne(1)].copy()
    base["analysis_sequence_in_block"] = base.original_sequence_in_block - 1
    base["sequence_in_block"] = base.analysis_sequence_in_block
    if "sequence_correct" not in base:
        base["sequence_correct"] = True

    invalid_blocks = []
    valid_participants = set(base.participant_id.unique())
    expected_positions = set(range(2, 11))
    for (participant, block), group in base.groupby(["participant_id", "block_id"], sort=False):
        positions = set(group.original_sequence_in_block.astype(int))
        valid = len(group) == 9 and positions == expected_positions and bool(group.sequence_correct.all())
        if not valid:
            invalid_blocks.append(
                {
                    "participant_id": str(participant),
                    "block_id": block,
                    "n_sequences": int(len(group)),
                    "original_positions": sorted(positions),
                    "all_sequences_correct": bool(group.sequence_correct.all()),
                }
            )
            valid_participants.discard(participant)
    base = base[base.participant_id.isin(valid_participants)].copy()
    if base.empty:
        raise EmptyDatasetError("No participant has complete nine-sequence PRD2 blocks.")

    # Preserve the predetermined modal complete-schedule rule (17 blocks in the
    # current experiment), excluding concatenated/repeated sessions.
    block_counts = base.groupby("participant_id").block_id.nunique()
    modal_blocks = int(block_counts.mode().iloc[0])
    nonmodal = block_counts[block_counts.ne(modal_blocks)].index.astype(str).tolist()
    base = base[~base.participant_id.isin(nonmodal)].copy()
    if base.empty:
        raise EmptyDatasetError("No participant remains after the modal-schedule rule.")

    base = base.sort_values(
        ["participant_id", "block_id", "analysis_sequence_in_block"], kind="stable"
    ).reset_index(drop=True)
    base["global_sequence"] = base.groupby("participant_id", sort=False).cumcount() + 1
    tagged = [
        _tag_epochs(group)
        for _, group in base.groupby("participant_id", sort=False)
    ]
    result = pd.concat(tagged).sort_index().reset_index(drop=True)
    result["epoch"] = pd.Categorical(result.epoch, categories=EPOCH_ORDER, ordered=True)
    result["block_sequence_count"] = result.groupby(
        ["participant_id", "block_id"]
    ).analysis_sequence_in_block.transform("size")
    result.attrs["invalid_blocks"] = invalid_blocks
    result.attrs["excluded_nonmodal_participants"] = nonmodal
    result.attrs["preprocessing_version"] = "v2"
    return result


def mirror_ab_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Return an A↔B label-reversed copy for symmetry/generalization tests."""
    mirrored = data.copy()
    swap_values = {"A": "B", "B": "A"}
    for column in ["context", "previous_context", "next_context", "return_context"]:
        if column in mirrored:
            mirrored[column] = mirrored[column].map(lambda value: swap_values.get(value, value))
    pairs = [
        ("state_A", "state_B"),
        ("pred_A_log_rt", "pred_B_log_rt"),
        ("pred_A_rt", "pred_B_rt"),
    ]
    for left, right in pairs:
        if left in mirrored and right in mirrored:
            temporary = mirrored[left].copy()
            mirrored[left] = mirrored[right]
            mirrored[right] = temporary
    if "counterbalance_history" in mirrored:
        mirrored["counterbalance_history"] = mirrored.counterbalance_history.map(
            {"A_before_Mixed": "B_before_Mixed", "B_before_Mixed": "A_before_Mixed"}
        )
    if "counterbalance_sign" in mirrored:
        mirrored["counterbalance_sign"] = -mirrored.counterbalance_sign.astype(float)
    mirrored["label_mirror"] = -mirrored.get("label_mirror", 1.0).astype(float) if isinstance(
        mirrored.get("label_mirror", 1.0), pd.Series
    ) else -1.0
    return mirrored

