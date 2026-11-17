"""PRD3 adapter supporting Both, Novel, and No Context probes together."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from probe_modeling.data import DataContractError, EmptyDatasetError, infer_column, normalize_probe_data

from .context_features import canonical_retrieval_context


def _is_target_participant(frame: pd.DataFrame, pid_column: str, context_column: str, dataset: str) -> set:
    labels = frame[context_column].astype(str).str.lower()
    if dataset == "Combined":
        target = labels.str.contains(r"\bmixed\b|no[ _-]?context", regex=True)
    else:
        # The actual export calls the third context ``Space Training`` and its
        # probe ``test spaceA``.  ``spaceA`` has no word boundary after
        # "space", so do not require one there.
        target = labels.str.contains(r"\bnovel\b|\bthird\b|space|no[ _-]?context", regex=True)
    return set(frame.loc[target, pid_column].dropna())


def _phase_preserving_alias(label: object, dataset: str) -> str:
    context = canonical_retrieval_context(label)
    raw = str(label).lower()
    if context in {"A", "B"}:
        return str(label)
    phase = "Training" if "training" in raw else "Test" if "test" in raw else "Training"
    return f"Mixed {phase}" if dataset == "Combined" else f"Novel {phase}"


def _original_sequence(sequence: pd.Series) -> pd.Series:
    values = pd.to_numeric(sequence, errors="coerce")
    if values.isna().any():
        raise DataContractError("Retrieval analysis requires numeric sequence positions.")
    integers = values.astype(int)
    unique = set(integers.unique())
    if 0 in unique:
        return integers + 1
    if unique == set(range(1, 10)):
        return integers + 1
    if unique.issubset(set(range(1, 11))):
        return integers
    raise DataContractError(f"Unexpected sequence positions: {sorted(unique)}")


def normalize_retrieval_data(
    frame: pd.DataFrame,
    dataset: str,
    *,
    complete_cases: bool = True,
) -> pd.DataFrame:
    """Normalize a raw dataset while retaining multiple probe families.

    The older loader intentionally enforces one probe family per dataset. This
    adapter aliases labels only for aggregation, then restores the predeclared
    Both/Novel/Default identity from the original experimental block labels.
    """
    dataset = str(dataset).strip().title()
    if dataset not in {"Combined", "Novel"}:
        raise DataContractError("Retrieval datasets must be Combined or Novel.")
    if frame is None or frame.empty:
        raise EmptyDatasetError(f"The {dataset} retrieval dataset is empty.")
    pid_column = infer_column(frame, ["participant_id", "ID", "pid", "subject"])
    context_column = infer_column(frame, ["retrieval_context", "context", "trialType", "condition", "group"])
    block_column = infer_column(frame, ["block_id", "new_trialNumber", "trialNumber", "block"])
    if None in {pid_column, context_column, block_column}:
        raise DataContractError("Missing participant, context, or block column.")

    selected = _is_target_participant(frame, pid_column, context_column, dataset)
    if not selected:
        raise EmptyDatasetError(f"No {dataset} probe participants were found.")
    source = frame[frame[pid_column].isin(selected)].copy()
    source["_retrieval_context"] = source[context_column].map(
        lambda value: canonical_retrieval_context(value) if "warmup" not in str(value).lower() else None
    )
    source["_retrieval_original_label"] = source[context_column].astype(str)
    source[context_column] = source.apply(
        lambda row: _phase_preserving_alias(row["_retrieval_original_label"], dataset)
        if row["_retrieval_context"] is not None else row[context_column],
        axis=1,
    )

    block_context = (
        source[source._retrieval_context.notna()]
        .groupby([pid_column, block_column], sort=False)
        ._retrieval_context.first()
    )
    block_label = (
        source[source._retrieval_context.notna()]
        .groupby([pid_column, block_column], sort=False)
        ._retrieval_original_label.first()
    )
    normalized = normalize_probe_data(
        source.drop(columns=["_retrieval_context", "_retrieval_original_label"]),
        dataset=dataset,
        cohort_only=False,
        complete_cases=False,
    )
    keys = list(zip(normalized.participant_id, normalized.block_id))
    # Source participant IDs may not be strings; normalize mapping keys.
    context_lookup = {(str(pid), block): value for (pid, block), value in block_context.items()}
    label_lookup = {(str(pid), block): value for (pid, block), value in block_label.items()}
    normalized["retrieval_context"] = [context_lookup[key] for key in keys]
    normalized["raw_context"] = [label_lookup[key] for key in keys]
    normalized["context"] = normalized.retrieval_context.map(
        lambda value: value if value in {"A", "B"} else "Probe"
    )
    normalized["probe_type"] = normalized.retrieval_context.where(
        ~normalized.retrieval_context.isin(["A", "B"]), None
    )
    normalized["is_probe"] = normalized.retrieval_context.isin(["Both", "Novel", "Default"])

    # Reconstruct phase from the untouched export label.  The shared PRD2
    # loader calls a test-only participant's sole probe "training" by design;
    # that convention is unsuitable here because CB1/CB6 are explicit
    # test-only controls.
    raw_lower = normalized.raw_context.astype(str).str.lower()
    raw_probe_training = normalized.is_probe & raw_lower.str.contains("training")
    has_probe_training = raw_probe_training.groupby(normalized.participant_id).transform("any")
    normalized["is_primary_probe"] = normalized.is_probe & (raw_probe_training | ~has_probe_training)
    normalized["probe_phase"] = np.where(
        normalized.is_probe & raw_lower.str.contains("training"), "training",
        np.where(normalized.is_probe & raw_lower.str.contains("test"), "test", None),
    )

    normalized["original_sequence_in_block"] = _original_sequence(normalized.sequence_in_block)
    normalized = normalized[normalized.original_sequence_in_block.ne(1)].copy()
    normalized["analysis_sequence_in_block"] = normalized.original_sequence_in_block - 1
    normalized["sequence_in_block"] = normalized.analysis_sequence_in_block

    valid_participants = set(normalized.participant_id.unique())
    invalid_blocks = []
    for (participant, block), group in normalized.groupby(["participant_id", "block_id"], sort=False):
        positions = set(group.original_sequence_in_block.astype(int))
        valid = len(group) == 9 and positions == set(range(2, 11)) and bool(group.sequence_correct.all())
        if not valid:
            valid_participants.discard(participant)
            invalid_blocks.append(
                {
                    "participant_id": participant, "block_id": block,
                    "n_sequences": int(len(group)), "original_positions": sorted(positions),
                }
            )
    normalized = normalized[normalized.participant_id.isin(valid_participants)].copy()
    if normalized.empty:
        raise EmptyDatasetError("No participant has complete nine-sequence retrieval blocks.")
    excluded_nonmodal = []
    if complete_cases and normalized.participant_id.nunique() > 1:
        counts = normalized.groupby("participant_id").block_id.nunique()
        modal = int(counts.mode().iloc[0])
        excluded_nonmodal = counts[counts.ne(modal)].index.astype(str).tolist()
        normalized = normalized[~normalized.participant_id.isin(excluded_nonmodal)].copy()
    normalized = normalized.sort_values(
        ["participant_id", "block_id", "analysis_sequence_in_block"], kind="stable"
    ).reset_index(drop=True)
    normalized["global_sequence"] = normalized.groupby("participant_id", sort=False).cumcount() + 1
    normalized["dataset"] = dataset
    normalized["counterbalance_history"] = np.where(
        normalized.previous_context.eq("A"), "A_before_current_block",
        np.where(normalized.previous_context.eq("B"), "B_before_current_block", "none"),
    )
    participant_training_context = (
        normalized[normalized.raw_context.astype(str).str.lower().str.contains("training") & normalized.is_probe]
        .groupby("participant_id", sort=False).retrieval_context.first()
    )
    participant_test_context = (
        normalized[normalized.is_probe]
        .groupby("participant_id", sort=False).retrieval_context.first()
    )
    normalized["probe_training_context"] = normalized.participant_id.map(participant_training_context)
    test_context = normalized.participant_id.map(participant_test_context)
    normalized["probe_exposure_cohort"] = np.where(
        normalized.probe_training_context.notna(),
        normalized.probe_training_context.astype(str) + "_training",
        test_context.astype(str) + "_test_only_control",
    )
    control = normalized.probe_training_context.isna()
    normalized["is_probe_training_participant"] = ~control
    normalized["is_probe_control"] = control
    probe_entry_context = (
        normalized[normalized.is_primary_probe]
        .sort_values(["participant_id", "global_sequence"], kind="stable")
        .groupby("participant_id", sort=False).previous_context.first()
    )
    normalized["probe_entry_context"] = normalized.participant_id.map(probe_entry_context)
    normalized.attrs["invalid_blocks"] = invalid_blocks
    normalized.attrs["excluded_nonmodal_participants"] = excluded_nonmodal
    normalized.attrs["preprocessing_version"] = "retrieval_v1"
    return normalized


def load_retrieval_data(path: str | Path, dataset: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    return normalize_retrieval_data(pd.read_csv(source), dataset)
