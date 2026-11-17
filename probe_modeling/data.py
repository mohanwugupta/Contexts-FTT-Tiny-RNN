"""Validated adapters from experiment CSVs to the shared probe schema."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


class DataContractError(ValueError):
    """Base class for an input that cannot satisfy the normalized schema."""


class EmptyDatasetError(DataContractError):
    """Raised when a supplied CSV contains no observations."""


class UnknownContextLabelError(DataContractError):
    """Raised rather than silently treating an unknown label as a probe."""


SCHEMA_COLUMNS = [
    "participant_id",
    "dataset",
    "block_id",
    "sequence_in_block",
    "global_sequence",
    "context",
    "probe_type",
    "is_probe",
    "is_primary_probe",
    "probe_block_number",
    "previous_context",
    "next_context",
    "mean_rt",
    "log_rt",
    "error",
    "break_duration",
    "counterbalance",
]


def infer_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Return the first case-insensitive candidate present in ``df``."""
    lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _dataset_name(dataset: str) -> str:
    value = str(dataset).strip().lower().replace("_", " ").replace("-", " ")
    if value in {"combined", "both", "mixed", "combined cue"}:
        return "Combined"
    if value in {"novel", "third", "third context", "novel context"}:
        return "Novel"
    if value in {"default", "no context", "nocontext"}:
        return "Default"
    raise DataContractError(f"Unknown dataset kind {dataset!r}; expected Combined or Novel.")


def _is_warmup(label: object) -> bool:
    return "warmup" in str(label).strip().lower()


def canonicalize_context(label: object) -> tuple[str, Optional[str]]:
    """Map an explicit experimental label to (context, probe_type).

    Matching is intentionally conservative.  New labels must be added here so
    an accidental spelling or new condition cannot silently become ambiguity.
    """
    raw = str(label).strip()
    norm = " ".join(raw.lower().replace("_", " ").replace("-", " ").split())
    if not norm or norm == "nan":
        raise UnknownContextLabelError(f"Missing context label: {label!r}")
    if _is_warmup(norm):
        raise UnknownContextLabelError(f"Warmup label reached context validation: {raw!r}")

    exact_a = {"a", "maze", "maze a", "maze training", "maze test"}
    exact_b = {
        "b",
        "melody",
        "melody a",
        "melody training",
        "melody test",
        "test melodya",
        "test melody a",
    }
    exact_both = {"probe", "both", "mixed", "mixed training", "mixed test", "combined"}
    exact_novel = {
        "novel",
        "novel training",
        "novel test",
        "third",
        "third training",
        "third test",
        "third context",
        "third context training",
        "third context test",
        "context c",
        "c",
    }
    exact_default = {"no context", "no context training", "no context test", "nocontext"}

    if norm in exact_a:
        return "A", None
    if norm in exact_b:
        return "B", None
    if norm in exact_both:
        return "Probe", "Both"
    if norm in exact_novel:
        return "Probe", "Novel"
    if norm in exact_default:
        return "Default", "Default"
    raise UnknownContextLabelError(
        f"Unknown context label {raw!r}. Add an explicit mapping before analysis."
    )


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes", "correct"})


def _select_experimental_cohort(
    frame: pd.DataFrame,
    pid_col: str,
    context_col: str,
    dataset: str,
) -> pd.DataFrame:
    """Keep the cohort that actually received primary probe training.

    The combined source file also contains A/B controls and a NoContext cohort.
    Selection is based on an explicit training label, never on counterbalance
    numbers whose meanings may differ between exports.
    """
    labels = frame[context_col].astype(str).str.lower()
    if dataset == "Combined":
        has_primary = labels.str.contains(r"\bmixed\b") & labels.str.contains("training")
    elif dataset == "Novel":
        has_primary = (
            labels.str.contains(r"\bnovel\b|\bthird\b", regex=True)
            & labels.str.contains("training")
        )
    else:
        has_primary = labels.str.contains("no context") & labels.str.contains("training")
    selected = frame.loc[has_primary, pid_col].dropna().unique()
    return frame[frame[pid_col].isin(selected)].copy() if len(selected) else frame.copy()


def _numeric_sort_key(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    codes, _ = pd.factorize(series, sort=False)
    return pd.Series(codes, index=series.index)


def _nearest_established_contexts(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest preceding/following A/B *block* to every row."""
    result = frame.copy()
    result["previous_context"] = None
    result["next_context"] = None
    for _, participant_index in result.groupby("participant_id", sort=False).groups.items():
        participant = result.loc[participant_index]
        blocks = (
            participant.groupby("block_id", sort=False, as_index=False)
            .agg(context=("context", "first"), order=("global_sequence", "min"))
            .sort_values("order", kind="stable")
        )
        block_ids = blocks.block_id.tolist()
        contexts = blocks.context.tolist()
        previous: dict[object, Optional[str]] = {}
        last: Optional[str] = None
        for block, context in zip(block_ids, contexts):
            previous[block] = last
            if context in {"A", "B"}:
                last = context
        following: dict[object, Optional[str]] = {}
        nxt: Optional[str] = None
        for block, context in reversed(list(zip(block_ids, contexts))):
            following[block] = nxt
            if context in {"A", "B"}:
                nxt = context
        result.loc[participant_index, "previous_context"] = participant.block_id.map(previous).values
        result.loc[participant_index, "next_context"] = participant.block_id.map(following).values
    return result


def normalize_probe_data(
    frame: pd.DataFrame,
    dataset: str,
    *,
    cohort_only: bool = True,
    complete_cases: bool = True,
) -> pd.DataFrame:
    """Normalize keypress- or sequence-level observations.

    Probe observations are retained, but no model fitting occurs here.  Source
    row order is used only as a stable tie breaker; participant timelines are
    ordered by explicit block and sequence columns.
    """
    if frame is None or frame.empty:
        raise EmptyDatasetError(f"The {dataset!r} dataset contains no observations.")
    dataset = _dataset_name(dataset)
    frame = frame.copy()
    frame["_source_order"] = np.arange(len(frame))

    pid_col = infer_column(frame, ["participant_id", "ID", "pid", "subject"])
    context_col = infer_column(frame, ["context", "trialType", "condition", "group"])
    block_col = infer_column(frame, ["block_id", "new_trialNumber", "trialNumber", "block"])
    sequence_col = infer_column(frame, ["sequence_in_block", "sequenceNumber", "sequence"])
    missing = [
        name
        for name, value in {
            "participant": pid_col,
            "context": context_col,
            "block": block_col,
            "sequence": sequence_col,
        }.items()
        if value is None
    ]
    if missing:
        raise DataContractError(f"Missing required columns: {', '.join(missing)}")

    frame = frame[~frame[context_col].map(_is_warmup)].copy()
    if frame.empty:
        raise EmptyDatasetError("Only warmup observations were found.")
    if cohort_only:
        frame = _select_experimental_cohort(frame, pid_col, context_col, dataset)
    if frame.empty:
        raise EmptyDatasetError(f"No participants matched the {dataset} probe cohort.")

    mapped = frame[context_col].map(canonicalize_context)
    frame["_context"] = mapped.str[0]
    frame["_probe_type"] = mapped.str[1]
    expected_probe = "Both" if dataset == "Combined" else ("Novel" if dataset == "Novel" else "Default")
    incompatible = frame[frame._probe_type.notna() & frame._probe_type.ne(expected_probe)]
    if not incompatible.empty:
        labels = sorted(incompatible[context_col].astype(str).unique())
        raise DataContractError(f"{dataset} data contain incompatible probe labels: {labels}")

    counter_col = infer_column(frame, ["counterbalance", "counterBalance"])
    break_col = infer_column(frame, ["break_duration", "breakDuration", "break_rt"])
    mean_col = infer_column(frame, ["mean_rt", "meanRT"])
    rt_col = infer_column(frame, ["rt", "reaction_time"])
    correct_col = infer_column(frame, ["correct", "accuracy"])
    if mean_col is None and rt_col is None:
        raise DataContractError("Expected a meanRT/mean_rt column or keypress-level rt column.")

    frame["_block_sort"] = _numeric_sort_key(frame[block_col])
    frame["_sequence_sort"] = _numeric_sort_key(frame[sequence_col])
    frame["_is_training_probe"] = (
        frame._probe_type.notna() & frame[context_col].astype(str).str.lower().str.contains("training")
    )
    group_cols = [pid_col, block_col, sequence_col, "_context", "_probe_type"]

    records = []
    for keys, group in frame.groupby(group_cols, dropna=False, sort=False):
        pid, block, sequence, context, probe_type = keys
        if mean_col is not None:
            values = pd.to_numeric(group[mean_col], errors="coerce")
            valid = values[np.isfinite(values) & values.gt(0)]
            mean_rt = float(valid.mean()) if len(valid) else np.nan
        else:
            values = pd.to_numeric(group[rt_col], errors="coerce")
            valid_mask = np.isfinite(values) & values.gt(0)
            if correct_col is not None:
                valid_mask &= _truthy(group[correct_col]).to_numpy()
            valid = values[valid_mask]
            mean_rt = float(valid.mean()) if len(valid) else np.nan
        if correct_col is not None:
            correct_values = _truthy(group[correct_col])
            accuracy = float(correct_values.mean())
            error = 1.0 - accuracy
            correct_keypress_count = int(correct_values.sum())
        else:
            error = np.nan
            correct_keypress_count = np.nan
        records.append(
            {
                "participant_id": str(pid),
                "dataset": dataset,
                "block_id": block,
                "sequence_in_block": sequence,
                "context": context,
                "probe_type": None if pd.isna(probe_type) else probe_type,
                "mean_rt": mean_rt,
                "error": error,
                "break_duration": (
                    float(pd.to_numeric(group[break_col], errors="coerce").mean())
                    if break_col is not None
                    else np.nan
                ),
                "counterbalance": (
                    str(group[counter_col].dropna().iloc[0])
                    if counter_col is not None and group[counter_col].notna().any()
                    else None
                ),
                "raw_context": str(group[context_col].iloc[0]),
                "_block_sort": float(group._block_sort.min()),
                "_sequence_sort": float(group._sequence_sort.min()),
                "_source_order": int(group._source_order.min()),
                "_is_training_probe": bool(group._is_training_probe.any()),
                "source_row_count": int(len(group)),
                "correct_keypress_count": correct_keypress_count,
                "sequence_correct": bool(correct_keypress_count >= 8) if np.isfinite(correct_keypress_count) else True,
            }
        )
    result = pd.DataFrame.from_records(records)
    result = result[np.isfinite(result.mean_rt) & result.mean_rt.gt(0)].copy()
    if result.empty:
        raise EmptyDatasetError("No finite, positive sequence RTs remained after aggregation.")
    excluded_participants: list[str] = []
    if complete_cases and result.participant_id.nunique() > 1:
        participant_counts = result.groupby("participant_id").size()
        modes = participant_counts.mode()
        if len(modes):
            modal_count = int(modes.iloc[0])
            excluded_participants = participant_counts[participant_counts.ne(modal_count)].index.astype(str).tolist()
            if excluded_participants:
                result = result[~result.participant_id.isin(excluded_participants)].copy()
    result = result.sort_values(
        ["participant_id", "_block_sort", "_sequence_sort", "_source_order"], kind="stable"
    ).reset_index(drop=True)
    result["global_sequence"] = result.groupby("participant_id", sort=False).cumcount() + 1
    result["log_rt"] = np.log(result.mean_rt)
    result["is_probe"] = result.probe_type.isin(["Both", "Novel"])

    # If explicit training probes exist, they define the primary two-block test.
    has_training = result.groupby("participant_id")._is_training_probe.transform("any")
    result["is_primary_probe"] = result.is_probe & (~has_training | result._is_training_probe)
    result["probe_block_number"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    for _, indices in result[result.is_probe].groupby("participant_id", sort=False).groups.items():
        blocks = result.loc[indices].sort_values("global_sequence").block_id.drop_duplicates().tolist()
        mapping = {block: number + 1 for number, block in enumerate(blocks)}
        result.loc[indices, "probe_block_number"] = result.loc[indices, "block_id"].map(mapping).astype("Int64")

    result = _nearest_established_contexts(result)
    result["probe_phase"] = np.where(
        result.is_primary_probe, "training", np.where(result.is_probe, "test", None)
    )
    result = result[SCHEMA_COLUMNS + [
        "raw_context", "probe_phase", "source_row_count",
        "correct_keypress_count", "sequence_correct",
    ]]
    result = result.reset_index(drop=True)
    result.attrs["excluded_nonmodal_participants"] = excluded_participants
    return result


def load_probe_data(
    path: str | Path,
    dataset: Optional[str] = None,
    *,
    cohort_only: bool = True,
    complete_cases: bool = True,
) -> pd.DataFrame:
    """Read and normalize a CSV, inferring its dataset kind from its name."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if dataset is None:
        name = path.name.lower()
        dataset = "Novel" if ("three" in name or "novel" in name or "third" in name) else "Combined"
    return normalize_probe_data(
        pd.read_csv(path),
        dataset=dataset,
        cohort_only=cohort_only,
        complete_cases=complete_cases,
    )
