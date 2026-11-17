"""Read-only source loading, provenance, and canonical profile construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from human_configured_state.data.load_human import (
    RAW_COLUMNS, build_token_rows, load_human_token_data,
)
from retrieval_modeling.data import load_retrieval_data

from .analysis import canonical_condition


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_audit(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    condition_columns = [
        c for c in ("condition", "probe_context", "context", "retrieval_context",
                    "raw_context", "original_probe_context") if c in frame
    ]
    transition_columns = [c for c in ("transition_position", "keypressNumber") if c in frame]
    identity = "participant_uid" if "participant_uid" in frame else (
        "participant_id" if "participant_id" in frame else (
            "ID" if "ID" in frame else ("seed" if "seed" in frame else None)
        )
    )
    return {
        "row_count": int(len(frame)),
        "column_names": list(frame.columns),
        "column_types": {c: str(t) for c, t in frame.dtypes.items()},
        "condition_levels": {
            c: sorted(frame[c].dropna().astype(str).unique().tolist())
            for c in condition_columns
        },
        "transition_levels": {
            c: sorted(pd.to_numeric(frame[c], errors="coerce").dropna().unique().tolist())
            for c in transition_columns
        },
        "identity_column": identity,
        "identity_count": int(frame[identity].nunique()) if identity else None,
        "missingness": {c: int(frame[c].isna().sum()) for c in frame.columns},
    }


def build_input_manifest(config) -> tuple[pd.DataFrame, dict]:
    """Hash and schema-audit every declared frozen input."""
    registry = config.payload["input_registry"]
    rows, audits = [], {}
    for item in registry:
        key = item["key"]
        path = config.input_path(key)
        exists = path.exists() and path.is_file()
        audit = _csv_audit(path) if exists and path.suffix.lower() == ".csv" else {}
        if exists and path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            audit = {
                "row_count": 1, "column_names": sorted(payload),
                "column_types": {k: type(v).__name__ for k, v in payload.items()},
                "condition_levels": {}, "transition_levels": {},
                "identity_column": None, "identity_count": None, "missingness": {},
            }
        status = "available" if exists else ("missing_required" if item["required"] else "missing_optional")
        record = {
            "unit_id": key, "condition": "all", "artifact_key": key,
            "purpose": item["purpose"], "branch": item["branch"],
            "required": bool(item["required"]), "status": status,
            "path": str(path.relative_to(config.repo_root)) if path.is_relative_to(config.repo_root) else str(path),
            "file_size_bytes": int(path.stat().st_size) if exists else np.nan,
            "sha256": sha256_file(path) if exists else None,
            "row_count": audit.get("row_count", np.nan),
            "column_names_json": json.dumps(audit.get("column_names", [])),
            "column_types_json": json.dumps(audit.get("column_types", {}), sort_keys=True),
            "condition_levels_json": json.dumps(audit.get("condition_levels", {}), sort_keys=True),
            "transition_levels_json": json.dumps(audit.get("transition_levels", {}), sort_keys=True),
            "identity_column": audit.get("identity_column"),
            "identity_count": audit.get("identity_count", np.nan),
            "missingness_json": json.dumps(audit.get("missingness", {}), sort_keys=True),
        }
        rows.append(record)
        audits[key] = {"path": record["path"], "status": status, **audit}
    manifest = pd.DataFrame(rows)
    missing_required = manifest[manifest.status.eq("missing_required")]
    if len(missing_required):
        names = ", ".join(missing_required.artifact_key)
        raise FileNotFoundError(f"Required frozen inputs are missing: {names}")
    return manifest, audits


def verify_manifest_unchanged(config, manifest: pd.DataFrame) -> tuple[bool, list[str]]:
    changed = []
    for row in manifest.itertuples(index=False):
        if row.status != "available":
            continue
        path = config.input_path(row.artifact_key)
        if not path.exists() or sha256_file(path) != row.sha256:
            changed.append(str(row.artifact_key))
    return not changed, changed


def load_human_tokens(config) -> pd.DataFrame:
    frames = [
        load_human_token_data(config.input_path("human_combined_clean"), "Combined", target_probe="Both"),
        load_human_token_data(config.input_path("human_novel_clean"), "Novel", target_probe="Novel"),
    ]
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["original_condition"] = out.context.astype(str)
    out["condition"] = out.original_condition.map(
        lambda x: canonical_condition(x, config.payload["condition_aliases"])
    )
    return out


def human_condition_profiles(tokens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["participant_uid", "dataset", "original_condition", "condition", "transition_position"]
    for scale, column, units in (
        ("mean_rt_ms", "mean_rt", "ms"), ("log_rt", "log_rt", "log_ms")
    ):
        grouped = tokens.groupby(keys, as_index=False).agg(
            M=(column, "mean"), n_observations=(column, "size"),
            standard_error=(column, "sem"), source_attempt_count=("source_attempt_count", "sum"),
        )
        grouped["unit_id"] = grouped.participant_uid
        grouped["system"] = "Human"
        grouped["analysis_scale"] = scale
        grouped["rt_units"] = units
        grouped["cell_estimator"] = "canonical cleaned participant cell mean; no new exclusions"
        grouped["missing_cell"] = False
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def load_srn_condition_profiles(config) -> pd.DataFrame:
    mixed = pd.read_csv(config.input_path("srn_mixed_probe"), low_memory=False)
    mixed = mixed[
        mixed.stage.eq("stage2_layout") & mixed.cohort.eq("confirmatory")
        & mixed.condition.eq("High_monotonic")
        & mixed.overlap_count.fillna(0).eq(0)
        & mixed.probe_context.isin(["A", "B", "Mixed"])
    ].copy()
    novel = pd.read_csv(config.input_path("srn_novel_probe"), low_memory=False)
    novel = novel[
        novel.stage.eq("stage2_layout") & novel.cohort.eq("confirmatory")
        & novel.condition.eq("High_monotonic") & novel.overlap_count.eq(0)
        & novel.probe_context.eq("Novel")
    ].copy()
    source = pd.concat([mixed, novel], ignore_index=True, sort=False)
    source["original_condition"] = source.probe_context.astype(str)
    source["condition"] = source.original_condition.map(
        lambda x: canonical_condition(x, config.payload["condition_aliases"])
    )
    source["unit_id"] = source.seed.map(lambda x: f"seed:{int(x)}")
    source["system"] = "SRN"
    source["analysis_scale"] = "normalized_srn_rt"
    source["rt_units"] = "frozen_normalized_RT"
    source["M"] = pd.to_numeric(source.normalized_RT, errors="raise")
    source["n_observations"] = 1
    source["standard_error"] = np.nan
    source["source_attempt_count"] = 1
    source["cell_estimator"] = "frozen zero-step probe output"
    source["missing_cell"] = False
    keep = [
        "unit_id", "seed", "system", "original_condition", "condition",
        "transition_position", "M", "n_observations", "standard_error",
        "source_attempt_count", "analysis_scale", "rt_units", "cell_estimator",
        "missing_cell", "stage", "cohort",
    ]
    out = source[keep].drop_duplicates(["unit_id", "condition", "transition_position"])
    expected = 20 * 4 * 7
    if len(out) != expected:
        raise ValueError(f"Expected {expected} frozen High-scale rows; found {len(out)}.")
    if out.normalized_RT.isna().any() if "normalized_RT" in out else False:
        raise ValueError("Frozen SRN RT contains missing values.")
    return out.reset_index(drop=True)


def load_default_control_tokens(config) -> pd.DataFrame:
    """Use the validated full contract, retaining independent Default participants."""
    frames = []
    for key, dataset in (("human_combined_clean", "Combined"), ("human_novel_clean", "Novel")):
        path = config.input_path(key)
        contract = load_retrieval_data(path, dataset)
        default_ids = contract.loc[contract.retrieval_context.eq("Default"), "participant_id"].unique()
        contract = contract[contract.participant_id.isin(default_ids)].copy()
        raw = pd.read_csv(path, usecols=list(RAW_COLUMNS))
        tokens = build_token_rows(raw, contract, dataset=dataset)
        tokens = tokens[tokens.context.eq("Default")].copy()
        tokens["original_condition"] = tokens.context
        tokens["condition"] = "Default"
        frames.append(tokens)
    return pd.concat(frames, ignore_index=True, sort=False)
