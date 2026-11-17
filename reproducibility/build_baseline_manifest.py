#!/usr/bin/env python3
"""Freeze the pre-cleanup paper baseline without changing any analysis output."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility"

FINAL_OUTPUTS = {
    "srn_execution_v2_1": ROOT / "outputs/ftt_srn_execution_v2_final_checkpointed",
    "srn_causal_v2_2": ROOT / "outputs/ftt_srn_causal_v22_final",
    "srn_context_sequence_v2_3": ROOT / "outputs/ftt_srn_context_sequence_v23_final",
    "human_configured_state_v3": ROOT / "outputs/human_configured_state_v3",
    "human_low_rank_v3_1": ROOT / "outputs/human_low_rank_interpretation_v31",
}

FINAL_COMMANDS = {
    "srn_execution_v2_1": (
        "python scripts/run_ftt_srn_execution_v2.py --profile final "
        "--output outputs/ftt_srn_execution_v2_final_checkpointed"
    ),
    "srn_causal_v2_2": (
        "python scripts/run_ftt_srn_causal_v22.py "
        "--source outputs/ftt_srn_execution_v2_final_checkpointed "
        "--output outputs/ftt_srn_causal_v22_final --profile final"
    ),
    "srn_context_sequence_v2_3": (
        "python scripts/run_ftt_srn_context_sequence_v23.py "
        "--source outputs/ftt_srn_execution_v2_final_checkpointed "
        "--output outputs/ftt_srn_context_sequence_v23_final --profile final"
    ),
    "human_configured_state_v3": (
        "python scripts/run_human_configured_state_v3.py "
        "--combined data/mixedBreak_clean.csv "
        "--novel data/threeContextsBreak_clean.csv "
        "--srn-source outputs/ftt_srn_execution_v2_final_checkpointed "
        "--output outputs/human_configured_state_v3 --profile final"
    ),
    "human_low_rank_v3_1": (
        "python scripts/run_human_low_rank_interpretation_v31.py "
        "--v3-output outputs/human_configured_state_v3 "
        "--combined data/mixedBreak_clean.csv "
        "--novel data/threeContextsBreak_clean.csv "
        "--output outputs/human_low_rank_interpretation_v31 --profile final"
    ),
}

CONFIG_FILES = [
    "ftt_srn_execution_v2/config.py",
    "ftt_srn_causal_representation/config.py",
    "ftt_srn_context_sequence_interaction/config.py",
    "human_configured_state/config.py",
    "human_low_rank_interpretation/config.py",
    "outputs/ftt_srn_execution_v2_final_checkpointed/run_config_v2.json",
    "outputs/ftt_srn_causal_v22_final/causal_run_config.json",
    "outputs/ftt_srn_context_sequence_v23_final/interaction_run_config.json",
    "outputs/human_configured_state_v3/human_run_config.json",
    "outputs/human_low_rank_interpretation_v31/low_rank_run_config.json",
]

RAW_INPUTS = [
    "data/mixedBreak_clean.csv",
    "data/threeContextsBreak_clean.csv",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _csv_record(path: Path) -> dict:
    frame = pd.read_csv(path, low_memory=False)
    return {
        "path": _relative(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def _records(path: str, *, query: str | None = None, columns=None) -> dict:
    source = ROOT / path
    frame = pd.read_csv(source, low_memory=False)
    if query:
        frame = frame.query(query)
    if columns:
        frame = frame[list(columns)]
    return {
        "source": path,
        "source_sha256": _sha(source),
        "filter": query,
        "records": _clean(frame.to_dict(orient="records")),
    }


def _group_snapshot(path: str, *, query: str | None, groups, values) -> dict:
    source = ROOT / path
    frame = pd.read_csv(source, low_memory=False)
    if query:
        frame = frame.query(query)
    rows = []
    group_key = groups[0] if len(groups) == 1 else list(groups)
    for key, group in frame.groupby(group_key, dropna=False, sort=True):
        key = (key,) if len(groups) == 1 else tuple(key)
        row = {name: value for name, value in zip(groups, key)}
        row["rows"] = int(len(group))
        for value in values:
            numeric = pd.to_numeric(group[value], errors="coerce").dropna()
            row[f"{value}_mean"] = float(numeric.mean()) if len(numeric) else None
            row[f"{value}_sem"] = float(numeric.sem()) if len(numeric) > 1 else None
        rows.append(row)
    return {
        "source": path,
        "source_sha256": _sha(source),
        "filter": query,
        "group_by": list(groups),
        "records": _clean(rows),
    }


def _headline_snapshots() -> dict:
    return {
        "figure_1_conceptual_model": {
            "kind": "conceptual",
            "numeric_sources": [],
        },
        "figure_2_srn_splitting_and_execution": {
            "splitting": _group_snapshot(
                "outputs/ftt_srn_execution_v2_final_checkpointed/splitting_metrics.csv",
                query="checkpoint == 'human_matched'",
                groups=("model_condition",),
                values=("sequence_accuracy", "ab_hidden_separation",
                        "sequence_transition_cost", "cross_context_transfer_cost",
                        "context_ab_weight_norm"),
            ),
            "execution_grounding": _records(
                "outputs/ftt_srn_execution_v2_final_checkpointed/splitting_metrics.csv",
                query="model_condition == 'High' and checkpoint == 'human_matched' and seed == 0",
                columns=("stage2_confidence_speed_correlation_overall",
                         "stage2_confidence_speed_correlation_A",
                         "stage2_confidence_speed_correlation_B",
                         "stage2_confidence_speed_controlled_coefficient",
                         "stage2_passed"),
            ),
            "graded_probes": _group_snapshot(
                "outputs/ftt_srn_execution_v2_final_checkpointed/probe_interpolation.csv",
                query="model_condition == 'High' and checkpoint == 'human_matched'",
                groups=("probe_type",),
                values=("rt_a", "rt_b", "rt_probe", "midpoint_deviation",
                        "interpolation_index", "between_endpoints"),
            ),
        },
        "figure_3_causal_configured_state": {
            "matched_vs_mismatched": _group_snapshot(
                "outputs/ftt_srn_context_sequence_v23_final/token_structured_causal_effects.csv",
                query="checkpoint == 'human_matched'",
                groups=("probe_context", "token_position"),
                values=("full_patch_effect", "wrong_token_patch_effect",
                        "random_patch_effect", "correct_token_advantage",
                        "full_prediction_identity_preservation"),
            ),
            "mechanism_decision": _records(
                "outputs/ftt_srn_context_sequence_v23_final/interaction_mechanism_summary.csv",
                query="checkpoint == 'human_matched' and seed == 0",
                columns=("probe_context", "group_fraction_M1_full_natural_patching",
                         "group_fraction_M2_correct_token_specificity",
                         "group_fraction_M3_token_structured_correspondence",
                         "group_fraction_M4_full_beats_shared",
                         "group_fraction_M5_necessity_with_sequence_preservation",
                         "group_fraction_random_direction_specificity",
                         "group_M1_pass", "group_M2_pass", "group_M3_pass",
                         "group_M4_pass", "group_M5_pass", "group_M6_pass",
                         "group_random_specificity_pass", "main_result_code"),
            ),
            "strict_necessity_negative_control": _group_snapshot(
                "outputs/ftt_srn_causal_v22_final/ablation_seed_summary.csv",
                query="checkpoint == 'human_matched'",
                groups=("checkpoint",),
                values=("ablation_execution_collapse", "ablation_sequence_preservation",
                        "natural_next_token_ce", "ablated_next_token_ce"),
            ),
        },
        "figure_4_human_structured_retrieval": {
            "context_position_test": _records(
                "outputs/human_configured_state_v3/context_position_model.csv",
                query="row_type == 'joint_test'",
            ),
            "mixed_transition_profile": _group_snapshot(
                "outputs/human_configured_state_v3/mixed_midpoint_deviation.csv",
                query=None,
                groups=("transition_position",),
                values=("r_a", "r_b", "delta", "midpoint_deviation", "responsibility_q"),
            ),
            "novel_transition_profile": _group_snapshot(
                "outputs/human_configured_state_v3/novel_midpoint_deviation.csv",
                query=None,
                groups=("transition_position",),
                values=("r_a", "r_b", "delta", "midpoint_deviation", "responsibility_q"),
            ),
            "mixed_model_order": _records(
                "outputs/human_configured_state_v3/mixed_model_comparison.csv"
            ),
            "novel_model_order": _records(
                "outputs/human_configured_state_v3/novel_model_comparison.csv"
            ),
        },
        "figure_5_low_rank_human_and_srn_boundary": {
            "rank_selection": _records(
                "outputs/human_low_rank_interpretation_v31/dimension_selection.csv",
                query="row_type == 'summary'",
            ),
            "rank_2_loadings": _records(
                "outputs/human_low_rank_interpretation_v31/component_loadings.csv"
            ),
            "cross_dataset_replication": _records(
                "outputs/human_low_rank_interpretation_v31/cross_dataset_subspace_similarity.csv"
            ),
            "human_srn_negative_result": _records(
                "outputs/human_low_rank_interpretation_v31/human_srn_subspace_similarity.csv"
            ),
            "interpretation_boundary": _records(
                "outputs/human_low_rank_interpretation_v31/final_dimension_interpretation_summary.csv"
            ),
        },
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, path in FINAL_OUTPUTS.items():
        if not path.is_dir():
            raise FileNotFoundError(f"Missing frozen output for {name}: {path}")

    artifacts = {}
    checksum_paths = set()
    for analysis, directory in FINAL_OUTPUTS.items():
        csvs = [_csv_record(path) for path in sorted(directory.glob("*.csv"))]
        artifacts[analysis] = {
            "directory": _relative(directory),
            "csvs": csvs,
            "csv_count": len(csvs),
        }
        checksum_paths.update(directory.glob("*.csv"))
        checksum_paths.update(directory.glob("*.json"))
        checksum_paths.update(directory.glob("*.md"))
        checksum_paths.update(directory.glob("figures*/*.png"))
    checkpoint_dir = FINAL_OUTPUTS["srn_execution_v2_1"] / "model_checkpoints"
    checksum_paths.update(checkpoint_dir.glob("*.pt"))
    checksum_paths.update(ROOT / path for path in CONFIG_FILES)
    checksum_paths.update(ROOT / path for path in RAW_INPUTS)

    manifest = {
        "manifest_version": "1.0.0-pre-cleanup",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Frozen scientific baseline recorded before paper-repository cleanup.",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "test_baseline": {
            "command": (
                "env PYTHONPYCACHEPREFIX=/private/tmp/v31_verify_pycache "
                "MPLCONFIGDIR=/private/tmp/v31_verify_mpl "
                "XDG_CACHE_HOME=/private/tmp/v31_verify_cache "
                "python -m pytest -q -p no:cacheprovider"
            ),
            "passed": 166,
            "failed": 0,
            "warnings": 1,
            "warning_note": "Pre-existing pandas FutureWarning in tests/test_epoch_models.py.",
            "elapsed_seconds": 317.53,
        },
        "final_commands": FINAL_COMMANDS,
        "reproduction_checks": {
            "srn_execution_v2_1": {"csvs_compared": 19, "exact_sha256_matches": 19},
            "srn_causal_v2_2": {"csvs_compared": 16, "exact_sha256_matches": 16},
            "srn_context_sequence_v2_3": {"csvs_compared": 25, "exact_sha256_matches": 25},
            "human_configured_state_v3": {"csvs_compared": 24, "exact_sha256_matches": 24},
            "human_low_rank_v3_1": {
                "full_pipeline_rerun_completed_immediately_before_cleanup": True,
                "selected_rank": 2,
            },
        },
        "frozen_decisions": {
            "fixed_sequence": [0, 1, 2, 0, 3, 1, 2, 3],
            "human_transition_index": "SRN-aligned transitions 1->2 through 7->8",
            "first_sequence_excluded": True,
            "counterbalance_preserved": True,
            "srn_curriculum": "17 alternating A10/B10 blocks",
            "mixed_novel_training_updates": 0,
            "human_probe_rows_used_for_pure_ab_profiles": 0,
            "selected_human_rank": 2,
            "model_selection_changed": False,
            "frozen_srn_weights_changed": False,
        },
        "raw_inputs": {
            path: {"sha256": _sha(ROOT / path), "bytes": (ROOT / path).stat().st_size}
            for path in RAW_INPUTS
        },
        "configuration_files": {
            path: {"sha256": _sha(ROOT / path), "bytes": (ROOT / path).stat().st_size}
            for path in CONFIG_FILES
        },
        "frozen_outputs": artifacts,
        "headline_figure_numerical_snapshots": _headline_snapshots(),
    }
    manifest_path = OUTPUT / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(_clean(manifest), indent=2, sort_keys=True) + "\n")

    checksum_lines = [
        f"{_sha(path)}  {_relative(path)}"
        for path in sorted(checksum_paths, key=_relative)
    ]
    (OUTPUT / "baseline_checksums.sha256").write_text("\n".join(checksum_lines) + "\n")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    print(f"Recorded {len(checksum_lines)} checksums")


if __name__ == "__main__":
    main()
