"""Restartable, read-only mechanism-first diagnostic pipeline."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .analysis import (
    additive_cost_rescore, bootstrap_population, canonicalize_whole_profiles,
    decompose_profiles, fit_probe_q_delta, score_prediction_rows,
)
from .config import DiagnosticConfig, load_config
from .data import (
    build_input_manifest, human_condition_profiles, load_default_control_tokens,
    load_human_tokens, load_srn_condition_profiles, verify_manifest_unchanged,
)
from .default_basis import compare_novel_default_models, estimate_independent_default_basis
from .history import (
    fit_identified_history_models, history_identifiability_audit,
    sequence_table_from_tokens,
)
from .plotting import (
    figure_delta_reuse, figure_history, figure_model_scores, figure_novel_default,
    figure_q_delta, figure_raw_centered,
)


VERSION = "mechanism-first-probe-diagnostics-1.0.0"


def _progress(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] PROBE-DIAGNOSTICS | {message}", flush=True)


def _json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _manifest_hash(manifest: pd.DataFrame) -> str:
    excluded = manifest.drop(columns=["config_hash", "input_manifest_hash"], errors="ignore")
    text = excluded.sort_values("artifact_key").to_json(orient="records", double_precision=15)
    return hashlib.sha256(text.encode()).hexdigest()


def _with_metadata(frame: pd.DataFrame, config_hash: str, manifest_hash: str) -> pd.DataFrame:
    out = frame.copy()
    if "unit_id" not in out:
        out.insert(0, "unit_id", [f"row:{i}" for i in range(len(out))])
    if "condition" not in out:
        if "probe_context" in out:
            out["condition"] = out.probe_context.replace({"Both": "Mixed", "Combined": "Mixed", "Third": "Novel"})
        else:
            out["condition"] = "all"
    out["config_hash"] = config_hash
    out["input_manifest_hash"] = manifest_hash
    return out


def _write_table(output: Path, name: str, frame: pd.DataFrame, config: DiagnosticConfig, manifest_hash: str):
    _with_metadata(frame, config.config_hash, manifest_hash).to_csv(output / name, index=False)


def _read_table(output: Path, name: str) -> pd.DataFrame:
    path = output / name
    if not path.exists():
        raise FileNotFoundError(f"Stage restart requires {path}")
    return pd.read_csv(path, low_memory=False)


def _mark(output: Path, stage: int, config: DiagnosticConfig, manifest_hash: str, status: str = "complete"):
    _json(output / f"stage_{stage}_status.json", {
        "analysis_version": VERSION, "stage": stage, "status": status,
        "config_hash": config.config_hash, "input_manifest_hash": manifest_hash,
        "completed_at": datetime.now().astimezone().isoformat(),
    })


def _stage0(config: DiagnosticConfig) -> tuple[pd.DataFrame, dict, str]:
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, output / "probe_diagnostics_config.json")
    manifest, schema = build_input_manifest(config)
    manifest_hash = _manifest_hash(manifest)
    manifest = _with_metadata(manifest, config.config_hash, manifest_hash)
    manifest.to_csv(output / "input_manifest.csv", index=False)
    _json(output / "schema_audit.json", {
        "analysis_version": VERSION, "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash, "artifacts": schema,
    })
    _json(output / "condition_aliases.json", {
        "analysis_version": VERSION, "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash,
        "canonical_internal_labels": {"Combined": "Mixed", "Third": "Novel"},
        "aliases": config.payload["condition_aliases"],
        "original_labels_retained": True,
    })
    _mark(output, 0, config, manifest_hash)
    return manifest, schema, manifest_hash


def _load_manifest(config: DiagnosticConfig) -> tuple[pd.DataFrame, str]:
    manifest = pd.read_csv(config.output_dir / "input_manifest.csv", low_memory=False)
    hashes = manifest.input_manifest_hash.dropna().unique()
    if len(hashes) != 1:
        raise ValueError("Saved input manifest does not contain one stable hash.")
    if set(manifest.config_hash.dropna().astype(str)) != {config.config_hash}:
        raise ValueError("Saved stage outputs were created with a different config.")
    return manifest, str(hashes[0])


def _decompose_by_scale(profiles: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        decompose_profiles(one)
        for _, one in profiles.groupby("analysis_scale", sort=False)
    ], ignore_index=True)


def _stage1(config: DiagnosticConfig, manifest_hash: str):
    output = config.output_dir
    _progress("Stage 1/5: building human and frozen SRN level/shape profiles")
    human_tokens = load_human_tokens(config)
    human_profiles = human_condition_profiles(human_tokens)
    # The repository's canonical log-RT endpoint estimator is the frozen,
    # probe-free shrinkage model.  Preserve raw-ms cell means for the primary
    # additive-cost analysis, but use those registered r_a/r_b values for the
    # fully labeled log-RT robustness analysis.
    frozen_ab = pd.read_csv(config.input_path("human_ab_profiles"), low_memory=False)
    if frozen_ab.probe_rows_used_for_profile.fillna(0).astype(int).sum() != 0:
        raise ValueError("Frozen log-RT endpoint profiles contain probe leakage.")
    frozen_long = frozen_ab.melt(
        id_vars=["participant_uid", "transition_position", "n_a", "n_b"],
        value_vars=["r_a", "r_b"], var_name="endpoint", value_name="frozen_M",
    )
    frozen_long["condition"] = frozen_long.endpoint.map({"r_a": "A", "r_b": "B"})
    frozen_long["frozen_n"] = np.where(frozen_long.condition.eq("A"), frozen_long.n_a, frozen_long.n_b)
    frozen_lookup = frozen_long.set_index(["participant_uid", "transition_position", "condition"])
    log_endpoint_mask = human_profiles.analysis_scale.eq("log_rt") & human_profiles.condition.isin(["A", "B"])
    endpoint_keys = pd.MultiIndex.from_frame(
        human_profiles.loc[log_endpoint_mask, ["participant_uid", "transition_position", "condition"]]
    )
    if not endpoint_keys.isin(frozen_lookup.index).all():
        raise ValueError("A canonical frozen log-RT endpoint is missing for an included participant.")
    human_profiles.loc[log_endpoint_mask, "M"] = frozen_lookup.loc[endpoint_keys, "frozen_M"].to_numpy()
    human_profiles.loc[log_endpoint_mask, "n_observations"] = frozen_lookup.loc[endpoint_keys, "frozen_n"].to_numpy()
    human_profiles.loc[log_endpoint_mask, "standard_error"] = np.nan
    human_profiles.loc[log_endpoint_mask, "cell_estimator"] = (
        "frozen probe-free canonical shrinkage endpoint profile"
    )
    human_decomposition = _decompose_by_scale(human_profiles)
    srn_profiles = load_srn_condition_profiles(config)
    srn_decomposition = decompose_profiles(srn_profiles)

    human_q = []
    for scale, scale_rows in human_decomposition.groupby("analysis_scale", sort=False):
        epsilon = float(config.payload["weak_contrast_epsilon"][scale])
        for dataset, probe in (("Combined", "Mixed"), ("Novel", "Novel")):
            fit = fit_probe_q_delta(
                scale_rows[scale_rows.dataset.eq(dataset)], probe=probe, weak_epsilon=epsilon,
            )
            fit["system"] = "Human"
            fit["dataset"] = dataset
            fit["analysis_scale"] = scale
            fit["rt_units"] = "ms" if scale == "mean_rt_ms" else "log_ms"
            fit["primary_analysis"] = scale == "mean_rt_ms"
            human_q.append(fit)
    human_q = pd.concat(human_q, ignore_index=True)

    srn_original = []
    for probe in ("Mixed", "Novel"):
        fit = fit_probe_q_delta(
            srn_decomposition, probe=probe,
            weak_epsilon=float(config.payload["weak_contrast_epsilon"]["normalized_srn_rt"]),
        )
        fit["system"] = "SRN"
        fit["dataset"] = "Frozen_High_scale_stage2_layout"
        fit["analysis_scale"] = "normalized_srn_rt"
        fit["rt_units"] = "frozen_normalized_RT"
        fit["primary_analysis"] = True
        srn_original.append(fit)
    srn_original = pd.concat(srn_original, ignore_index=True)

    canonical_ab, canonical_map = canonicalize_whole_profiles(srn_profiles)
    expected_map = pd.read_csv(config.input_path("srn_canonical_map"))
    derived = canonical_map.assign(seed=canonical_map.unit_id.str.replace("seed:", "", regex=False).astype(int))
    check = derived.merge(expected_map, on="seed", suffixes=("_derived", "_frozen"), validate="one_to_one")
    if not (
        check.fast_original_condition.eq(check.Fast_original_context)
        & check.slow_original_condition.eq(check.Slow_original_context)
        & check.whole_profile_swapped_derived.eq(check.whole_profile_swapped_frozen)
    ).all():
        raise ValueError("Derived whole-profile canonicalization does not match frozen A/B-only map.")
    probe_rows = srn_profiles[srn_profiles.condition.isin(["Mixed", "Novel"])].copy()
    canonical_profiles = pd.concat([canonical_ab, probe_rows], ignore_index=True, sort=False)
    canonical_decomp = decompose_profiles(canonical_profiles)
    srn_canonical = []
    for probe in ("Mixed", "Novel"):
        fit = fit_probe_q_delta(
            canonical_decomp, probe=probe, endpoint_a="Fast", endpoint_b="Slow",
            weak_epsilon=float(config.payload["weak_contrast_epsilon"]["normalized_srn_rt"]),
            orientation="fast_slow",
        )
        fit["system"] = "SRN"
        fit["dataset"] = "Frozen_High_scale_stage2_layout"
        fit["analysis_scale"] = "normalized_srn_rt"
        fit["rt_units"] = "frozen_normalized_RT"
        fit["primary_analysis"] = False
        srn_canonical.append(fit)
    srn_q = pd.concat([srn_original, *srn_canonical], ignore_index=True)

    metrics = ["q_unconstrained", "q_constrained", "delta", "profile_rmse",
               "profile_r_squared", "profile_correlation", "cosine_similarity",
               "fraction_raw_between_endpoints", "fraction_centered_between_endpoints"]
    population_inputs = pd.concat([human_q, srn_q], ignore_index=True)
    summary, draws = bootstrap_population(
        population_inputs,
        group_cols=["system", "condition", "analysis_scale", "endpoint_orientation"],
        metrics=metrics, draws=config.bootstrap_draws, seed=config.random_seed,
    )
    summary["unit_id"] = "population"
    draws["unit_id"] = "bootstrap"

    for name, frame in (
        ("human_condition_profiles.csv", human_profiles),
        ("srn_condition_profiles.csv", srn_profiles),
        ("human_level_shape_decomposition.csv", human_decomposition),
        ("srn_level_shape_decomposition.csv", srn_decomposition),
        ("human_probe_q_delta.csv", human_q),
        ("srn_probe_q_delta.csv", srn_q),
        ("probe_q_delta_population_summary.csv", summary),
        ("probe_q_delta_bootstrap_draws.csv", draws),
        ("srn_whole_profile_canonicalization_audit.csv", canonical_map),
    ):
        _write_table(output, name, frame, config, manifest_hash)

    figure1 = pd.concat([
        human_decomposition[human_decomposition.analysis_scale.eq("mean_rt_ms")],
        canonical_decomp,
    ], ignore_index=True, sort=False)
    figure23 = population_inputs[
        (
            population_inputs.system.eq("Human")
            & population_inputs.analysis_scale.eq("mean_rt_ms")
            & population_inputs.endpoint_orientation.eq("original_ab")
        ) | (
            population_inputs.system.eq("SRN")
            & population_inputs.endpoint_orientation.eq("fast_slow")
        )
    ].copy()
    _write_table(output, "figure_1_source.csv", figure1, config, manifest_hash)
    _write_table(output, "figure_2_source.csv", figure23, config, manifest_hash)
    _write_table(output, "figure_3_source.csv", figure23, config, manifest_hash)
    figure_raw_centered(figure1, output)
    figure_delta_reuse(figure23, output)
    figure_q_delta(figure23, output)
    _mark(output, 1, config, manifest_hash)


def _load_existing_predictions(config: DiagnosticConfig) -> pd.DataFrame:
    existing = pd.read_csv(config.input_path("existing_model_predictions"), low_memory=False)
    existing = existing[existing.model.isin(["M0_scalar_context", "M1_scalar_mixture",
                                             "M2_token_specific", "M3_low_rank"])].copy()
    if existing.heldout_participant_in_group_training.astype(bool).any():
        raise ValueError("Existing-model fold firewall failed: held-out participant in training.")
    if existing.probe_rows_used_for_ab_profiles.fillna(0).astype(int).sum() != 0:
        raise ValueError("Existing-model probe leakage detected in A/B profiles.")
    rank2 = pd.read_csv(config.input_path("rank2_predictions"), low_memory=False)
    if rank2.heldout_participant_in_basis.astype(bool).any() or rank2.probe_rows_used_for_basis.sum() != 0:
        raise ValueError("Rank-2 basis firewall failed.")
    frozen_errors = pd.read_csv(config.input_path("rank2_frozen_scores"), low_memory=False)
    join_keys = ["fold", "dataset", "probe_context", "model", "participant_uid", "unit_id"]
    rank2 = rank2.merge(
        frozen_errors[join_keys + ["heldout_log_likelihood"]], on=join_keys,
        how="left", validate="one_to_one",
    )
    common = sorted(set(existing.columns).union(rank2.columns))
    existing = existing.reindex(columns=common)
    rank2 = rank2.reindex(columns=common)
    return pd.concat([existing, rank2], ignore_index=True, sort=False)


def _stage2(config: DiagnosticConfig, manifest_hash: str):
    output = config.output_dir
    _progress("Stage 2/5: re-scoring frozen predictions on raw level and centered shape")
    predictions = _load_existing_predictions(config)
    predictions["profile_id"] = (
        predictions.participant_uid.astype(str) + "|" + predictions.probe_epoch.astype(str)
        + "|r" + predictions.sequence_repetition.astype(str)
    )
    raw, centered, scored = score_prediction_rows(
        predictions, actual_col="actual_log_rt", predicted_col="predicted_log_rt",
        profile_cols=["model", "dataset", "participant_uid", "probe_context", "probe_epoch",
                      "sequence_repetition"],
        group_cols=["model", "model_label", "dataset", "probe_context"],
        ll_col="heldout_log_likelihood",
    )
    for frame in (raw, centered):
        frame["unit_id"] = "heldout_population"
        frame["condition"] = frame.probe_context.replace({"Both": "Mixed"})
        frame["analysis_scale"] = "log_rt"
        frame["fold_firewall_pass"] = True
        frame["probe_rows_used_for_basis"] = 0

    # Existing categorical result is a frozen historical reference, not a rescored row.
    categorical = pd.read_csv(config.input_path("categorical_reference"))
    categorical = categorical[categorical.readout_family.eq("categorical")]
    if len(categorical):
        reference = pd.DataFrame([{
            "unit_id": "historical_reference", "condition": "all",
            "model": "categorical_similarity_existing_reference",
            "model_label": "Categorical A-or-B (existing reference only)",
            "dataset": "historical_retrieval_analysis", "probe_context": "all",
            "score_type": "raw_historical_reference", "n_rows": int(categorical.heldout_probe_n.iloc[0]),
            "heldout_rmse": float(categorical.heldout_probe_rmse.iloc[0]),
            "heldout_log_likelihood": float(categorical.heldout_probe_log_likelihood.iloc[0]),
            "analysis_scale": "historical_reference_scale",
            "fold_firewall_pass": np.nan, "probe_rows_used_for_basis": np.nan,
            "reference_only_not_refit": True,
        }])
        raw = pd.concat([raw, reference], ignore_index=True, sort=False)

    additive_input = predictions[predictions.model.isin([
        "M1_scalar_mixture", "M3_low_rank", "shared_ab_basis", "separate_dataset_ab_basis"
    ])].copy()
    additive, additive_rows = additive_cost_rescore(additive_input)
    additive["unit_id"] = "heldout_population"
    additive["condition"] = additive.probe_context.replace({"Both": "Mixed"})
    additive["analysis_scale"] = "log_rt"
    additive["transition_specific_intercept_used"] = False
    additive["probe_rows_used_for_structure_basis"] = 0

    _write_table(output, "existing_model_raw_scores.csv", raw, config, manifest_hash)
    _write_table(output, "existing_model_centered_scores.csv", centered, config, manifest_hash)
    _write_table(output, "additive_cost_model_comparison.csv", additive, config, manifest_hash)
    _write_table(output, "additive_cost_prediction_rows.csv", additive_rows, config, manifest_hash)
    figure4 = pd.concat([raw, centered], ignore_index=True, sort=False)
    _write_table(output, "figure_4_source.csv", figure4, config, manifest_hash)
    figure_model_scores(figure4[figure4.score_type.isin(["raw", "centered_shape"])], output)
    _mark(output, 2, config, manifest_hash)


def _empty_branch_table(branch: str, condition: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "unit_id": branch, "condition": condition, "status": "not_estimated",
        "identifiability_status": "stopped", "reason": reason,
    }])


def _stage3(config: DiagnosticConfig, manifest_hash: str):
    output = config.output_dir
    _progress("Stage 3/5: auditing history identifiability before any history fit")
    tokens = load_human_tokens(config)
    sequence = sequence_table_from_tokens(tokens)
    audit, confounds, status = history_identifiability_audit(
        sequence, thresholds=config.payload["history_thresholds"]
    )
    _write_table(output, "history_identifiability.csv", audit, config, manifest_hash)
    _write_table(output, "history_confound_matrix.csv", confounds, config, manifest_hash)
    status.update({"config_hash": config.config_hash, "input_manifest_hash": manifest_hash})
    _json(output / "history_branch_status.json", status)
    if status["status"] == "pass":
        human_profiles = _read_table(output, "human_condition_profiles.csv")
        comparison, parameters, prediction_rows = fit_identified_history_models(
            tokens, sequence, human_profiles, audit, seed=config.random_seed,
        )
        status["status"] = "complete_individual_predictors"
        status["reason"] = (
            "Every eligible history predictor was fit individually as H0/Hdelta/Hq/Hboth; "
            "no post-coefficient multivariable model was constructed."
        )
        status["negative_controls"] = [
            "within-participant permutation preserving probe epoch",
            "trial/order fields included in preregistered identifiability audit",
            "T11 synthetic delta-only and q-only recovery",
        ]
        _write_table(output, "history_prediction_rows.csv", prediction_rows, config, manifest_hash)
        _json(output / "history_branch_status.json", status)
    else:
        comparison = _empty_branch_table("history_models", "Mixed", status["reason"])
        comparison["models_considered"] = "H0|Hdelta|Hq|Hboth"
        comparison["psychological_null_inferred"] = False
        parameters = _empty_branch_table("history_parameters", "Mixed", status["reason"])
        parameters["parameter"] = "not_estimated"
    _write_table(output, "history_model_comparison.csv", comparison, config, manifest_hash)
    _write_table(output, "history_parameter_summary.csv", parameters, config, manifest_hash)
    _write_table(output, "figure_5_source.csv", parameters, config, manifest_hash)
    figure_history(parameters, output)
    _mark(output, 3, config, manifest_hash, status=status["status"])


def _stage4(config: DiagnosticConfig, manifest_hash: str):
    output = config.output_dir
    _progress("Stage 4/5: auditing independent default sources and testing Novel")
    default_tokens = load_default_control_tokens(config)
    basis, default_audit = estimate_independent_default_basis(default_tokens, dataset="Novel")
    human_tokens = load_human_tokens(config)
    overlap = set(default_tokens.participant_uid).intersection(human_tokens.participant_uid)
    if overlap:
        raise ValueError(f"Default-source firewall failed: {len(overlap)} target participants entered the basis.")
    human_profiles = _read_table(output, "human_condition_profiles.csv")
    rank2 = pd.read_csv(config.input_path("rank2_predictions"), low_memory=False)
    comparison, prediction_rows = compare_novel_default_models(
        human_tokens, human_profiles, basis, rank2_predictions=rank2,
    )
    scale_sd = prediction_rows.groupby("model").actual_centered.transform("std").replace(0, np.nan)
    prediction_rows["centered_standardization_sd"] = scale_sd
    prediction_rows["actual_centered_standardized"] = prediction_rows.actual_centered / scale_sd
    prediction_rows["predicted_centered_standardized"] = prediction_rows.predicted_centered / scale_sd
    sequence_only = pd.read_csv(config.input_path("srn_novel_probe"), low_memory=False)
    seq = sequence_only[sequence_only.condition.eq("High_sequence_only")]
    srn_has_rt = bool(seq.normalized_RT.notna().any())
    manifest = pd.DataFrame([
        {
            "unit_id": "human_no_context_control", "condition": "Default", "system": "Human",
            "candidate_priority": 1, "source_dataset": default_audit["source_dataset"],
            "rows_used": default_audit["rows_used"], "units_used": default_audit["participants_used"],
            "ab_probe_rows_used": 0, "novel_rows_used": 0,
            "normalization": default_audit["normalization"], "fold_assignment": default_audit["fold_assignment"],
            "reliability": default_audit["split_half_reliability"], "is_ab_midpoint": False,
            "target_participant_overlap": 0,
            "status": "accepted_independent", "reason": "disjoint no-context control participants",
        },
        {
            "unit_id": "srn_sequence_only_control", "condition": "Default", "system": "SRN",
            "candidate_priority": 1, "source_dataset": "frozen High_sequence_only",
            "rows_used": len(seq), "units_used": seq.seed.nunique(), "ab_probe_rows_used": 0,
            "novel_rows_used": 0, "normalization": "none", "fold_assignment": "frozen seeds",
            "reliability": np.nan, "is_ab_midpoint": False,
            "target_participant_overlap": np.nan,
            "status": "accepted_independent" if srn_has_rt else "rejected_missing_execution_rt",
            "reason": "available frozen RT" if srn_has_rt else "sequence-only control has no execution RT readout",
        },
        {
            "unit_id": "srn_training_invariant_component", "condition": "Default", "system": "SRN",
            "candidate_priority": 2, "source_dataset": "not_saved",
            "rows_used": 0, "units_used": 0, "ab_probe_rows_used": 0, "novel_rows_used": 0,
            "normalization": "not_available", "fold_assignment": "not_available",
            "reliability": np.nan, "is_ab_midpoint": False,
            "target_participant_overlap": np.nan,
            "status": "rejected_missing_artifact",
            "reason": "no frozen training-only invariant RT component was saved",
        },
    ])
    basis["system"] = "Human"
    basis["analysis_scale"] = "mean_rt_ms_centered"
    _write_table(output, "default_basis_manifest.csv", manifest, config, manifest_hash)
    _write_table(output, "independent_default_basis.csv", basis, config, manifest_hash)
    _write_table(output, "novel_default_model_comparison.csv", comparison, config, manifest_hash)
    _write_table(output, "novel_default_prediction_rows.csv", prediction_rows, config, manifest_hash)
    _write_table(output, "figure_6_source.csv", prediction_rows, config, manifest_hash)
    figure_novel_default(prediction_rows, output)
    _mark(output, 4, config, manifest_hash, status="human_complete_srn_stopped_no_default_rt")


def _geometry_summary(config: DiagnosticConfig, probe: str) -> float:
    key = "srn_mixed_geometry" if probe == "Mixed" else "srn_novel_geometry"
    frame = pd.read_csv(config.input_path(key), low_memory=False)
    selected = frame[
        frame.stage.eq("stage2_layout") & frame.cohort.eq("confirmatory")
        & frame.condition.eq("High_monotonic") & frame.probe_context.eq(probe)
    ]
    return float(selected.distance_to_AB_segment.mean()) if len(selected) else np.nan


def _stage5(config: DiagnosticConfig, manifest: pd.DataFrame, manifest_hash: str):
    output = config.output_dir
    _progress("Stage 5/5: adjudicating the joint human–SRN pattern")
    qpop = _read_table(output, "probe_q_delta_population_summary.csv")
    centered = _read_table(output, "existing_model_centered_scores.csv")
    default = _read_table(output, "novel_default_model_comparison.csv")
    history_status = json.loads((output / "history_branch_status.json").read_text())
    rows = []
    for system, condition, scale in (
        ("Human", "Mixed", "mean_rt_ms"), ("Human", "Novel", "mean_rt_ms"),
        ("SRN", "Mixed", "normalized_srn_rt"), ("SRN", "Novel", "normalized_srn_rt"),
    ):
        q = qpop[
            qpop.system.eq(system) & qpop.condition.eq(condition)
            & qpop.analysis_scale.eq(scale) & qpop.endpoint_orientation.eq("original_ab")
        ]
        qrow = q.iloc[0] if len(q) else pd.Series(dtype=float)
        if system == "Human":
            probe_label = "Both" if condition == "Mixed" else "Novel"
            model = centered[
                centered.probe_context.eq(probe_label)
                & centered.model.isin(["shared_ab_basis", "separate_dataset_ab_basis"])
            ].sort_values("centered_shape_variance_explained", ascending=False)
            lowrank = float(model.centered_shape_variance_explained.iloc[0]) if len(model) else np.nan
            default_evidence = "not_applicable"
            if condition == "Novel":
                ms = default[default.analysis_scale.eq("mean_rt_ms")].set_index("model")
                default_evidence = (
                    float(ms.loc["N3_default_plus_ab", "centered_shape_variance_explained_centered"]
                          - ms.loc["N2_ab_contrast", "centered_shape_variance_explained_centered"])
                    if {"N3_default_plus_ab", "N2_ab_contrast"}.issubset(ms.index) else np.nan
                )
            history_delta = history_status["status"] if condition == "Mixed" else "not_fit_before_default_gate"
            history_q = history_delta
            off_manifold = np.nan
            branch = "identified" if condition == "Mixed" else "human_default_identified"
        else:
            lowrank = np.nan
            default_evidence = "no_independent_default_rt" if condition == "Novel" else "not_applicable"
            history_delta = history_q = "not_applicable_frozen_one_shot_probe"
            off_manifold = _geometry_summary(config, condition)
            branch = "identified_level_shape" + (";default_stopped" if condition == "Novel" else "")
        rows.append({
            "unit_id": f"{system}:{condition}", "condition": condition, "system": system,
            "analysis_scale": scale,
            "global_cost_delta": qrow.get("delta_estimate", np.nan),
            "global_cost_ci_lower": qrow.get("delta_ci_lower", np.nan),
            "global_cost_ci_upper": qrow.get("delta_ci_upper", np.nan),
            "ab_centered_profile_reuse_r2": qrow.get("profile_r_squared_estimate", np.nan),
            "ab_centered_profile_reuse_ci_lower": qrow.get("profile_r_squared_ci_lower", np.nan),
            "ab_centered_profile_reuse_ci_upper": qrow.get("profile_r_squared_ci_upper", np.nan),
            "low_rank_centered_profile_reuse_r2": lowrank,
            "history_effect_on_delta": history_delta, "history_effect_on_q": history_q,
            "shared_default_basis_evidence_incremental_r2": default_evidence,
            "off_manifold_hidden_state_distance": off_manifold,
            "branch_identifiability_status": branch,
        })
    summary = pd.DataFrame(rows)

    human_shape = summary[summary.system.eq("Human")].low_rank_centered_profile_reuse_r2.max()
    srn_mixed = summary[(summary.system.eq("SRN")) & summary.condition.eq("Mixed")].iloc[0]
    if np.isfinite(human_shape) and human_shape <= 0:
        decision, gate = "E. revise the human target before further network work", "Result F"
    elif srn_mixed.global_cost_delta > 0 and srn_mixed.ab_centered_profile_reuse_r2 >= float(
        config.payload["decision_thresholds"]["good_centered_r2"]
    ):
        decision, gate = "A. separate global cost from the existing SRN readout", "Result A"
    else:
        decision, gate = "C. add an explicit compositional retrieval/gating mechanism", "Result B"

    unchanged, changed = verify_manifest_unchanged(config, manifest)
    if not unchanged:
        raise RuntimeError(f"Frozen source immutability failed for: {changed}")
    human_q = _read_table(output, "human_probe_q_delta.csv")
    srn_q = _read_table(output, "srn_probe_q_delta.csv")
    canonical = _read_table(output, "srn_whole_profile_canonicalization_audit.csv")
    gate_audit = pd.DataFrame([
        {"unit_id": "G0.1", "condition": "all", "gate": "transition_identity", "status": "pass",
         "evidence": "human and SRN use ordinal positions 1--7 (transitions 1->2 through 7->8)"},
        {"unit_id": "G0.2-human", "condition": "Mixed|Novel", "gate": "endpoint_availability", "status": "pass",
         "evidence": f"q/delta estimated for {human_q.unit_id.nunique()} human units with complete A and B endpoints"},
        {"unit_id": "G0.2-srn", "condition": "Mixed|Novel", "gate": "endpoint_availability", "status": "pass",
         "evidence": f"q/delta estimated for {srn_q.unit_id.nunique()} seeds with complete A and B endpoints"},
        {"unit_id": "G0.3", "condition": "Mixed|Novel", "gate": "probe_leakage", "status": "pass",
         "evidence": "all predictive A/B and rank-2 artifacts report zero evaluated-probe rows in basis estimation"},
        {"unit_id": "G0.4", "condition": "A|B", "gate": "whole_profile_canonicalization", "status": "pass",
         "evidence": f"{len(canonical)} seed maps use A/B only and match the frozen whole-profile map"},
        {"unit_id": "G0.5", "condition": "all", "gate": "reproducibility_and_immutability", "status": "pass",
         "evidence": "fixed config/manifest hashes; deterministic bootstrap/folds; all source hashes unchanged"},
        {"unit_id": "G3", "condition": "Mixed", "gate": "history_identifiability",
         "status": history_status["status"], "evidence": history_status["reason"]},
        {"unit_id": "G4-human", "condition": "Novel", "gate": "independent_default_source", "status": "pass",
         "evidence": "disjoint human no-context controls; zero Novel/A/B probe rows in basis"},
        {"unit_id": "G4-srn", "condition": "Novel", "gate": "independent_default_source",
         "status": "stopped_no_independent_execution_rt",
         "evidence": "sequence-only control has no RT; no frozen training-only invariant RT component"},
    ])
    _write_table(output, "analysis_gate_audit.csv", gate_audit, config, manifest_hash)
    branch_status = {
        "analysis_version": VERSION, "config_hash": config.config_hash,
        "input_manifest_hash": manifest_hash, "source_hashes_unchanged": unchanged,
        "human_level_shape": "complete", "srn_level_shape": "complete",
        "existing_model_rescore": "complete", "history": history_status["status"],
        "human_default": "complete", "srn_default": "stopped_no_independent_execution_rt",
        "new_srn_training": False, "probe_rerun": False, "next_model_decision": decision,
        "decision_gate": gate,
    }
    _write_table(output, "human_srn_diagnostic_summary.csv", summary, config, manifest_hash)
    _json(output / "analysis_branch_status.json", branch_status)
    _write_report(output, summary, centered, default, branch_status)
    _mark(output, 5, config, manifest_hash)


def _fmt(value) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4g}"


def _write_report(output: Path, summary: pd.DataFrame, centered: pd.DataFrame,
                  default: pd.DataFrame, status: dict):
    lines = [
        "# Mechanism-first probe diagnostics report", "", "## 1. Provenance and audit", "",
        "All inputs were SHA-256 recorded before analysis and verified unchanged afterward. "
        "Human rows use the canonical retrieval-v1 exclusions; SRN rows are frozen zero-step "
        "High-scale Stage 2 layout probes. No SRN was trained and no probe was rerun.", "",
        "The human independent Default branch was identifiable from disjoint no-context control "
        "participants. The SRN Default branch stopped because the frozen sequence-only control "
        "has no execution RT and no training-only invariant RT component was saved.", "",
        "## 2. Level-versus-shape decomposition", "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.system} {row.condition} ({row.analysis_scale}): δ = {_fmt(row.global_cost_delta)} "
            f"[95% bootstrap CI {_fmt(row.global_cost_ci_lower)}, {_fmt(row.global_cost_ci_upper)}]; "
            f"A/B centered-profile R² = {_fmt(row.ab_centered_profile_reuse_r2)}."
        )
    primary_centered = centered[
        centered.model.isin(["M0_scalar_context", "M1_scalar_mixture", "M2_token_specific",
                             "M3_low_rank", "shared_ab_basis", "separate_dataset_ab_basis"])
    ].sort_values(["condition", "centered_shape_variance_explained"], ascending=[True, False])
    winners = primary_centered.groupby("condition", as_index=False).first()
    lines += ["", "## 3. Existing retrieval models after centering", ""]
    for row in winners.itertuples(index=False):
        lines.append(
            f"- {row.condition}: best existing centered score was {row.model_label} "
            f"(held-out centered R² {_fmt(row.centered_shape_variance_explained)}, "
            f"RMSE {_fmt(row.heldout_rmse)} on log RT)."
        )
    lines += [
        "", "Raw and centered scores use exactly the original held-out rows. The additive-cost "
        "augmentation estimates one global δ on frozen calibration repetitions (2/6) and scores "
        "frozen evaluation repetitions (4/8); it cannot alter centered-shape error.", "",
        "## 4. History-dependent retrieval", "",
        f"History branch status: `{status['history']}`. A stopped identifiability audit is not a "
        "null psychological result. See `history_identifiability.csv` for the support and "
        "epoch/order confounds.", "", "## 5. Novel shared/default retrieval", "",
    ]
    for row in default.itertuples(index=False):
        lines.append(
            f"- {row.model} ({row.analysis_scale}): raw held-out RMSE {_fmt(row.heldout_rmse_raw)}; "
            f"centered-shape R² {_fmt(row.centered_shape_variance_explained_centered)} "
            f"({row.evaluation_status})."
        )
    lines += [
        "", "The human Default profile is participant-centered and averaged only across disjoint "
        "no-context controls. It is not `(A+B)/2`, and no Novel row sets its orientation or scale.",
        "", "## 6. Human–SRN comparison", "",
        "Human milliseconds and normalized SRN RT are reported separately. Cross-system claims "
        "concern centered-profile reuse, not numerical equality of δ. Frozen SRN off-manifold "
        "distance is included in `human_srn_diagnostic_summary.csv`.", "",
        "## 7. Next-model decision", "", f"**{status['next_model_decision']}**", "",
        f"This follows the preregistered **{status['decision_gate']}** gate. Branch failures and "
        "missing artifacts were not replaced by newly fit SRNs, new RT readouts, or A/B midpoints.", "",
    ]
    (output / "MECHANISM_FIRST_PROBE_DIAGNOSTICS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_probe_diagnostics(
    config_path: str | Path, *, from_stage: int = 0, through_stage: int = 5,
) -> Path:
    config = load_config(config_path)
    if not (0 <= from_stage <= through_stage <= 5):
        raise ValueError("Stages must satisfy 0 <= from_stage <= through_stage <= 5.")
    _progress(f"Starting {VERSION}; stages {from_stage}--{through_stage}")
    if from_stage == 0:
        manifest, _, manifest_hash = _stage0(config)
    else:
        manifest, manifest_hash = _load_manifest(config)
    stages = {1: _stage1, 2: _stage2, 3: _stage3, 4: _stage4}
    for stage in range(max(1, from_stage), min(4, through_stage) + 1):
        stages[stage](config, manifest_hash)
    if from_stage <= 5 <= through_stage:
        _stage5(config, manifest, manifest_hash)
    _progress(f"Finished requested stages; outputs: {config.output_dir}")
    return config.output_dir
