"""End-to-end PRD v3.1 low-rank human-dimension interpretation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LowRankInterpretationConfig
from .data import (
    build_behavioral_transition_features,
    build_transition_feature_table,
    load_v3_inputs,
)
from .decomposition import (
    bootstrap_components,
    evaluate_dimensions,
    fit_pca_basis,
    participant_component_scores,
)
from .decomposition.estimate_basis import assert_pure_profile_contract, profile_matrix
from .interpretation.behavioral_alignment import learning_alignment, switch_cost_alignment
from .interpretation.difficulty import difficulty_residualized_results
from .interpretation.permutation_tests import component_feature_associations
from .probes import (
    compare_shared_separate_basis,
    participant_structural_predictions,
    summarize_probe_coefficients,
    temporal_probe_coefficients,
)
from .recovery import apply_semantic_recovery_gate, run_semantic_recovery
from .replication import compare_cross_dataset_bases
from .srn import compare_human_srn_dimensionality


REQUIRED_CSVS = (
    "dimension_selection.csv", "component_loadings.csv", "component_bootstrap.csv",
    "component_stability.csv", "participant_component_scores.csv",
    "transition_feature_table.csv", "component_feature_associations.csv",
    "component_feature_permutations.csv", "difficulty_residualized_results.csv",
    "mixed_component_coefficients.csv", "novel_component_coefficients.csv",
    "shared_vs_separate_basis.csv", "temporal_mixed_coefficients.csv",
    "participant_structural_predictions.csv", "switch_cost_component_alignment.csv",
    "learning_component_alignment.csv", "cross_dataset_component_alignment.csv",
    "cross_dataset_subspace_similarity.csv", "human_srn_dimensionality_comparison.csv",
    "human_srn_shared_specific_variance.csv", "human_srn_subspace_similarity.csv",
    "semantic_recovery_matrix.csv", "final_dimension_interpretation_summary.csv",
)


def _progress(enabled, message):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] HUMAN-LOW-RANK-V3.1 | {message}", flush=True)


def _component_loadings(basis, loading_summary, stability):
    stable = stability[stability.resample_type.eq("all")].set_index("component")
    rows = []
    for component in range(basis.rank):
        for position in range(7):
            rows.append({
                "component": component + 1, "transition_position": position + 1,
                "loading": float(basis.components[position, component]),
                "component_variance_explained": float(basis.variance_explained[component]),
                "cumulative_variance_explained": float(basis.cumulative_variance[component]),
                "reference_dataset": "Combined",
                "basis_training_participants": len(basis.training_participants),
                "basis_source_contexts": basis.source_contexts,
                "probe_rows_used_for_basis": basis.probe_rows_used,
                "median_matched_cosine": float(stable.loc[component + 1, "median_matched_cosine"]),
                "stable_component": bool(stable.loc[component + 1, "stable_component"]),
            })
    result = pd.DataFrame(rows).merge(
        loading_summary, on=["component", "transition_position"], how="left"
    )
    result["interpretation_label"] = result.component.map(lambda value: f"Component {value}")
    result["semantic_label_assigned"] = False
    return result


def _coefficient_output(participant, summary):
    participant = participant.copy(); summary = summary.copy()
    participant["row_type"] = "participant"
    summary["row_type"] = "group_summary"
    return pd.concat([participant, summary], ignore_index=True, sort=False)


def _semantic_noise(tokens):
    pure = tokens[tokens.context.isin(["A", "B"])].copy()
    halves = pure.groupby(
        ["participant_uid", "transition_position", "probe_split", "context"],
        as_index=False,
    ).log_rt.mean().pivot_table(
        index=["participant_uid", "transition_position"],
        columns=["probe_split", "context"], values="log_rt",
    )
    train_delta = halves[("train", "A")] - halves[("train", "B")]
    test_delta = halves[("test", "A")] - halves[("test", "B")]
    value = float((train_delta - test_delta).std(ddof=1) / np.sqrt(2))
    return max(value if np.isfinite(value) else 0, .05)


def _assign_labels(loadings, associations, difficulty, recovery_gate, cross_summary):
    reproducible = bool(
        int(cross_summary["rank"].iloc[0]) < 7
        and cross_summary.mean_matched_component_cosine.iloc[0] >= .70
    )
    maps = {
        "mean_pure_logRT": "transition difficulty",
        "A_minus_B": "context sensitivity",
        "switch_cost": "switch vulnerability",
        "post_switch_recovery": "switch recovery",
        "early_learning_slope": "early learning",
        "late_learning_slope": "late learning",
        "initial_log_rt": "initial execution state",
        "asymptotic_log_rt": "asymptotic execution state",
    }
    result = loadings.copy()
    for component in sorted(result.component.unique()):
        candidates = associations[
            associations.component.eq(component)
            & associations.semantic_association_supported.astype(bool)
            & associations.feature.isin(maps)
        ].sort_values(["permutation_p", "association_statistic"], ascending=[True, False])
        label = f"Component {component}"
        allowed = False
        if recovery_gate.semantic_labels_allowed and reproducible and not candidates.empty:
            feature = str(candidates.feature.iloc[0])
            if feature in {"A_minus_B", "switch_cost", "post_switch_recovery"}:
                target = {
                    "A_minus_B": "A_minus_B", "switch_cost": "switch_cost",
                    "post_switch_recovery": "switch_cost",
                }[feature]
                beyond = difficulty[
                    difficulty.component.eq(component)
                    & difficulty.row_type.eq("target_association")
                    & difficulty.target.eq(target)
                ].beyond_difficulty_supported.any()
                allowed = bool(beyond)
            else:
                allowed = True
            if allowed:
                label = maps[feature]
        result.loc[result.component.eq(component), "interpretation_label"] = label
        result.loc[result.component.eq(component), "semantic_label_assigned"] = allowed
    result["cross_dataset_replication_gate"] = reproducible
    result["semantic_recovery_gate"] = recovery_gate.semantic_labels_allowed
    return result


def _final_summary(
    *, selected_rank, loadings, associations, difficulty, shared_separate,
    cross_summary, srn_similarity, recovery_gate, participants,
):
    stable = bool(loadings.groupby("component").stable_component.first().any())
    labeled = bool(loadings.semantic_label_assigned.any())
    beyond = bool(difficulty.beyond_difficulty_supported.any())
    shared_index = shared_separate.set_index(["dataset", "model"])
    shared_wins_scalar = all(
        shared_index.loc[(dataset, "shared_ab_basis"), "heldout_log_likelihood"]
        > shared_index.loc[(dataset, "scalar_baseline"), "heldout_log_likelihood"]
        for dataset in ("Combined", "Novel")
    )
    separate_wins = sum(
        shared_index.loc[(dataset, "separate_dataset_ab_basis"),
                         "mean_fold_log_likelihood_gain_over_shared"]
        > shared_index.loc[(dataset, "separate_dataset_ab_basis"),
                           "paired_gain_over_shared_se"]
        for dataset in ("Combined", "Novel")
    ) >= 1
    reproducible = bool(
        int(selected_rank) < 7
        and cross_summary.mean_matched_component_cosine.iloc[0] >= .70
    )
    srn_match = bool(
        int(selected_rank) < 7
        and srn_similarity.mean_principal_angle_degrees.iloc[0] <= 30
        and srn_similarity.maximum_principal_angle_degrees.iloc[0] <= 45
        and srn_similarity.minimum_canonical_correlation.iloc[0] >= .50
        and srn_similarity.variance_partition_difference.iloc[0] <= .20
    )
    labels = set(loadings.loc[loadings.semantic_label_assigned, "interpretation_label"])
    if srn_match and stable:
        main_code = "G"
        text = "Human and SRN systems match at dimensionality/geometry level, not exact token loadings."
    elif labels & {"switch vulnerability", "switch recovery", "early learning", "late learning"}:
        main_code = "C"
        text = "Stable components align with switch or learning dynamics."
    elif "context sensitivity" in labels and beyond:
        main_code = "B"
        text = "Context sensitivity survives the generic-difficulty control."
    elif "transition difficulty" in labels:
        main_code = "A"
        text = "The interpretable component primarily reflects transition difficulty."
    else:
        main_code = "D"
        text = "The low-rank basis is stable but remains behaviorally unidentified."
    shared_code = "F" if separate_wins else ("E" if shared_wins_scalar else "Not supported")
    return pd.DataFrame([{
        "main_result_code": main_code, "main_result_text": text,
        "shared_probe_result_code": shared_code,
        "shared_probe_result_text": (
            "Mixed and Novel require different pure-A/B bases."
            if separate_wins else (
                "Mixed and Novel share a pure-A/B basis with different coefficients."
                if shared_wins_scalar else
                "The shared A/B basis does not outperform both scalar probe accounts."
            )
        ),
        "selected_rank": int(selected_rank),
        "stable_basis_success": bool(stable and int(selected_rank) < 7),
        "stable_component_count": int(loadings.groupby("component").stable_component.first().sum()),
        "semantic_labels_assigned": labeled,
        "semantic_recovery_gate_passed": recovery_gate.semantic_labels_allowed,
        "semantic_recovery_reason": recovery_gate.reason,
        "minimum_semantic_recovery": recovery_gate.minimum_diagonal_recovery,
        "random_axis_false_label_rate": recovery_gate.random_false_label_rate,
        "beyond_difficulty_success": beyond,
        "shared_basis_beats_scalar_both_probes": shared_wins_scalar,
        "cross_dataset_reproducibility_success": reproducible,
        "shared_subspace_variance_fraction": float(
            cross_summary.shared_subspace_variance_fraction.iloc[0]
        ),
        "human_srn_abstraction_match": srn_match,
        "human_srn_mean_principal_angle_degrees": float(
            srn_similarity.mean_principal_angle_degrees.iloc[0]
        ),
        "human_outcomes_used_to_modify_srn": 0,
        "mixed_participants": participants["Combined"],
        "novel_participants": participants["Novel"],
        "components_neutrally_named_unless_all_gates_pass": True,
    }])


def _write_report(path, summary, labels, feature_hash):
    row = summary.iloc[0]
    named = labels.groupby("component").interpretation_label.first().tolist()
    path.write_text(f"""# PRD v3.1 low-rank human-dimension interpretation

## Decision

**Result {row.main_result_code}.** {row.main_result_text}

- Selected rank: **{row.selected_rank}**
- Stable components: **{row.stable_component_count}**
- Component labels: **{', '.join(named)}**
- Semantic recovery gate: **{row.semantic_recovery_gate_passed}**
- Beyond-difficulty support: **{row.beyond_difficulty_success}**
- Shared-basis result: **{row.shared_probe_result_code}** — {row.shared_probe_result_text}
- Cross-dataset replication: **{row.cross_dataset_reproducibility_success}**
- Human–SRN abstraction match: **{row.human_srn_abstraction_match}**

The primary semantic basis was estimated from Combined pure A/B profiles only. Novel pure A/B
profiles were fit independently and aligned only after estimation. Mixed and Novel outcomes
estimated coefficients but never changed a basis. The frozen transition-feature table hash is
`{feature_hash}`. Exact transition-label permutation tests over every estimable position
(up to all 7! labelings) supply primary semantic inference.
Human outcomes did not change SRN seeds, weights, or token profiles.
""", encoding="utf-8")


def run_low_rank_interpretation_pipeline(
    v3_output_dir, combined_path, novel_path, output_dir,
    *, config: LowRankInterpretationConfig | None = None,
):
    config = config or LowRankInterpretationConfig()
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress, "Loading frozen v3 tables and validated token rows")
    inputs = load_v3_inputs(
        v3_output_dir, combined_path, novel_path,
        participant_limit_per_dataset=config.participant_limit_per_dataset,
    )
    profiles = inputs.pure_profiles.copy()
    assert_pure_profile_contract(profiles)
    probes = {"Combined": inputs.mixed, "Novel": inputs.novel}
    participants = profiles.groupby("dataset").participant_uid.nunique().astype(int).to_dict()

    _progress(config.show_progress, "Selecting dimensionality by held-out reconstruction and probe prediction")
    dimension_selection, selected_rank = evaluate_dimensions(
        profiles, probes, candidate_ranks=config.candidate_ranks,
        n_folds=config.cv_folds,
        parallel_replicates=config.parallel_analysis_replicates,
        random_seed=config.random_seed,
    )
    combined_profiles = profiles[profiles.dataset.eq("Combined")]
    primary_basis = fit_pca_basis(combined_profiles, rank=selected_rank)
    _progress(config.show_progress, "Aligning participant folds and bootstrap components")
    component_bootstrap, loading_summary, component_stability = bootstrap_components(
        combined_profiles, primary_basis, replicates=config.bootstrap_replicates,
        n_folds=config.cv_folds, random_seed=config.random_seed,
        stability_threshold=config.stability_cosine_threshold,
    )
    component_loadings = _component_loadings(
        primary_basis, loading_summary, component_stability
    )

    _progress(config.show_progress, "Freezing formal, learning, error, and switch features")
    behavioral_features, participant_learning, participant_switch = (
        build_behavioral_transition_features(
            inputs.tokens,
            raw_paths={"Combined": inputs.combined_raw_path, "Novel": inputs.novel_raw_path},
        )
    )
    transition_features = build_transition_feature_table(profiles, behavioral_features)
    associations, permutations = component_feature_associations(
        component_loadings, component_stability, transition_features, component_bootstrap
    )
    difficulty = difficulty_residualized_results(
        component_loadings, transition_features, inputs.mixed, inputs.novel,
        component_stability,
    )

    _progress(config.show_progress, "Projecting Mixed, Novel, temporal, participant, switch, and learning profiles")
    mixed_participant, mixed_summary = summarize_probe_coefficients(
        inputs.mixed, primary_basis, replicates=config.bootstrap_replicates,
        random_seed=config.random_seed + 10,
    )
    novel_participant, novel_summary = summarize_probe_coefficients(
        inputs.novel, primary_basis, replicates=config.bootstrap_replicates,
        random_seed=config.random_seed + 20,
    )
    mixed_coefficients = _coefficient_output(mixed_participant, mixed_summary)
    novel_coefficients = _coefficient_output(novel_participant, novel_summary)
    temporal = temporal_probe_coefficients(
        pd.concat([inputs.mixed, inputs.novel], ignore_index=True), primary_basis,
        replicates=config.bootstrap_replicates, random_seed=config.random_seed + 30,
    )
    shared_separate, shared_predictions = compare_shared_separate_basis(
        profiles, probes, rank=selected_rank, n_folds=config.cv_folds
    )
    scores = pd.concat([
        participant_component_scores(
            group, primary_basis, shrinkage_strength=config.score_shrinkage_strength
        ) for _, group in profiles.groupby("dataset", sort=False)
    ], ignore_index=True)
    structural = participant_structural_predictions(
        scores, pd.concat([mixed_participant, novel_participant], ignore_index=True),
        shrinkage_strength=config.score_shrinkage_strength,
    )
    switch_alignment = switch_cost_alignment(participant_switch, primary_basis)
    learning_component_alignment = learning_alignment(associations)

    _progress(config.show_progress, "Testing independent dataset replication and frozen SRN abstraction")
    _bases, cross_alignment, cross_summary = compare_cross_dataset_bases(
        profiles, rank=selected_rank
    )
    srn_dimensionality, srn_variance, srn_similarity = compare_human_srn_dimensionality(
        primary_basis, inputs.srn_mixed
    )

    _progress(config.show_progress, "Running observed-scale semantic recovery")
    recovery, recovery_detail = run_semantic_recovery(
        transition_features, participant_count=participants["Combined"],
        noise_sd=_semantic_noise(inputs.tokens),
        replicates=config.semantic_recovery_replicates,
        random_seed=config.random_seed,
    )
    recovery_gate = apply_semantic_recovery_gate(
        recovery, minimum_recovery=config.semantic_recovery_threshold,
        maximum_random_false_label=config.random_false_label_threshold,
    )
    component_loadings = _assign_labels(
        component_loadings, associations, difficulty, recovery_gate, cross_summary
    )
    summary = _final_summary(
        selected_rank=selected_rank, loadings=component_loadings,
        associations=associations, difficulty=difficulty,
        shared_separate=shared_separate, cross_summary=cross_summary,
        srn_similarity=srn_similarity, recovery_gate=recovery_gate,
        participants=participants,
    )

    tables = {
        "dimension_selection": dimension_selection,
        "component_loadings": component_loadings,
        "component_bootstrap": component_bootstrap,
        "component_stability": component_stability,
        "participant_component_scores": scores,
        "transition_feature_table": transition_features,
        "component_feature_associations": associations,
        "component_feature_permutations": permutations,
        "difficulty_residualized_results": difficulty,
        "mixed_component_coefficients": mixed_coefficients,
        "novel_component_coefficients": novel_coefficients,
        "shared_vs_separate_basis": shared_separate,
        "temporal_mixed_coefficients": temporal,
        "participant_structural_predictions": structural,
        "switch_cost_component_alignment": switch_alignment,
        "learning_component_alignment": learning_component_alignment,
        "cross_dataset_component_alignment": cross_alignment,
        "cross_dataset_subspace_similarity": cross_summary,
        "human_srn_dimensionality_comparison": srn_dimensionality,
        "human_srn_shared_specific_variance": srn_variance,
        "human_srn_subspace_similarity": srn_similarity,
        "semantic_recovery_matrix": recovery,
        "final_dimension_interpretation_summary": summary,
    }
    for name, frame in tables.items():
        if frame.empty:
            raise AssertionError(f"Required v3.1 output {name} is empty.")
        frame.to_csv(output / f"{name}.csv", index=False)
    recovery_detail.to_csv(output / "semantic_recovery_detail.csv", index=False)
    shared_predictions.to_csv(output / "shared_vs_separate_basis_predictions.csv", index=False)
    participant_learning.to_csv(output / "participant_learning_features.csv", index=False)
    participant_switch.to_csv(output / "participant_switch_features.csv", index=False)
    inputs.srn_audit.to_csv(output / "source_srn_frozen_audit.csv", index=False)

    record = {
        **asdict(config), "analysis_version": "3.1.0",
        "v3_output_dir": str(inputs.root),
        "combined_path": str(inputs.combined_raw_path),
        "novel_path": str(inputs.novel_raw_path),
        "participants": participants, "selected_rank": selected_rank,
        "primary_basis_dataset": "Combined",
        "primary_basis_source_contexts": ["A", "B"],
        "probe_rows_used_for_primary_basis": 0,
        "transition_feature_table_sha256": transition_features.feature_table_sha256.iloc[0],
        "transition_permutation_mode": "exact_factorial_over_estimable_positions",
        "maximum_transition_permutations_per_association": 5040,
        "srn_seed_count": int(inputs.srn_audit.seed.nunique()),
        "srn_weights_changed": False,
        "human_outcomes_used_to_select_srn": 0,
        "human_outcomes_used_to_modify_srn": 0,
        "final_inference_ready": bool(
            config.profile == "final" and config.participant_limit_per_dataset is None
        ),
        "required_csvs": list(REQUIRED_CSVS),
        "main_result_code": summary.main_result_code.iloc[0],
    }
    (output / "low_rank_run_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(
        output / "low_rank_interpretation_report.md", summary, component_loadings,
        transition_features.feature_table_sha256.iloc[0],
    )
    if config.make_plots:
        _progress(config.show_progress, "Rendering eleven registered figures")
        cache = Path(tempfile.gettempdir()) / "human-low-rank-v31-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        from .plotting import make_required_figures
        make_required_figures(output / "figures_human_low_rank_v31", tables=tables)
    _progress(config.show_progress, f"Complete: {output}")
    return tables
