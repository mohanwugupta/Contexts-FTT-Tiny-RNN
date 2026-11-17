"""End-to-end PRD v3 human behavioral configured-state analysis."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    counterbalance_sensitivity,
    evaluate_shared_vs_separate,
    fit_context_position_models,
    human_srn_alignment,
    mixed_novel_profile_comparison,
    temporal_probe_profiles,
)
from .config import HumanConfiguredStateConfig
from .data import load_human_token_data
from .models import fit_probe_model_ladder, summarize_model_comparison
from .pure_profiles import compute_midpoint_deviation, estimate_ab_profiles, profile_reliability
from .recovery import run_recovery
from .srn import load_frozen_srn_profiles


REQUIRED_CSVS = (
    "pure_ab_token_profiles.csv", "pure_ab_profile_reliability.csv",
    "context_position_model.csv", "mixed_midpoint_deviation.csv",
    "novel_midpoint_deviation.csv", "model_M0_scalar_context.csv",
    "model_M1_scalar_mixture.csv", "model_M2_token_specific.csv",
    "model_M3_low_rank.csv", "model_M4_srn_informed.csv",
    "novel_model_comparison.csv", "mixed_model_comparison.csv",
    "srn_mixed_profile.csv", "srn_novel_profile.csv", "human_srn_alignment.csv",
    "mixed_vs_novel_profile_comparison.csv", "shared_vs_separate_probe_profile.csv",
    "counterbalance_sensitivity.csv", "temporal_probe_profiles.csv",
    "model_recovery_matrix.csv", "identifiability_summary.csv",
    "final_behavioral_mechanism_summary.csv",
)


def _progress(enabled, message):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] HUMAN-CONFIGURED-V3 | {message}", flush=True)


def _limit_participants(frame, limit):
    if limit is None:
        return frame
    participants = sorted(frame.participant_uid.unique())[:int(limit)]
    return frame[frame.participant_uid.isin(participants)].copy()


def _srn_output(seed_profiles, aggregate, probe):
    seed = seed_profiles[seed_profiles.probe_context.eq(probe)].copy()
    seed["row_type"] = "seed"
    group = aggregate[aggregate.probe_context.eq(probe)].copy()
    group["row_type"] = "all_seed_aggregate"
    group["seed"] = "ALL"
    return pd.concat([seed, group], ignore_index=True, sort=False)


def _recovery_templates(profiles, mixed, srn_aggregate):
    positions = list(range(1, 8))
    mixed_profiles = profiles[profiles.dataset.eq("Combined")]
    delta = mixed_profiles.groupby("transition_position").delta.mean().reindex(positions)
    stable = mixed[mixed.ratio_reliable & mixed.responsibility_q.notna()]
    m2 = (stable.groupby("transition_position").responsibility_q.mean() - 0.5).reindex(
        positions
    ).fillna(0)
    matrix = mixed_profiles.pivot_table(
        index="participant_uid", columns="transition_position", values="delta", aggfunc="first"
    ).dropna().reindex(columns=positions)
    centered = matrix.to_numpy() - matrix.to_numpy().mean(axis=0)
    if len(centered) >= 2 and np.linalg.norm(centered) > 1e-12:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        m3 = vt[0]
        if float(m3 @ m2.to_numpy()) < 0:
            m3 *= -1
    else:
        m3 = m2.to_numpy()
    srn = srn_aggregate[srn_aggregate.probe_context.eq("Mixed_norm")].set_index(
        "transition_position"
    ).execution_q.reindex(positions).fillna(0.5) - 0.5
    templates = {
        "M0": np.zeros(7), "M1": np.ones(7), "M2": m2.to_numpy(dtype=float),
        "M3": np.asarray(m3, dtype=float), "M4": srn.to_numpy(dtype=float),
    }
    for name in ("M2", "M3", "M4"):
        norm = float(np.std(templates[name]))
        if norm > 1e-12:
            templates[name] = templates[name] / norm
    endpoint = delta.abs().fillna(delta.abs().mean()).to_numpy(dtype=float)
    endpoint = np.maximum(endpoint, 0.01)
    counts = mixed.groupby(
        ["participant_uid", "transition_position"]
    ).size().unstack(fill_value=0).reindex(columns=positions, fill_value=0).to_numpy()
    residual = mixed.log_rt - mixed.groupby(
        ["participant_uid", "transition_position"]
    ).log_rt.transform("mean")
    noise = max(float(residual.std()), 0.05)
    return templates, endpoint, counts, noise


def _final_summary(context_model, reliability, mixed_comparison, novel_comparison,
                   alignment, identifiability, config, participants):
    joint = context_model[context_model.row_type.eq("joint_test")]
    interaction = bool(joint.context_position_interaction_detected.any())
    participant_reliability = reliability[reliability.row_type.eq("participant")]
    mean_reliability = float(participant_reliability.spearman_brown_reliability.mean())
    reliability_pass = bool(np.isfinite(mean_reliability) and
                            mean_reliability >= config.reliability_threshold)
    mixed_winner = str(mixed_comparison.sort_values("primary_rank").model.iloc[0])
    novel_winner = str(novel_comparison.sort_values("primary_rank").model.iloc[0])
    mixed_index = mixed_comparison.set_index("model")
    novel_index = novel_comparison.set_index("model")
    m4_gain_mixed = float(
        mixed_index.loc["M4_srn_informed", "heldout_log_likelihood"] -
        mixed_index.loc["M1_scalar_mixture", "heldout_log_likelihood"]
    )
    m4_gain_novel = float(
        novel_index.loc["M4_srn_informed", "heldout_log_likelihood"] -
        novel_index.loc["M1_scalar_mixture", "heldout_log_likelihood"]
    )
    align_mixed = alignment[
        alignment.probe_context.eq("Both") & alignment.profile_variant.eq("execution")
    ].iloc[0]
    align_novel = alignment[
        alignment.probe_context.eq("Novel") & alignment.profile_variant.eq("execution")
    ].iloc[0]
    m4_identifiable = bool(
        identifiability.set_index("model").loc["M4", "recovery_above_chance"]
    )
    mixed_srn_predicts = bool(m4_gain_mixed > 0 and align_mixed.pearson_r > 0)
    novel_srn_predicts = bool(m4_gain_novel > 0 and align_novel.pearson_r > 0)
    if not interaction:
        code = "A"; text = "No reliable human Context x Position interaction was detected."
    elif mixed_winner in {"M0_scalar_context", "M1_scalar_mixture"}:
        code = "B"; text = "Pure A/B token structure exists, but a scalar Mixed model predicts best."
    elif mixed_winner == "M2_token_specific":
        code = "C"; text = "A generic token-specific Mixed model outpredicts the scalar mixture."
    elif mixed_winner == "M3_low_rank":
        code = "D"; text = "A low-rank human configured-state model predicts Mixed best."
    elif mixed_srn_predicts and novel_srn_predicts:
        code = "G"; text = "The frozen SRN profile predicts both Mixed and Novel human structure."
    elif mixed_srn_predicts:
        code = "F"; text = "The frozen SRN profile predicts Mixed but not Novel."
    else:
        code = "E"; text = "The SRN-informed model ranks first, but the full alignment test is incomplete."
    strong_allowed = bool(reliability_pass and m4_identifiable)
    return pd.DataFrame([{
        "main_result_code": code, "main_result_text": text,
        "context_position_interaction_detected": interaction,
        "minimum_context_position_p": float(joint.joint_interaction_p.min()),
        "group_mean_profile_reliability": mean_reliability,
        "reliability_gate_passed": reliability_pass,
        "mixed_winning_model": mixed_winner, "novel_winning_model": novel_winner,
        "mixed_M4_log_likelihood_gain_over_M1": m4_gain_mixed,
        "novel_M4_log_likelihood_gain_over_M1": m4_gain_novel,
        "mixed_srn_human_pearson_r": float(align_mixed.pearson_r),
        "novel_srn_human_pearson_r": float(align_novel.pearson_r),
        "mixed_srn_prediction_supported": mixed_srn_predicts,
        "novel_srn_prediction_supported": novel_srn_predicts,
        "M4_recovery_above_chance": m4_identifiable,
        "strong_srn_interpretation_allowed": strong_allowed,
        "interpretation_limited_by_identifiability_or_reliability": not strong_allowed,
        "mixed_participants": participants["Combined"],
        "novel_participants": participants["Novel"],
        "primary_metric": "heldout_log_likelihood",
        "position_definition": "SRN-aligned transitions 1->2 through 7->8",
    }])


def _write_report(path, *, config, summary, source, tables):
    row = summary.iloc[0]
    mixed = tables["mixed_model_comparison"].sort_values("primary_rank")
    novel = tables["novel_model_comparison"].sort_values("primary_rank")
    path.write_text(f"""# PRD v3 human configured-state behavioral report

Preprocessing: **retrieval_v1 exclusions, expanded to seven SRN-aligned response transitions**  
Primary DV: **log RT**  
Primary metric: **held-out log likelihood**  
Frozen SRN source: `{source.root}`  
Human outcomes used to select SRN seeds/profile shape: **0**

## Decision

**Result {row.main_result_code}.** {row.main_result_text}

- Context × Position detected: **{row.context_position_interaction_detected}**
- Mean split-half profile reliability: **{row.group_mean_profile_reliability:.3f}**
- Reliability gate: **{row.reliability_gate_passed}**
- Mixed winner: **{row.mixed_winning_model}**
- Novel winner: **{row.novel_winning_model}**
- M4 recovery above chance: **{row.M4_recovery_above_chance}**
- Strong SRN interpretation allowed: **{row.strong_srn_interpretation_allowed}**

## Held-out model comparison

Mixed ranks: {', '.join(mixed.model.tolist())}  
Novel ranks: {', '.join(novel.model.tolist())}

Pure A/B endpoints were estimated and frozen before any probe model was fit. Group token
structure was learned only from training participants. Held-out participants used odd sequence
repetitions for the permitted scalar calibration and even repetitions for evaluation. Mixed and
Novel were fit separately except in the explicitly labeled shared-profile hypothesis. Human
outcomes never changed an SRN checkpoint, seed set, seed weight, or token-profile shape.

The result code is descriptive when profile reliability or model recovery fails its gate; the
report does not elevate such a result to the paper-level mechanistic claim.
""", encoding="utf-8")


def run_human_configured_state_pipeline(
    combined_path, novel_path, srn_source_dir, output_dir,
    *, config: HumanConfiguredStateConfig | None = None,
):
    config = config or HumanConfiguredStateConfig()
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress, "Loading human CSVs through the validated preprocessing contract")
    combined = load_human_token_data(combined_path, "Combined", target_probe="Both")
    novel = load_human_token_data(novel_path, "Novel", target_probe="Novel")
    original_participants = {
        "Combined": int(combined.participant_uid.nunique()),
        "Novel": int(novel.participant_uid.nunique()),
    }
    combined = _limit_participants(combined, config.participant_limit_per_dataset)
    novel = _limit_participants(novel, config.participant_limit_per_dataset)
    participants = {
        "Combined": int(combined.participant_uid.nunique()),
        "Novel": int(novel.participant_uid.nunique()),
    }
    human = pd.concat([combined, novel], ignore_index=True)
    _progress(config.show_progress,
              f"Estimating frozen pure A/B profiles for {sum(participants.values())} participants")
    profiles = estimate_ab_profiles(
        human, shrinkage_strength=config.profile_shrinkage_strength
    )
    reliability = profile_reliability(human)
    context_model = fit_context_position_models(human)
    mixed = compute_midpoint_deviation(
        combined[combined.context.eq("Both")], profiles,
        endpoint_epsilon=config.endpoint_epsilon_log_rt,
    )
    novel_probe = compute_midpoint_deviation(
        novel[novel.context.eq("Novel")], profiles,
        endpoint_epsilon=config.endpoint_epsilon_log_rt,
    )

    _progress(config.show_progress, "Deriving preregistered all-seed frozen SRN profiles")
    srn_seed, srn_aggregate, srn_audit, source = load_frozen_srn_profiles(
        srn_source_dir, endpoint_epsilon=config.srn_endpoint_epsilon_rt
    )
    _progress(config.show_progress, "Fitting separate held-out Mixed and Novel model ladders")
    mixed_predictions = fit_probe_model_ladder(
        mixed, profiles, srn_aggregate, config=config
    )
    novel_predictions = fit_probe_model_ladder(
        novel_probe, profiles, srn_aggregate, config=config
    )
    predictions = pd.concat([mixed_predictions, novel_predictions], ignore_index=True)
    comparison = summarize_model_comparison(predictions)
    mixed_comparison = comparison[comparison.probe_context.eq("Both")].copy()
    novel_comparison = comparison[comparison.probe_context.eq("Novel")].copy()

    temporal = temporal_probe_profiles(mixed, novel_probe)
    counterbalance = counterbalance_sensitivity([mixed, novel_probe], predictions)
    alignment = human_srn_alignment(
        [mixed, novel_probe], srn_aggregate, comparison
    )
    mixed_novel = mixed_novel_profile_comparison(mixed, novel_probe)
    shared_separate = evaluate_shared_vs_separate(
        mixed, novel_probe, n_folds=config.cv_folds
    )

    _progress(config.show_progress, "Running observed-scale model recovery")
    templates, endpoint, counts, noise = _recovery_templates(
        profiles, mixed, srn_aggregate
    )
    recovery, identifiability = run_recovery(
        templates, participant_count=participants["Combined"],
        observations_per_position=int(np.median(counts[counts > 0])),
        noise_sd=noise, replicates=config.recovery_replicates,
        random_seed=config.random_seed, observation_counts=counts,
        endpoint_scale=endpoint, effect_scale=config.recovery_effect_scale,
    )
    summary = _final_summary(
        context_model, reliability, mixed_comparison, novel_comparison,
        alignment, identifiability, config, participants,
    )

    srn_mixed = _srn_output(srn_seed, srn_aggregate, "Mixed_norm")
    srn_novel = _srn_output(srn_seed, srn_aggregate, "Novel")
    tables = {
        "pure_ab_token_profiles": profiles,
        "pure_ab_profile_reliability": reliability,
        "context_position_model": context_model,
        "mixed_midpoint_deviation": mixed,
        "novel_midpoint_deviation": novel_probe,
        "model_M0_scalar_context": predictions[predictions.model.eq("M0_scalar_context")],
        "model_M1_scalar_mixture": predictions[predictions.model.eq("M1_scalar_mixture")],
        "model_M2_token_specific": predictions[predictions.model.eq("M2_token_specific")],
        "model_M3_low_rank": predictions[predictions.model.eq("M3_low_rank")],
        "model_M4_srn_informed": predictions[predictions.model.str.startswith("M4_")],
        "novel_model_comparison": novel_comparison,
        "mixed_model_comparison": mixed_comparison,
        "srn_mixed_profile": srn_mixed,
        "srn_novel_profile": srn_novel,
        "human_srn_alignment": alignment,
        "mixed_vs_novel_profile_comparison": mixed_novel,
        "shared_vs_separate_probe_profile": shared_separate,
        "counterbalance_sensitivity": counterbalance,
        "temporal_probe_profiles": temporal,
        "model_recovery_matrix": recovery,
        "identifiability_summary": identifiability,
        "final_behavioral_mechanism_summary": summary,
        "srn_frozen_audit": srn_audit,
    }
    for name, frame in tables.items():
        if frame.empty:
            raise AssertionError(f"Required v3 output {name} is empty.")
        frame.to_csv(output / f"{name}.csv", index=False)
    preprocessing_audit = pd.DataFrame([
        {
            "dataset": name, "original_valid_participants": original_participants[name],
            "analyzed_participants": participants[name],
            "token_rows": int((combined if name == "Combined" else novel).shape[0]),
            "transition_positions": "1|2|3|4|5|6|7",
            "raw_first_sequence_excluded": True,
            "response_position_one_excluded_for_srn_alignment": True,
            "preprocessing_contract": "retrieval_v1",
            "position_definition": "SRN-aligned transitions 1->2 through 7->8",
        } for name in ("Combined", "Novel")
    ])
    preprocessing_audit.to_csv(output / "human_token_preprocessing_audit.csv", index=False)
    record = {
        **asdict(config), "analysis_version": "3.0.0",
        "combined_path": str(Path(combined_path).resolve()),
        "novel_path": str(Path(novel_path).resolve()),
        "srn_source_dir": str(source.root),
        "participants": participants, "original_valid_participants": original_participants,
        "preprocessing_contract": "retrieval_v1",
        "position_definition": "SRN-aligned transitions 1->2 through 7->8",
        "pure_profile_source_contexts": ["A", "B"],
        "probe_rows_used_for_pure_profiles": 0,
        "srn_aggregation": "all_registered_seed_mean",
        "srn_seed_count": int(srn_seed.seed.nunique()),
        "human_outcomes_used_for_srn_seed_selection": 0,
        "human_outcomes_used_for_srn_profile_shape": 0,
        "primary_metric": "heldout_log_likelihood",
        "final_inference_ready": bool(config.profile == "final" and
                                      config.participant_limit_per_dataset is None),
        "required_csvs": list(REQUIRED_CSVS),
        "main_result_code": summary.main_result_code.iloc[0],
    }
    (output / "human_run_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_report(
        output / "human_behavioral_report.md", config=config, summary=summary,
        source=source, tables=tables,
    )
    if config.make_plots:
        _progress(config.show_progress, "Rendering eleven registered figures")
        cache = Path(tempfile.gettempdir()) / "human-configured-v3-matplotlib"
        cache.mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(cache))
        from .plotting.figures import make_required_figures
        make_required_figures(
            output / "figures_human_configured_v3", tables=tables, config=config
        )
    _progress(config.show_progress, f"Complete: {output}")
    return tables

