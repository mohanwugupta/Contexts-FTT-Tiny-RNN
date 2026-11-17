"""Discovery, confirmation, and mechanistic analysis for PRD v1.2."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analysis.diagnostics import (
    batch_to_probe_frame,
    evaluate_checkpoints,
    evaluate_diagnostic,
    evaluate_heldout_prediction,
    generate_diagnostic_battery,
)
from .analysis.hidden_probes import run_hidden_probes
from .analysis.interventions import run_causal_interventions
from .config import (
    CUE_ACCURACY_LEVELS,
    OUTCOME_NOISE_LEVELS,
    STATE_SEPARATION_LEVELS,
    VALIDITY_LEVELS,
    EnvironmentalCondition,
    ExperimentConfig,
    discovery_conditions,
)
from .confirmation.seed_registry import validate_fresh_seeds
from .discovery.selection import select_promising, summarize_conditions
from .environment.cue_evidence import cue_sd_for_accuracy
from .training.trainer import train_condition


REQUIRED_CSVS = (
    "all_environmental_conditions.csv", "all_seed_results.csv", "condition_summary.csv",
    "posterior_mean_alignment.csv", "implied_responsibility.csv", "ambiguity_diagnostics.csv",
    "phase_only_predictions.csv", "cue_only_predictions.csv", "full_bayes_predictions.csv",
    "model_similarity_comparison.csv", "emergence_by_switch.csv",
    "phase_diagram_validity_x_uncertainty.csv",
    "phase_diagram_uncertainty_x_separation.csv", "promising_regimes.csv",
    "confirmation_results.csv", "confirmed_regimes.csv", "environmental_ablation_results.csv",
    "hidden_state_probes.csv", "causal_interventions.csv",
)


def _condition_row(condition: EnvironmentalCondition, config: ExperimentConfig):
    return {
        "condition_id": condition.condition_id,
        "context_validity": condition.context_validity,
        "cue_accuracy": condition.cue_accuracy,
        "cue_sd": cue_sd_for_accuracy(condition.cue_accuracy, config.cue_mean),
        "state_separation": condition.state_separation,
        "outcome_noise_sd": condition.outcome_noise_sd,
        "state_variant": condition.state_variant,
        "hidden_dim": config.hidden_dim,
        "block_length": config.block_length,
        "n_blocks": config.n_blocks,
        "training_trials": config.training_trials,
        "weight_updates": config.weight_updates,
        "training_batch_size": config.training_batch_size,
        "learning_rate": config.learning_rate,
    }


def _condition_from_row(row) -> EnvironmentalCondition:
    return EnvironmentalCondition(
        float(row.context_validity), float(row.cue_accuracy), float(row.state_separation),
        float(row.outcome_noise_sd), str(row.state_variant),
    )


def _attach(frame: pd.DataFrame, condition: EnvironmentalCondition, seed: int, status="completed"):
    result = frame.copy()
    metadata = _condition_row(condition, ExperimentConfig())
    # Only environmental identity belongs in row-level analysis tables.
    for name in (
        "state_variant", "outcome_noise_sd", "state_separation", "cue_sd",
        "cue_accuracy", "context_validity", "condition_id",
    ):
        result.insert(0, name, metadata[name])
    result.insert(0, "seed", int(seed))
    result.insert(0, "status", status)
    return result


def _not_run(stage: str, reason: str):
    return pd.DataFrame([{"status": "not_run", "stage": stage, "reason": reason}])


def _run_seed(condition, config, seed, report, *, stage):
    report(f"{stage} | {condition.condition_id} | seed {seed} | training")
    training = train_condition(
        condition, config, seed=seed,
        progress=lambda message: report(
            f"{stage} | {condition.condition_id} | seed {seed} | {message}"
        ),
    )
    battery = generate_diagnostic_battery(
        condition, config, seed=50_000 + int(seed)
    )
    diagnostics, metrics, diagnostic_hidden = evaluate_diagnostic(training.model, battery)
    heldout_metrics, heldout_batch, heldout_hidden = evaluate_heldout_prediction(
        training.model, condition, config, seed=100_000 + int(seed)
    )
    emergence = evaluate_checkpoints(training.checkpoints, condition, config, battery)
    seed_row = {
        "status": "completed",
        "stage": stage,
        "seed": int(seed),
        **_condition_row(condition, config),
        **heldout_metrics,
        **metrics,
        "training_updates_observed": training.weight_updates,
        "fixed_exposure_verified": training.weight_updates == config.weight_updates,
    }
    diagnostics = _attach(diagnostics, condition, seed)
    emergence = _attach(emergence, condition, seed)
    probe_frame = batch_to_probe_frame(heldout_batch, heldout_hidden)
    return {
        "seed_row": seed_row,
        "diagnostics": diagnostics,
        "emergence": emergence,
        "model": training.model,
        "probe_frame": probe_frame,
        "diagnostic_hidden": diagnostic_hidden,
    }


def _similarity_long(seed_results: pd.DataFrame):
    candidates = {
        "full_bayes": "srn_to_full_bayes_mse",
        "phase_only": "srn_to_phase_only_mse",
        "cue_only": "srn_to_cue_only_mse",
        "hard_map": "srn_to_hard_map_mse",
        "global_midpoint": "srn_to_global_midpoint_mse",
    }
    identity = [
        "status", "stage", "condition_id", "seed", "context_validity", "cue_accuracy",
        "state_separation", "outcome_noise_sd", "state_variant",
    ]
    rows = []
    for row in seed_results.itertuples(index=False):
        best = min(candidates, key=lambda name: float(getattr(row, candidates[name])))
        for candidate, column in candidates.items():
            record = {name: getattr(row, name) for name in identity}
            record.update({
                "candidate_model": candidate,
                "srn_similarity_mse": float(getattr(row, column)),
                "best_matching_model": candidate == best,
            })
            rows.append(record)
    return pd.DataFrame(rows)


def _phase_diagrams(summary: pd.DataFrame):
    validity_cue = summary.groupby(["context_validity", "cue_accuracy"], as_index=False).agg(
        mean_posterior_mean_r2=("mean_posterior_mean_r2", "mean"),
        mean_pass_rate=("pass_rate", "mean"),
        n_conditions=("condition_id", "nunique"),
    )
    validity_cue.insert(0, "status", "completed")
    cue_separation = summary.groupby(["cue_accuracy", "state_separation"], as_index=False).agg(
        mean_posterior_mean_r2=("mean_posterior_mean_r2", "mean"),
        mean_pass_rate=("pass_rate", "mean"),
        n_conditions=("condition_id", "nunique"),
    )
    cue_separation.insert(0, "status", "completed")
    return validity_cue, cue_separation


def _confirmation_summary(results: pd.DataFrame, required: int):
    completed = results[results.status.eq("completed")] if "status" in results else results
    if completed.empty:
        return _not_run("confirmed_regimes", "confirmation was not run")
    grouping = [
        "condition_id", "context_validity", "cue_accuracy", "cue_sd",
        "state_separation", "outcome_noise_sd", "state_variant",
    ]
    summary = completed.groupby(grouping, as_index=False).agg(
        n_fresh_seeds=("seed", "nunique"), passing_fresh_seeds=("passed", "sum"),
        pass_rate=("passed", "mean"), mean_posterior_mean_r2=("posterior_mean_r2", "mean"),
        mean_implied_q_r2=("implied_q_r2", "mean"),
    )
    summary.insert(0, "status", "completed")
    summary["confirmation_pass_required"] = int(required)
    summary["confirmed"] = summary.passing_fresh_seeds >= int(required)
    return summary


def _ablation_conditions(anchor: EnvironmentalCondition, config: ExperimentConfig):
    specifications = [
        ("A1_restore_perfect_validity", EnvironmentalCondition(1.0, anchor.cue_accuracy, anchor.state_separation,
                                                               anchor.outcome_noise_sd, anchor.state_variant)),
        ("A2_collapse_state_separation", EnvironmentalCondition(anchor.context_validity, anchor.cue_accuracy, 0.05,
                                                                 anchor.outcome_noise_sd, anchor.state_variant)),
        ("A3_remove_graded_cues", EnvironmentalCondition(anchor.context_validity, 1.0, anchor.state_separation,
                                                          anchor.outcome_noise_sd, anchor.state_variant)),
        ("A4_destroy_cue_information", EnvironmentalCondition(anchor.context_validity, 0.51, anchor.state_separation,
                                                               anchor.outcome_noise_sd, anchor.state_variant)),
    ]
    if config.run_followups:
        specifications.extend(
            (f"outcome_noise_{noise:.2f}", EnvironmentalCondition(
                anchor.context_validity, anchor.cue_accuracy, anchor.state_separation,
                noise, anchor.state_variant,
            )) for noise in OUTCOME_NOISE_LEVELS
        )
        specifications.extend(
            (f"state_variant_{variant}", EnvironmentalCondition(
                anchor.context_validity, anchor.cue_accuracy, anchor.state_separation,
                anchor.outcome_noise_sd, variant,
            )) for variant in ("fixed_global", "fixed_per_run", "context_learning")
        )
    seen = set()
    unique = []
    for label, condition in specifications:
        key = (label, condition.condition_id)
        if key not in seen:
            seen.add(key)
            unique.append((label, condition))
    return unique


def _write_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_text(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _scientific_decision(config, frames):
    confirmed = _completed_rows(frames["confirmed_regimes"])
    confirmed = confirmed[confirmed.confirmed] if not confirmed.empty and "confirmed" in confirmed else pd.DataFrame()
    if confirmed.empty:
        similarity = _completed_rows(frames["model_similarity_comparison"])
        if not similarity.empty:
            closest = similarity.groupby("candidate_model").srn_similarity_mse.mean().idxmin()
            if closest == "phase_only":
                return "Result D: the SRN most closely follows the block-phase shortcut; no regime was confirmed."
            if closest == "global_midpoint":
                return "Result E: the SRN most closely follows a constant midpoint; no regime was confirmed."
        return "No environmental regime passed fresh-seed confirmation."
    labels = []
    if ((confirmed.context_validity < 1.0) & (confirmed.context_validity > 0.55)).any():
        labels.append("A (intermediate context validity)")
    if (confirmed.state_separation >= 0.60).all():
        labels.append("B (large state separation)")
    if (confirmed.cue_accuracy < 0.97).any():
        labels.append("C (graded cue overlap)")
    emergence = _completed_rows(frames["emergence_by_switch"])
    emergence = emergence[emergence.condition_id.isin(set(confirmed.condition_id))]
    first_passes = []
    for _, group in emergence.groupby(["condition_id", "seed"]):
        passed = group[group.passed]
        if not passed.empty:
            first_passes.append(int(passed.switch_number.min()))
    if first_passes and float(np.median(first_passes)) <= 3:
        labels.append("F (emergence within the first three A/B pairs)")
    joined = ", ".join(labels) if labels else "confirmed posterior averaging"
    return f"Confirmed support: {joined}."


def _report_text(config, frames, decision):
    summary = frames["condition_summary"]
    promising = _completed_rows(frames["promising_regimes"])
    confirmed = _completed_rows(frames["confirmed_regimes"])
    confirmed = confirmed[confirmed.confirmed] if not confirmed.empty and "confirmed" in confirmed else confirmed
    best = summary.sort_values("mean_posterior_mean_r2", ascending=False).iloc[0]
    lines = [
        "# Environmental-pressure SRN report", "",
        f"**Decision: {decision}**", "",
        "## Fixed training ecology", "",
        f"- Architecture: eight-unit tanh Elman SRN",
        f"- Schedule: {config.n_blocks} blocks × exactly 10 trials, A/B alternating",
        f"- Updates per seed: {config.weight_updates}",
        f"- Optimizer: Adam, learning rate {config.learning_rate}",
        "- Optimization dimensions searched: none", "",
        "## Environmental search", "",
        f"- Conditions evaluated: {len(summary)}",
        f"- Promising regimes: {len(promising)}",
        f"- Confirmed regimes: {len(confirmed)}",
        f"- Highest mean posterior R² condition: {best.condition_id} ({best.mean_posterior_mean_r2:.4f})",
        "",
    ]
    if config.profile == "smoke":
        lines.append("This is a software smoke run and must not be interpreted as a scientific phase diagram.")
    elif len(confirmed):
        lines.append(
            "Fresh-seed confirmation identified at least one environment meeting the prespecified "
            "Bayesian-averaging gate. Environmental ablations should be used to assess necessity."
        )
    else:
        lines.append(
            "No regime met fresh-seed confirmation. Do not claim necessary or sufficient environmental pressures."
        )
    lines.extend(["", "No human Mixed/Novel outcomes were used in training or evaluation."])
    ablations = _completed_rows(frames["environmental_ablation_results"])
    if config.profile != "smoke" and len(confirmed) and not ablations.empty:
        lines.extend(["", "## Necessity and sufficiency within the tested ecology", ""])
        anchor_r2 = float(confirmed.mean_posterior_mean_r2.max())
        causal = ablations[ablations.manipulation.str.startswith("A", na=False)]
        necessary = []
        for manipulation, group in causal.groupby("manipulation"):
            ablated_r2 = float(group.posterior_mean_r2.mean())
            change = anchor_r2 - ablated_r2
            lines.append(
                f"- {manipulation}: mean R²={ablated_r2:.4f}; change from confirmed anchor={change:+.4f}."
            )
            if change >= 0.20:
                necessary.append(manipulation)
        if necessary:
            lines.append(
                "- Operational necessity criterion (R² decrease ≥ .20) was met by: "
                + ", ".join(necessary) + "."
            )
        else:
            lines.append("- No single ablation met the prespecified operational necessity threshold of a .20 R² decrease.")
        lines.append(
            "- Any confirmed anchor is operationally sufficient within the tested architecture, exposure, and seed criterion; "
            "this is not a claim of universal causal sufficiency."
        )
    return "\n".join(lines) + "\n"


def _completed_rows(frame):
    if frame.empty:
        return frame
    return frame[frame.status.eq("completed")] if "status" in frame else frame


def run_pipeline(output_dir: str | Path, *, config: ExperimentConfig | None = None):
    config = config or ExperimentConfig()
    config.validate()
    validate_fresh_seeds(config.discovery_seeds, config.confirmation_seeds)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def report(message):
        if config.show_progress:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] SRN-ENV | {message}", flush=True)

    conditions = config.conditions or discovery_conditions()
    condition_table = pd.DataFrame([_condition_row(condition, config) for condition in conditions])
    condition_table.insert(0, "status", "completed")
    condition_table.insert(1, "stage", "discovery")
    seed_rows, diagnostic_frames, emergence_frames, discovery_bundles = [], [], [], []
    total = len(conditions) * len(config.discovery_seeds)
    report(f"Starting environmental discovery: {len(conditions)} conditions × "
           f"{len(config.discovery_seeds)} seeds = {total} networks")
    counter = 0
    for condition in conditions:
        for seed in config.discovery_seeds:
            counter += 1
            report(f"Discovery network {counter}/{total}")
            bundle = _run_seed(condition, config, int(seed), report, stage="discovery")
            seed_rows.append(bundle["seed_row"])
            diagnostic_frames.append(bundle["diagnostics"])
            emergence_frames.append(bundle["emergence"])
            discovery_bundles.append(bundle)
    seed_results = pd.DataFrame(seed_rows)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    emergence = pd.concat(emergence_frames, ignore_index=True)
    condition_summary = summarize_conditions(seed_results, config.discovery_pass_required)
    condition_summary.insert(0, "status", "completed")
    promising_raw = select_promising(condition_summary, config.max_confirmation_regimes)
    promising = promising_raw.copy()
    if promising.empty:
        promising = _not_run("promising_regimes", "no condition met the discovery gate")
    else:
        promising.insert(0, "status", "completed")
    validity_cue, cue_separation = _phase_diagrams(condition_summary)

    confirmation_frames = []
    confirmation_bundles = []
    candidates = promising_raw
    if candidates.empty and config.force_staged_analyses:
        candidates = condition_summary.sort_values("mean_posterior_mean_r2", ascending=False).head(1)
    if config.run_confirmation and not candidates.empty:
        confirmation_total = len(candidates) * len(config.confirmation_seeds)
        report(f"Starting fresh-seed confirmation: {confirmation_total} networks")
        count = 0
        for candidate in candidates.itertuples(index=False):
            condition = _condition_from_row(candidate)
            for seed in config.confirmation_seeds:
                count += 1
                report(f"Confirmation network {count}/{confirmation_total}")
                bundle = _run_seed(condition, config, int(seed), report, stage="confirmation")
                confirmation_frames.append(pd.DataFrame([bundle["seed_row"]]))
                confirmation_bundles.append(bundle)
        confirmation_results = pd.concat(confirmation_frames, ignore_index=True)
    else:
        reason = "disabled by configuration" if not config.run_confirmation else "no promising regime"
        confirmation_results = _not_run("confirmation_results", reason)
    confirmed_regimes = _confirmation_summary(confirmation_results, config.confirmation_pass_required)
    confirmed_rows = _completed_rows(confirmed_regimes)
    confirmed_rows = confirmed_rows[confirmed_rows.confirmed] \
        if not confirmed_rows.empty and "confirmed" in confirmed_rows else pd.DataFrame()

    anchor = None
    if not confirmed_rows.empty:
        anchor = _condition_from_row(confirmed_rows.sort_values("mean_posterior_mean_r2", ascending=False).iloc[0])
    elif config.force_staged_analyses and not candidates.empty:
        anchor = _condition_from_row(candidates.iloc[0])

    ablation_frames = []
    if anchor is not None and config.run_ablations:
        specifications = _ablation_conditions(anchor, config)
        report(f"Running {len(specifications)} environmental ablation/follow-up conditions")
        for manipulation, condition in specifications:
            for seed in config.discovery_seeds:
                bundle = _run_seed(condition, config, int(seed), report, stage="ablation")
                row = bundle["seed_row"].copy()
                row["manipulation"] = manipulation
                row["anchor_condition_id"] = anchor.condition_id
                ablation_frames.append(pd.DataFrame([row]))
        ablations = pd.concat(ablation_frames, ignore_index=True)
    else:
        reason = "disabled by configuration" if not config.run_ablations else "no confirmed regime"
        ablations = _not_run("environmental_ablation_results", reason)

    hidden_frames, intervention_frames = [], []
    mechanism_bundles = confirmation_bundles
    if config.force_staged_analyses and not mechanism_bundles:
        mechanism_bundles = discovery_bundles[:1]
    confirmed_ids = set(confirmed_rows.condition_id) if not confirmed_rows.empty else set()
    if config.run_hidden_analysis and mechanism_bundles and (confirmed_ids or config.force_staged_analyses):
        report("Running held-out-run hidden probes and prior/likelihood interventions")
        for bundle in mechanism_bundles:
            condition_id = bundle["seed_row"]["condition_id"]
            if confirmed_ids and condition_id not in confirmed_ids:
                continue
            seed = int(bundle["seed_row"]["seed"])
            probes = run_hidden_probes(bundle["probe_frame"], seed=seed + 71)
            probes.insert(0, "seed", seed)
            probes.insert(0, "condition_id", condition_id)
            hidden_frames.append(probes)
            q_r2 = float(probes.loc[probes.target.eq("q_star"), "heldout_r2"].iloc[0])
            interventions = run_causal_interventions(
                bundle["model"], bundle["probe_frame"], q_r2,
                force=config.force_staged_analyses,
            )
            interventions.insert(0, "seed", seed)
            interventions.insert(0, "condition_id", condition_id)
            intervention_frames.append(interventions)
        hidden_probes = pd.concat(hidden_frames, ignore_index=True) if hidden_frames else _not_run(
            "hidden_state_probes", "no confirmed model bundle"
        )
        interventions = pd.concat(intervention_frames, ignore_index=True) if intervention_frames else _not_run(
            "causal_interventions", "no decodable posterior representation"
        )
    else:
        reason = "disabled by configuration" if not config.run_hidden_analysis else "behavior not confirmed"
        hidden_probes = _not_run("hidden_state_probes", reason)
        interventions = _not_run("causal_interventions", reason)

    posterior_alignment_columns = [
        "status", "stage", "condition_id", "seed", "context_validity", "cue_accuracy",
        "state_separation", "outcome_noise_sd", "state_variant", "posterior_mean_r2",
        "implied_q_r2", "implied_q_slope", "implied_q_intercept", "implied_q_mae",
        "endpoint_error", "monotonicity_fraction", "passed",
    ]
    similarity = _similarity_long(seed_results)
    frames = {
        "all_environmental_conditions": condition_table,
        "all_seed_results": seed_results,
        "condition_summary": condition_summary,
        "posterior_mean_alignment": seed_results[posterior_alignment_columns].copy(),
        "implied_responsibility": diagnostics[[
            "status", "condition_id", "seed", "battery_id", "matched_state_history_id",
            "nominal_context", "block_position", "q_star", "implied_q", "skill_a", "skill_b",
            "srn_prediction", "bayes_mean",
        ]].copy(),
        "ambiguity_diagnostics": diagnostics,
        "phase_only_predictions": diagnostics[[
            "status", "condition_id", "seed", "battery_id", "matched_state_history_id",
            "nominal_context", "block_position", "x_t", "prior_q", "q_star",
            "phase_only_prediction", "srn_prediction",
        ]].copy(),
        "cue_only_predictions": diagnostics[[
            "status", "condition_id", "seed", "battery_id", "matched_state_history_id",
            "nominal_context", "block_position", "x_t", "cue_only_q", "q_star",
            "cue_only_prediction", "srn_prediction",
        ]].copy(),
        "full_bayes_predictions": diagnostics[[
            "status", "condition_id", "seed", "battery_id", "matched_state_history_id",
            "nominal_context", "block_position", "x_t", "prior_q", "likelihood_ratio",
            "q_star", "bayes_mean", "srn_prediction",
        ]].copy(),
        "model_similarity_comparison": similarity,
        "emergence_by_switch": emergence,
        "phase_diagram_validity_x_uncertainty": validity_cue,
        "phase_diagram_uncertainty_x_separation": cue_separation,
        "promising_regimes": promising,
        "confirmation_results": confirmation_results,
        "confirmed_regimes": confirmed_regimes,
        "environmental_ablation_results": ablations,
        "hidden_state_probes": hidden_probes,
        "causal_interventions": interventions,
    }
    export_map = {filename: filename[:-4] for filename in REQUIRED_CSVS}
    report(f"Writing {len(export_map)} required CSV artifacts")
    for filename, key in export_map.items():
        _write_csv(frames[key], output / filename)
    decision = _scientific_decision(config, frames)
    run_record = {
        "analysis": "PRD v1.2 environmental pressures for Bayes-optimal averaging",
        "version": __version__,
        "profile": config.profile,
        "config": asdict(config),
        "primary_architecture": "8-unit Elman SRN",
        "fixed_training": {
            "schedule": "A x 10, B x 10, repeated",
            "block_length": config.block_length,
            "n_blocks": config.n_blocks,
            "training_trials": config.training_trials,
            "weight_updates": config.weight_updates,
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
        },
        "optimization_dimensions_searched": [],
        "environmental_dimensions_searched": [
            "context_validity", "cue_accuracy", "state_separation"
        ],
        "network_inputs": ["x_t", "y_t_minus_1", "previous_outcome_mask", "session_start"],
        "forbidden_training_inputs": ["nominal_context", "latent_context", "q_star", "S_A", "S_B", "current_y_t"],
        "objective": "deterministic MSE only",
        "human_data_used": False,
        "decision": decision,
        "outputs": {filename: int(len(frames[key])) for filename, key in export_map.items()},
    }
    _write_text(json.dumps(run_record, indent=2), output / "run_config.json")
    _write_text(_report_text(config, frames, decision), output / "environmental_pressures_report.md")
    if config.make_plots:
        report("Rendering nine required figures")
        from .plotting import make_figures
        make_figures(output, frames)
    report(f"Analysis complete | {decision}")
    return frames
