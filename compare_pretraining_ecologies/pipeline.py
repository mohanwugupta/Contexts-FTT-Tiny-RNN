"""Gated P0--P5 comparison with a strict synthetic-to-human firewall."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from emergent_retrieval_program.analysis import profile_fits
from human_svg_geometry_transfer.geometry import load_coordinate_registry
from structural_prior_curriculum.analysis import human_tables

from .analysis import condition_scores, gate_conditions, generic_human_gates, select_conditions
from .config import EcologyComparisonConfig, PAIRED_CONTROLS, STRUCTURED
from .human import ablation_states, probe_saved_state, train_human
from .modeling import EcologyRun, new_engine, no_pretraining, pretrain
from .plotting import human_figures, internal_figures
from .validation import run_validation


PRETRAINING_REQUIRED = (
    "pretraining_ecology_config.json", "pretraining_world_manifest.csv",
    "pretraining_training_curves.csv", "pretraining_validation_v1.csv",
    "pretraining_validation_v2.csv", "pretraining_validation_v3.csv",
    "pretraining_validation_v4.csv", "pretraining_condition_scores.csv",
    "pretraining_gate_results.csv", "PRETRAINING_ECOLOGY_COMPARISON_REPORT.md",
)
HUMAN_REQUIRED = (
    "human_transfer_training.csv", "human_transfer_familiar_gates.csv",
    "human_transfer_first_probe.csv", "human_transfer_profiles.csv",
    "human_transfer_hidden_geometry.csv",
)


def _progress(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] ECOLOGY-COMPARE | {message}", flush=True)


def _checkpoint(output: Path, run: EcologyRun):
    target = output / "pretraining_checkpoints"; target.mkdir(parents=True, exist_ok=True)
    torch.save({
        "phase": run.phase, "pretraining_condition": run.condition,
        "paired_structured_condition": run.paired_structured_condition, "seed": run.seed,
        "model_state_dict": run.engine.model.state_dict(), "policy_state_dict": run.engine.policy.state_dict(),
        "audit": run.audit,
    }, target / f"{run.phase}_{run.condition}_seed_{run.seed}.pt")


def _run_phase(config, registry, phase, structured_conditions, seeds, worlds, output):
    runs, training, manifests, audits = {}, [], [], []
    v1s, v2s, v3s, v4s, validation_audits = [], [], [], [], []
    conditions = [value for condition in structured_conditions for value in (f"P1_for_{condition}", condition)]
    _progress(f"{phase}: {worlds} worlds; conditions={','.join(conditions)}")
    for seed in seeds:
        p0 = no_pretraining(config, seed); p0.phase = phase
        tables = run_validation(config, registry, p0, phase)
        for target, frame in zip((v1s, v2s, v3s, v4s), tables[:4]): target.append(frame)
        validation_audits.append(tables[4])
        for condition in conditions:
            run = pretrain(config, registry, condition, phase, seed, worlds, progress=_progress)
            runs[(condition, seed)] = run; training.append(run.training); audits.append(run.audit)
            if condition in STRUCTURED: manifests.append(run.worlds)
            _checkpoint(output, run)
            tables = run_validation(config, registry, run, phase)
            for target, frame in zip((v1s, v2s, v3s, v4s), tables[:4]): target.append(frame)
            validation_audits.append(tables[4])
    return {
        "runs": runs,
        "training": pd.concat(training, ignore_index=True) if training else pd.DataFrame(),
        "worlds": pd.DataFrame([row for frame in manifests for row in frame.to_dict("records")]),
        "audits": pd.DataFrame(audits), "validation_audits": pd.DataFrame(validation_audits),
        "v1": pd.concat(v1s, ignore_index=True), "v2": pd.concat(v2s, ignore_index=True),
        "v3": pd.concat(v3s, ignore_index=True), "v4": pd.concat(v4s, ignore_index=True),
    }


def _paired_exposure_assertion(audits: pd.DataFrame):
    for structured in STRUCTURED:
        one = audits[audits.paired_structured_condition.eq(structured)]
        if one.empty: continue
        pivot = one.pivot(index=["phase", "seed"], columns="pretraining_condition", values=[
            "external_sequences", "optimizer_steps", "cue_dimensionality", "layout_dimensionality"])
        for metric in ("external_sequences", "optimizer_steps", "cue_dimensionality", "layout_dimensionality"):
            if not (pivot[(metric, structured)] == pivot[(metric, f"P1_for_{structured}")]).all():
                raise AssertionError(f"Matched exposure failed for {structured}: {metric}")


def _profile_table(probes):
    rows = probes.groupby(["pretraining_condition", "seed", "probe_epoch", "context", "transition_position"], as_index=False).agg(
        normalized_RT=("normalized_RT", "mean"), prediction_NLL=("correct_token_nll", "mean"),
        next_token_accuracy=("token_correct", "mean"), correct_token_probability=("correct_token_probability", "mean"),
    )
    return rows


def _mechanism_geometry(probes: pd.DataFrame) -> pd.DataFrame:
    hidden_columns = [f"hidden_state_{index}" for index in range(1, 9)]
    rows = []
    for key, one in probes.groupby(["pretraining_condition", "seed", "human_training_checkpoint"]):
        lookup = one.set_index(["context", "transition_position"])
        states = one[hidden_columns].to_numpy(float)
        position_centroids = one.groupby("transition_position")[hidden_columns].mean().to_numpy(float)
        grand = states.mean(axis=0)
        total = float(np.square(states - grand).sum())
        ordinal = float(len(one.context.unique()) * np.square(position_centroids - grand).sum())
        rows.append({
            "pretraining_condition": key[0], "seed": key[1], "human_training_checkpoint": key[2],
            "metric": "shared_ordinal_variance_fraction", "context_pair": "all",
            "value": ordinal / total if total else np.nan,
        })
        for first, second in (("A", "B"), ("A", "Mixed"), ("B", "Mixed"),
                              ("A", "Novel"), ("B", "Novel"), ("Novel", "Neutral")):
            values = []
            for position in range(1, 8):
                a = lookup.loc[(first, position), hidden_columns].to_numpy(float)
                b = lookup.loc[(second, position), hidden_columns].to_numpy(float)
                values.append(float(np.linalg.norm(a - b)))
            rows.append({
                "pretraining_condition": key[0], "seed": key[1], "human_training_checkpoint": key[2],
                "metric": "mean_context_distance", "context_pair": f"{first}--{second}",
                "value": float(np.mean(values)),
            })
        for probe in ("Mixed", "Novel"):
            off, qs = [], []
            for position in range(1, 8):
                a = lookup.loc[("A", position), hidden_columns].to_numpy(float)
                b = lookup.loc[("B", position), hidden_columns].to_numpy(float)
                p = lookup.loc[(probe, position), hidden_columns].to_numpy(float)
                direction = b - a; denominator = float(direction @ direction)
                q = float(((p - a) @ direction) / denominator) if denominator > 1e-12 else 0.
                off.append(float(np.linalg.norm(p - (a + np.clip(q, 0, 1) * direction)))); qs.append(q)
            rows.extend([
                {"pretraining_condition": key[0], "seed": key[1], "human_training_checkpoint": key[2],
                 "metric": "AB_corridor_off_distance", "context_pair": probe, "value": float(np.mean(off))},
                {"pretraining_condition": key[0], "seed": key[1], "human_training_checkpoint": key[2],
                 "metric": "AB_corridor_global_q", "context_pair": probe, "value": float(np.mean(qs))},
            ])
    return pd.DataFrame(rows)


def _human_stage(config, registry, selected, selected_runs, output):
    top = selected[0]; groups = ["P0", f"P1_for_{top}", *selected]
    human_runs = []; human_run_lookup = {}
    for seed in config.confirmatory_seeds:
        sources = {"P0": no_pretraining(config, seed)}
        for group in groups[1:]: sources[group] = selected_runs[(group, seed)]
        for group in groups:
            result = train_human(config, registry, sources[group], group)
            human_runs.append(result); human_run_lookup[(group, seed)] = result
            _progress(f"human transfer {group} seed {seed}")
    training = pd.concat([run.training for run in human_runs], ignore_index=True)
    probes = pd.concat([run.probes for run in human_runs], ignore_index=True)
    audits = pd.DataFrame([run.audit for run in human_runs])
    summary, fits, classes, predictions, hidden, metrics, interpolation, corridor, first, frontier = human_tables(training, probes)
    gates = generic_human_gates(
        summary, hidden, classes, audits, groups,
        .5 if config.smoke else float(config.payload["human_gates"]["seed_fraction"]),
        float(config.payload["human_gates"]["minimum_P0_differentiation_ratio"]),
        float(config.payload["human_gates"]["minimum_RT_sawtooth"]),
    )
    profiles = _profile_table(probes)
    hidden_geometry = pd.concat([
        hidden.assign(metric_family="AB_hidden_separation"),
        corridor.assign(metric_family="probe_hidden_corridor"),
    ], ignore_index=True, sort=False)
    outputs = {
        "human_transfer_training.csv": training, "human_transfer_training_summary.csv": summary,
        "human_transfer_training_audit.csv": audits, "human_transfer_familiar_gates.csv": gates,
        "human_transfer_state_space.csv": fits, "human_transfer_state_space_classification.csv": classes,
        "human_transfer_state_space_predictions.csv": predictions,
        "human_transfer_probe_transition_level.csv": probes, "human_transfer_probe_metrics.csv": metrics,
        "human_transfer_interpolation.csv": interpolation, "human_transfer_first_probe.csv": first,
        "human_transfer_profiles.csv": profiles, "human_transfer_hidden_geometry.csv": hidden_geometry,
        "human_transfer_differentiation_frontier.csv": frontier,
    }
    for name, frame in outputs.items(): frame.to_csv(output / name, index=False)
    human_figures(output, training, probes, frontier[frontier.probe_context.eq("Mixed")])

    gate_lookup = gates.set_index("pretraining_condition").gate_passed
    first_mean = first.groupby("pretraining_condition").first_probe_between_endpoint_fraction.mean()
    improved = [condition for condition in selected if bool(gate_lookup.get(condition, False)) and
                first_mean.get(condition, float("-inf")) > first_mean.get(f"P1_for_{top}", float("inf"))]
    if improved and bool(config.payload["mechanistic_followups"]["enabled_after_structured_human_improvement"]):
        ablation_rows, geometry_rows, retention_rows = [], [], []
        for condition in improved:
            for seed in config.confirmatory_seeds:
                source = selected_runs[(condition, seed)]
                for label, states in ablation_states(config, source).items():
                    result = train_human(config, registry, source, f"{condition}:{label}", model_state=states[0], policy_state=states[1])
                    ablation_rows.append(result.probes.assign(ablation=label, source_condition=condition))
                human_run = human_run_lookup[(condition, seed)]
                before_model, before_policy = human_run.snapshots[0]
                before = probe_saved_state(config, registry, source, condition, before_model, before_policy, 0, "before_human")
                after = human_run.probes[human_run.probes.probe_epoch.eq("late")].copy()
                geometry_rows.extend([before.assign(human_training_checkpoint=0),
                                      after.assign(human_training_checkpoint=config.human_training_sequences)])
                for checkpoint, (model_state, policy_state) in sorted(human_run.snapshots.items()):
                    engine = new_engine(config, seed, max(config.endpoint_sequences, config.compositional_sequences))
                    engine.model.load_state_dict(deepcopy(model_state)); engine.policy.load_state_dict(deepcopy(policy_state))
                    checkpoint_run = EcologyRun(
                        condition, condition, "retention", seed, engine,
                        pd.DataFrame(), pd.DataFrame(), {},
                    )
                    rv1, rv2, rv3, rv4, _ = run_validation(config, registry, checkpoint_run, "retention")
                    x = rv2.distance_to_nearest_endpoint.to_numpy(float); y = rv2.transfer_quality.to_numpy(float)
                    corr = float(__import__("numpy").corrcoef(x, y)[0, 1]) if x.std() and y.std() else 0.
                    retention_rows.append({
                        "pretraining_condition": condition, "seed": seed,
                        "human_training_checkpoint": checkpoint,
                        "S_smooth": -float(rv1.prediction_deficit_over_worse_endpoint.mean()),
                        "S_gradient": -corr, "S_boundary": float(rv3.boundary_selectivity.mean()),
                        "S_comp": float(rv4.modified_position_accuracy.mean()),
                        "validation_probe_updates": 0,
                    })
        pd.concat(ablation_rows, ignore_index=True).to_csv(output / "mechanism_weight_subsystem_ablation.csv", index=False)
        geometry = pd.concat(geometry_rows, ignore_index=True)
        _mechanism_geometry(geometry).to_csv(output / "mechanism_hidden_geometry_before_after.csv", index=False)
        pd.DataFrame(retention_rows).to_csv(output / "mechanism_prior_retention.csv", index=False)
    return outputs, groups, improved


def _report(output, config, pilot_gates, confirm_gates, ranking, selected, human_groups, improved):
    if not pilot_gates.gate_passed.any():
        outcome = "Outcome A: none of P2–P5 passed the internal pilot gate; confirmatory and human stages were not run."
    elif not selected:
        outcome = "Structured methods reached confirmation, but none passed the confirmatory hard gates."
    elif improved:
        outcome = "Outcome G: structured pretraining improved first-probe transfer while passing familiar gates."
    else:
        outcome = "Outcome F: internally selected structured priors did not establish improved gated human transfer."
    lines = [
        "# Pretraining Ecology Comparison Report", "", "## Outcome", "", outcome, "",
        f"- Smoke mode: `{config.smoke}`", f"- Selected structured conditions: `{selected or 'none'}`",
        f"- Human groups: `{human_groups or 'not run'}`", f"- Mechanistic follow-ups triggered: `{improved or 'none'}`",
        "", "## Pilot hard gates", "", pilot_gates.to_markdown(index=False),
        "", "## Confirmatory hard gates", "",
        confirm_gates.to_markdown(index=False) if len(confirm_gates) else "Not reached.",
        "", "## Preregistered ranking", "",
        ranking.to_markdown(index=False) if len(ranking) else "No confirmatory-passing method to rank.",
        "", "## Firewalls and causal contrast", "",
        "Every structured condition was compared with its exact source-matched shuffled control at equal update count. Internal probes ran on disposable descendants and left parent checkpoints unchanged. Human transfer began only after confirmatory selection; no human probe outcome entered pretraining, gating, or ranking.",
        "", "The single P1 human group is the paired control for the top-ranked structured ecology. This preserves a four-group human design when two structured methods advance.",
        "", "## Reproducibility", "", f"- Analysis version: `{config.payload['analysis_version']}`",
        f"- Config SHA-256: `{config.config_sha256}`",
    ]
    (output / "PRETRAINING_ECOLOGY_COMPARISON_REPORT.md").write_text("\n".join(lines) + "\n")


def run_ecology_comparison(config: EcologyComparisonConfig):
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, output / "pretraining_ecology_config.json")
    registry = load_coordinate_registry(config.coordinate_registry, config.human["coordinate_array_assignment"])
    _progress(f"start; smoke={config.smoke}; output={output}")
    pilot = _run_phase(config, registry, "pilot", list(STRUCTURED), config.seeds, config.pilot_worlds, output)
    pilot_scores = condition_scores(pilot["v1"], pilot["v2"], pilot["v3"], pilot["v4"], pilot["training"])
    pilot_gates = gate_conditions(pilot_scores, "pilot", config.required_fraction("pilot"),
                                  float(config.payload["validation"]["familiar_nll_tolerance"]))
    if config.smoke and bool(config.payload["smoke"]["force_advance_for_integration_test"]):
        pilot_gates["gate_passed"] = True; pilot_gates["failure_reason"] = "smoke integration override"
    passing_pilot = pilot_gates[pilot_gates.gate_passed].structured_condition.tolist()
    _progress(f"pilot gates complete; passing={passing_pilot or 'none'}")
    confirm = None; confirm_scores = pd.DataFrame(); confirm_gates = pd.DataFrame(); ranking = pd.DataFrame(); selected = []
    if passing_pilot:
        confirm = _run_phase(config, registry, "confirmatory", passing_pilot, config.confirmatory_seeds,
                             config.confirmatory_worlds, output)
        confirm_scores = condition_scores(confirm["v1"], confirm["v2"], confirm["v3"], confirm["v4"], confirm["training"])
        confirm_gates = gate_conditions(confirm_scores, "confirmatory", config.required_fraction("confirmatory"),
                                        float(config.payload["validation"]["familiar_nll_tolerance"]))
        confirm_gates = confirm_gates[confirm_gates.structured_condition.isin(passing_pilot)].reset_index(drop=True)
        if config.smoke and bool(config.payload["smoke"]["force_advance_for_integration_test"]):
            confirm_gates["gate_passed"] = True; confirm_gates["failure_reason"] = "smoke integration override"
        ranking, selected = select_conditions(confirm_scores, confirm_gates)
        _progress(f"confirmatory selection finalized; selected={selected or 'none'}")
    all_phases = [pilot] + ([confirm] if confirm is not None else [])
    training = pd.concat([item["training"] for item in all_phases], ignore_index=True)
    worlds = pd.concat([item["worlds"] for item in all_phases], ignore_index=True)
    audits = pd.concat([item["audits"] for item in all_phases], ignore_index=True)
    validation_audits = pd.concat([item["validation_audits"] for item in all_phases], ignore_index=True)
    _paired_exposure_assertion(audits)
    if not validation_audits.parent_parameters_unchanged.all():
        raise AssertionError("Frozen validation audit failed.")
    v1 = pd.concat([item["v1"] for item in all_phases], ignore_index=True)
    v2 = pd.concat([item["v2"] for item in all_phases], ignore_index=True)
    v3 = pd.concat([item["v3"] for item in all_phases], ignore_index=True)
    v4 = pd.concat([item["v4"] for item in all_phases], ignore_index=True)
    scores = pd.concat([pilot_scores, confirm_scores], ignore_index=True)
    gates = pd.concat([pilot_gates, confirm_gates], ignore_index=True)
    if len(ranking):
        scores = scores.merge(ranking[["pretraining_condition", "selection_composite", "selection_rank"]],
                              on="pretraining_condition", how="left")
    tables = {
        "pretraining_world_manifest.csv": worlds, "pretraining_training_curves.csv": training,
        "pretraining_validation_v1.csv": v1, "pretraining_validation_v2.csv": v2,
        "pretraining_validation_v3.csv": v3, "pretraining_validation_v4.csv": v4,
        "pretraining_condition_scores.csv": scores, "pretraining_gate_results.csv": gates,
        "pretraining_training_audit.csv": audits, "pretraining_validation_audit.csv": validation_audits,
        "pretraining_selection_ranking.csv": ranking,
    }
    for name, frame in tables.items(): frame.to_csv(output / name, index=False)
    internal_figures(output, v1, v2, v3, v4, scores)
    human_outputs = {}; human_groups = []; improved = []
    if selected:
        human_outputs, human_groups, improved = _human_stage(config, registry, selected, confirm["runs"], output)
        tables.update(human_outputs)
    _report(output, config, pilot_gates, confirm_gates, ranking, selected, human_groups, improved)
    missing = [name for name in PRETRAINING_REQUIRED if not (output / name).exists()]
    if selected: missing += [name for name in HUMAN_REQUIRED if not (output / name).exists()]
    if missing: raise AssertionError(f"Required outputs missing: {missing}")
    manifest = {
        "analysis_version": config.payload["analysis_version"], "smoke": config.smoke,
        "pilot_passing": passing_pilot, "confirmatory_selected": selected,
        "human_transfer_reached": bool(selected), "human_groups": human_groups,
        "mechanistic_followups_triggered": improved, "config_sha256": config.config_sha256,
        "coordinate_registry_sha256": registry.source_sha256,
    }
    (output / "pretraining_ecology_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _progress("finished")
    return tables
