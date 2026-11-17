"""Capacity-complete pilot, synthetic gates, confirmation, and human firewall."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .analysis import (capacity_table, hidden_differentiation, human_gates,
                       human_probe_tables, human_training_summary,
                       state_space_tables, synthetic_gate, synthetic_scores)
from .config import CONDITIONS, MetaCapacityConfig
from .human import train_human
from .model import VariableElmanSRN
from .plotting import human_figures, synthetic_figures
from .training import TrainingRun, train_condition
from .validation import validate_checkpoint


SYNTHETIC_REQUIRED = ("episodic_meta_config.json", "meta_capacity_grid.json", "synthetic_family_generator_config.json",
    "meta_training_curves.csv", "ordinary_training_curves.csv", "meta_episode_manifest.csv", "meta_support_query_manifest.csv",
    "synthetic_v1_intermediate.csv", "synthetic_v2_similarity_gradient.csv", "synthetic_v3_asymmetric_queries.csv",
    "synthetic_v4_copy_retention.csv", "synthetic_v5_family_robustness.csv", "synthetic_prior_gate.csv",
    "capacity_comparison.csv", "EPISODIC_META_LEARNING_CAPACITY_REPORT.md")
HUMAN_REQUIRED = ("meta_human_training.csv", "meta_human_state_space.csv", "meta_human_hidden_differentiation.csv",
    "meta_human_probe_metrics.csv", "meta_human_checkpoint_metrics.csv", "meta_human_interpolation.csv", "meta_human_hidden_corridor.csv")


def _progress(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] EPISODIC-META | {message}", flush=True)


def _checkpoint(output: Path, run: TrainingRun):
    target = output / "meta_checkpoints"; target.mkdir(parents=True, exist_ok=True)
    torch.save({"phase": run.phase, "training_condition": run.condition, "hidden_size": run.hidden_size,
                "seed": run.seed, "model_state_dict": run.model.state_dict(), "audit": run.audit},
               target / f"{run.phase}_{run.condition}_H{run.hidden_size}_seed_{run.seed}.pt")


def _phase(config, registry, phase, widths, seeds, episodes, output):
    runs = {}; curves = []; families = []; support_query = []; audits = []
    validation = [[] for _ in range(5)]; validation_audits = []
    _progress(f"{phase}: widths={list(widths)}, seeds={list(seeds)}, episodes={episodes}")
    for width in widths:
        for seed in seeds:
            for condition in CONDITIONS:
                run = train_condition(config, registry, condition, width, phase, seed, episodes, progress=_progress)
                runs[(condition, width, seed)] = run; curves.append(run.curves); families.append(run.families)
                support_query.append(run.support_query); audits.append(run.audit); _checkpoint(output, run)
                values = validate_checkpoint(config, registry, run, phase, config.validation_families(phase if phase != "smoke" else "pilot"))
                for target, frame in zip(validation, values[:5]): target.append(frame)
                validation_audits.append(values[5])
    curve = pd.concat(curves, ignore_index=True); audit = pd.DataFrame(audits)
    # T1--T3 are hard runtime assertions, not merely diagnostics.
    roles = pd.concat(support_query, ignore_index=True)
    if roles[roles.role.str.contains("query")].used_for_inner_adaptation.any(): raise AssertionError("A query entered inner adaptation.")
    for (width, seed), one in curve.groupby(["hidden_size", "seed"]):
        hashes = one.pivot(index="episode", columns="training_condition", values="family_sha256")
        if not hashes.apply(lambda row: row.nunique() == 1, axis=1).all(): raise AssertionError("Paired families differ across conditions.")
    if not audit.query_gradient_reached_meta_parameters.all(): raise AssertionError("An outer query gradient failed to reach model parameters.")
    return {"runs": runs, "curves": curve, "families": pd.concat(families, ignore_index=True),
            "support_query": roles, "audits": audit, "validation_audits": pd.DataFrame(validation_audits),
            **{f"v{index + 1}": pd.concat(parts, ignore_index=True) for index, parts in enumerate(validation)}}


def _mixed_model(frame, outcome, formula, family):
    if frame.seed.nunique() < 2:
        return pd.DataFrame([{"analysis_family": family, "outcome": outcome, "term": "not_estimable",
                              "estimate": np.nan, "standard_error": np.nan, "p_value": np.nan,
                              "model": formula + " + (1|seed)", "error": "requires at least two seeds"}])
    try:
        import statsmodels.formula.api as smf
        fit = smf.mixedlm(formula, frame, groups=frame["seed"]).fit(reml=False, method="lbfgs")
        return pd.DataFrame({"analysis_family": family, "outcome": outcome, "term": fit.params.index,
                             "estimate": fit.params.values, "standard_error": fit.bse.reindex(fit.params.index).values,
                             "p_value": fit.pvalues.reindex(fit.params.index).values, "model": formula + " + (1|seed)"})
    except Exception as error:
        return pd.DataFrame([{"analysis_family": family, "outcome": outcome, "term": "model_failure",
                              "estimate": np.nan, "standard_error": np.nan, "p_value": np.nan,
                              "model": formula + " + (1|seed)", "error": str(error)}])


def _synthetic_statistics(v5):
    source = v5.copy(); source["log2_hidden_size"] = np.log2(source.hidden_size)
    parts = [_mixed_model(source, outcome, f"{outcome} ~ C(training_condition) * log2_hidden_size", "synthetic")
             for outcome in ("mean_query_NLL", "similarity_gradient_score", "mean_copy_accuracy")]
    return pd.concat(parts, ignore_index=True)


def _bootstrap_contrasts(frame, index_columns, condition_column, value_column, contrasts, draws, seed, family):
    rng = np.random.default_rng(seed); rows = []
    for key, one in frame.groupby(index_columns, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        pivot = one.pivot_table(index="seed", columns=condition_column, values=value_column, aggfunc="mean")
        for first, second in contrasts:
            if first not in pivot or second not in pivot: continue
            delta = (pivot[first] - pivot[second]).dropna().to_numpy(float)
            if not len(delta): continue
            samples = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(1)
            row = dict(zip(index_columns, key)) | {"analysis_family": family, "metric": value_column,
                "contrast": f"{first} - {second}", "n_paired_seeds": len(delta), "mean_difference": float(delta.mean()),
                "bootstrap_CI_low": float(np.quantile(samples, .025)), "bootstrap_CI_high": float(np.quantile(samples, .975)),
                "standardized_effect": float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


def _human_statistics(metrics):
    source = metrics[metrics.row_type.eq("seed_mean")].copy(); parts = []
    for context in ("Mixed", "Novel"):
        one = source[source.probe_context.eq(context)]
        for outcome in ("NLL_deficit_over_worse_endpoint", "RT_deficit_over_worse_endpoint"):
            parts.append(_mixed_model(one, outcome, f"{outcome} ~ C(model_variant) * C(checkpoint)", f"human_{context}"))
    return pd.concat(parts, ignore_index=True)


def _human_phase(config, registry, selected_widths, source_runs, output):
    smallest = min(selected_widths); specifications = [("P0", smallest, None),
        ("P2_ordinary", smallest, "P2_ordinary"), ("P1_shuffled_meta", smallest, "P1_shuffled_meta")]
    specifications += [("P2_meta", width, "P2_meta") for width in selected_widths]
    runs = []; lookup = {}
    for seed in config.confirmatory_seeds:
        for label, width, source_condition in specifications:
            source = source_runs[(source_condition, width, seed)] if source_condition is not None else None
            result = train_human(config, registry, label, width, seed, source)
            runs.append(result); lookup[(label, width, seed)] = result
            _progress(f"human {label} H{width} seed {seed}")
    training = pd.concat([run.training for run in runs], ignore_index=True); probes = pd.concat([run.probes for run in runs], ignore_index=True)
    audits = pd.DataFrame([run.audit for run in runs]); summary = human_training_summary(training)
    fits, classes, predictions = state_space_tables(training); hidden = hidden_differentiation(probes)
    metrics, interpolation, corridor = human_probe_tables(probes)
    gates = human_gates(summary, hidden, classes, audits, .5 if config.smoke else float(config.payload["human_gates"]["seed_fraction"]),
                        float(config.payload["human_gates"]["minimum_P0_differentiation_ratio"]), float(config.payload["human_gates"]["minimum_RT_sawtooth"]))
    checkpoint_metrics = metrics[metrics.row_type.eq("seed_mean")].copy()
    outputs = {"meta_human_training.csv": training, "meta_human_training_summary.csv": summary,
        "meta_human_training_audit.csv": audits, "meta_human_state_space.csv": fits,
        "meta_human_state_space_classification.csv": classes, "meta_human_state_space_predictions.csv": predictions,
        "meta_human_hidden_differentiation.csv": hidden, "meta_human_probe_transition_level.csv": probes,
        "meta_human_probe_metrics.csv": metrics, "meta_human_checkpoint_metrics.csv": checkpoint_metrics,
        "meta_human_interpolation.csv": interpolation, "meta_human_hidden_corridor.csv": corridor,
        "meta_human_familiar_gates.csv": gates, "meta_human_statistical_models.csv": _human_statistics(metrics)}
    draws = 200 if config.smoke else int(config.payload["bootstrap"]["draws"])
    outputs["meta_human_paired_contrasts.csv"] = _bootstrap_contrasts(
        checkpoint_metrics, ["hidden_size", "checkpoint", "probe_context"], "training_condition",
        "NLL_deficit_over_worse_endpoint", [("P2_meta", "P1_shuffled_meta"), ("P2_meta", "P2_ordinary")],
        draws, int(config.payload["bootstrap"]["seed"]), "human")
    for name, frame in outputs.items(): frame.to_csv(output / name, index=False)
    human_figures(output, training, metrics, interpolation, hidden)
    early = interpolation[(interpolation.row_type.eq("seed_mean")) & interpolation.probe_context.eq("Mixed") & interpolation.checkpoint.eq(min(config.probe_checkpoints))]
    values = early.groupby("model_variant").between_endpoint_fraction.mean(); meta_name = f"P2_meta_H{smallest}"; control_name = f"P1_shuffled_meta_H{smallest}"
    gate_lookup = gates.set_index("model_variant").gate_passed
    improved = bool(gate_lookup.get(meta_name, False) and values.get(meta_name, -np.inf) > values.get(control_name, np.inf))
    if improved:
        retention, ablations = [], []
        for seed in config.confirmatory_seeds:
            source = source_runs[("P2_meta", smallest, seed)]; human_run = lookup[("P2_meta", smallest, seed)]
            for checkpoint, state in sorted(human_run.snapshots.items()):
                model = VariableElmanSRN(42, smallest, float(config.payload["initialization_scale"])); model.load_state_dict(deepcopy(state))
                temporary = TrainingRun("P2_meta", smallest, "retention", seed, model, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
                rv = validate_checkpoint(config, registry, temporary, "retention", config.validation_families("pilot"))
                retention.append(rv[0].assign(human_checkpoint=checkpoint))
            torch.manual_seed(seed); random = VariableElmanSRN(42, smallest, float(config.payload["initialization_scale"])); random_state = random.state_dict()
            for label, names in {"input": ["recurrent.weight_ih_l0"], "recurrent": ["recurrent.weight_hh_l0"],
                                 "output": ["sequence_head.weight", "sequence_head.bias"]}.items():
                state = deepcopy(source.model.state_dict())
                for name in names: state[name] = random_state[name].detach().clone()
                model = VariableElmanSRN(42, smallest, float(config.payload["initialization_scale"])); model.load_state_dict(state)
                temporary = TrainingRun("P2_meta", smallest, "ablation", seed, model, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})
                ablations.append(train_human(config, registry, f"P2_meta_{label}_reinitialized", smallest, seed, temporary).probes.assign(ablation=label))
        pd.concat(retention, ignore_index=True).to_csv(output / "mechanism_meta_prior_retention.csv", index=False)
        pd.concat(ablations, ignore_index=True).to_csv(output / "mechanism_weight_subsystem_ablation.csv", index=False)
        hidden[hidden.model_variant.isin([f"P2_meta_H{smallest}", f"P2_ordinary_H{smallest}", f"P1_shuffled_meta_H{smallest}"])].to_csv(output / "mechanism_hidden_geometry.csv", index=False)
    return outputs, improved


def _report(output, config, pilot_gates, confirm_gates, selected, human_reached, improved):
    if not pilot_gates.gate_passed.any(): outcome = "No capacity passed the synthetic pilot gate; the human firewall remained closed."
    elif not selected: outcome = "Pilot candidates reached confirmation, but none passed confirmatory synthetic gates."
    elif improved: outcome = "Target outcome: structured episodic pretraining improved early gated human transfer."
    else: outcome = "Synthetic prior candidates advanced, but gated early human transfer did not improve."
    interpreted = confirm_gates if len(confirm_gates) else pilot_gates
    if config.smoke:
        capacity_interpretation = "Engineering smoke only; capacity outcomes are not interpreted."
    elif interpreted.gate_passed.all():
        capacity_interpretation = "Outcome B candidate: all tested widths passed; the objective, rather than capacity alone, is the leading explanation."
    elif 8 not in interpreted[interpreted.gate_passed].hidden_size.tolist() and interpreted.gate_passed.any():
        capacity_interpretation = "Outcome A candidate: H8 failed while at least one larger Elman SRN passed."
    elif not interpreted.gate_passed.any():
        capacity_interpretation = "All widths failed the full synthetic prior gate; episodic training is insufficient under this architecture/ecology."
    else:
        capacity_interpretation = "Capacity was not monotonically beneficial; selection uses the smallest robustly passing width."
    lines = ["# Episodic Meta-Learning Capacity Report", "", "## Outcome", "", outcome, "",
             f"- Smoke mode: `{config.smoke}`", f"- Selected capacities: `{selected or 'none'}`", f"- Human stage reached: `{human_reached}`",
             f"- Capacity interpretation: {capacity_interpretation}",
             "", "## Pilot synthetic gates", "", pilot_gates.to_markdown(index=False), "", "## Confirmatory gates", "",
             confirm_gates.to_markdown(index=False) if len(confirm_gates) else "Not reached.", "", "## Firewalls", "",
             "Synthetic optimization contained prediction loss only. The execution readout did not exist during pretraining and was freshly initialized for human transfer. Validation queries made zero updates, and human files were created only after synthetic selection.",
             "", "## Statistical outputs", "", "Mixed-effects coefficient tables and paired-seed bootstrap differences with 95% intervals and standardized effects are saved alongside the descriptive tables.",
             "", "## Reproducibility", "", f"- Version: `{config.payload['analysis_version']}`", f"- Config hashes: `{config.hashes}`"]
    (output / "EPISODIC_META_LEARNING_CAPACITY_REPORT.md").write_text("\n".join(lines) + "\n")


def run_meta_capacity_experiment(config: MetaCapacityConfig):
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True)
    for source, name in ((config.path, "episodic_meta_config.json"), (config.grid_path, "meta_capacity_grid.json"),
                         (config.generator_path, "synthetic_family_generator_config.json")): shutil.copyfile(source, output / name)
    registry = load_coordinate_registry(config.coordinate_registry, config.payload["human_transfer"]["coordinate_array_assignment"])
    phase_name = "smoke" if config.smoke else "pilot"
    pilot = _phase(config, registry, phase_name, config.hidden_sizes, config.pilot_seeds, config.pilot_episodes, output)
    pilot_scores = synthetic_scores(pilot["v1"], pilot["v2"], pilot["v3"], pilot["v4"])
    pilot_gates = synthetic_gate(pilot_scores, phase_name, config.hidden_sizes, config.required_fraction("pilot"), config.payload["gate"])
    if config.smoke and config.payload["smoke"]["force_advance_for_integration_test"]: pilot_gates["gate_passed"] = True; pilot_gates["failure_reason"] = "smoke integration override"
    passing = pilot_gates[pilot_gates.gate_passed].hidden_size.astype(int).tolist(); confirm = None; confirm_scores = pd.DataFrame(); confirm_gates = pd.DataFrame(); selected = []
    if passing and not config.smoke:
        confirm = _phase(config, registry, "confirmatory", passing, config.confirmatory_seeds, config.confirmatory_episodes, output)
        confirm_scores = synthetic_scores(confirm["v1"], confirm["v2"], confirm["v3"], confirm["v4"])
        confirm_gates = synthetic_gate(confirm_scores, "confirmatory", passing, config.required_fraction("confirmatory"), config.payload["gate"])
        confirmed = sorted(confirm_gates[confirm_gates.gate_passed].hidden_size.astype(int).tolist()); selected = confirmed[:1] + confirmed[1:2]
    elif passing and config.payload["smoke"]["force_advance_for_integration_test"]:
        selected = sorted(passing)[:2]
    phases = [pilot] + ([confirm] if confirm else [])
    curves = pd.concat([x["curves"] for x in phases], ignore_index=True); audits = pd.concat([x["audits"] for x in phases], ignore_index=True)
    validation_audits = pd.concat([x["validation_audits"] for x in phases], ignore_index=True)
    if not validation_audits.parent_parameters_unchanged.all(): raise AssertionError("Validation parent hash failed.")
    v = {name: pd.concat([x[name] for x in phases], ignore_index=True) for name in ("v1", "v2", "v3", "v4", "v5")}
    scores = pd.concat([pilot_scores, confirm_scores], ignore_index=True); gates = pd.concat([pilot_gates, confirm_gates], ignore_index=True)
    for audit_frame, curve_frame in ((audits, curves),):
        exposure = audit_frame.pivot_table(index=["phase", "hidden_size", "seed"], columns="training_condition", values=["external_sequences", "persistent_optimizer_updates"])
        for metric in ("external_sequences", "persistent_optimizer_updates"):
            if not exposure[metric].nunique(axis=1).eq(1).all(): raise AssertionError("Condition exposure mismatch.")
    capacity = capacity_table(scores, audits, gates)
    tables = {"meta_training_curves.csv": curves[curves.training_condition.str.endswith("meta")],
              "ordinary_training_curves.csv": curves[curves.training_condition.eq("P2_ordinary")],
              "meta_episode_manifest.csv": pd.concat([x["families"] for x in phases], ignore_index=True).drop_duplicates(["phase", "hidden_size", "seed", "episode"]),
              "meta_support_query_manifest.csv": pd.concat([x["support_query"] for x in phases], ignore_index=True),
              "synthetic_v1_intermediate.csv": v["v1"], "synthetic_v2_similarity_gradient.csv": v["v2"],
              "synthetic_v3_asymmetric_queries.csv": v["v3"], "synthetic_v4_copy_retention.csv": v["v4"],
              "synthetic_v5_family_robustness.csv": v["v5"], "synthetic_prior_gate.csv": gates,
              "capacity_comparison.csv": capacity, "meta_training_audit.csv": audits,
              "synthetic_validation_audit.csv": validation_audits, "synthetic_statistical_models.csv": _synthetic_statistics(v["v5"])}
    v5_seed = v["v5"].groupby(["phase", "hidden_size", "seed", "training_condition"], as_index=False).mean_query_NLL.mean()
    tables["synthetic_paired_contrasts.csv"] = _bootstrap_contrasts(
        v5_seed, ["phase", "hidden_size"], "training_condition", "mean_query_NLL",
        [("P2_meta", "P2_ordinary"), ("P2_meta", "P1_shuffled_meta")],
        200 if config.smoke else int(config.payload["bootstrap"]["draws"]),
        int(config.payload["bootstrap"]["seed"]), "synthetic")
    for name, frame in tables.items(): frame.to_csv(output / name, index=False)
    synthetic_figures(output, scores, v["v2"], v["v3"], v["v4"], gates)
    human_outputs = {}; improved = False
    if selected:
        sources = confirm["runs"] if confirm else pilot["runs"]
        human_outputs, improved = _human_phase(config, registry, selected, sources, output); tables.update(human_outputs)
    _report(output, config, pilot_gates, confirm_gates, selected, bool(selected), improved)
    missing = [name for name in SYNTHETIC_REQUIRED if not (output / name).exists()]
    if selected: missing += [name for name in HUMAN_REQUIRED if not (output / name).exists()]
    if missing: raise AssertionError(f"Required outputs missing: {missing}")
    manifest = {"analysis_version": config.payload["analysis_version"], "smoke": config.smoke,
                "pilot_passing_capacities": passing, "selected_capacities": selected, "human_reached": bool(selected),
                "mechanisms_triggered": improved, "config_hashes": config.hashes, "coordinate_registry_sha256": registry.source_sha256}
    (output / "episodic_meta_learning_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _progress("finished")
    return tables
