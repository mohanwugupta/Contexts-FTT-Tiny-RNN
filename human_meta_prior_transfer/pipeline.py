"""End-to-end locked-checkpoint target transfer with hard familiar gates."""

from __future__ import annotations

from datetime import datetime
import json
import math
import shutil

import numpy as np
import pandas as pd

from episodic_meta_learning_capacity.model import VariableElmanSRN, model_hash
from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .analysis import (add_human_profile_correspondence, canonicalize_familiar_profiles,
                       enforce_familiar_gate, familiar_gates, hidden_separation,
                       planned_statistics, state_space_tables, transfer_metrics)
from .checkpoints import load_locked_checkpoint, materialize_locked_checkpoints
from .config import HumanMetaTransferConfig
from .plotting import make_figures
from .retention import synthetic_prior_retention
from .training import ablated_state, train_transfer_group


REQUIRED = ("human_meta_transfer_config.json", "human_meta_transfer_group_registry.csv", "human_meta_transfer_checkpoint_hashes.json",
    "human_meta_transfer_training.csv", "human_meta_transfer_rt_training.csv", "human_meta_transfer_familiar_gates.csv",
    "human_meta_transfer_state_space.csv", "human_meta_transfer_hidden_separation.csv", "human_meta_transfer_probe_raw.csv",
    "human_meta_transfer_first_exposure.csv", "human_meta_transfer_mixed_metrics.csv", "human_meta_transfer_novel_metrics.csv",
    "human_meta_transfer_hidden_geometry.csv", "human_meta_transfer_planned_contrasts.csv", "human_meta_transfer_bootstrap.csv",
    "HUMAN_META_PRIOR_TRANSFER_REPORT.md")


def _with_schema(frame, columns):
    result = frame.copy()
    for column in columns:
        if column not in result: result[column] = np.nan
    extras = [column for column in result if column not in columns]
    return result[columns + extras]


def _progress(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] HUMAN-META-TRANSFER | {message}", flush=True)


def _group_registry(config):
    rows = []
    for specification in config.payload["groups"]:
        rows.append({**specification, "seed_count": len(config.seeds), "seeds": json.dumps(config.seeds),
                     "target_training_sequences": config.human_training_sequences,
                     "probe_checkpoints": json.dumps(config.probe_checkpoints),
                     "target_optimizer": config.payload["target_task"]["optimizer"],
                     "target_learning_rate": config.payload["target_task"]["learning_rate"],
                     "execution_objective": config.payload["target_task"]["rt_readout"],
                     "human_probe_based_selection": False})
    return pd.DataFrame(rows)


def _assert_training_contract(config, training, probes, audits, probe_audit):
    primary = training[training.group.isin(["G1", "G2", "G3", "G4"])]
    if primary.target_contract_sha256.nunique() != 1: raise AssertionError("G1-G4 target contracts differ.")
    schedules = primary.pivot_table(index=["seed", "training_sequence"], columns="group", values="context", aggfunc="first")
    if not schedules.apply(lambda row: row.nunique() == 1, axis=1).all(): raise AssertionError("G1-G4 A/B schedules differ.")
    if not audits.target_optimizer_empty_at_start.all() or audits.synthetic_optimizer_state_carried.any(): raise AssertionError("Synthetic optimizer state crossed the transfer boundary.")
    if not audits.execution_readout_freshly_initialized.all(): raise AssertionError("An RT readout was not freshly initialized.")
    if audits.mixed_training_updates.sum() or audits.novel_training_updates.sum(): raise AssertionError("A probe context entered target training.")
    if not probes.probe_exposure_index.eq(1).all() or not probes.primary_first_exposure.all(): raise AssertionError("Primary probes are not isolated first exposures.")
    if not probes.hidden_state_policy.eq("zero_reset").all(): raise AssertionError("Probe hidden state was not reset.")
    if not probe_audit.parameters_unchanged.all() or not probe_audit.optimizer_unchanged.all(): raise AssertionError("A frozen probe changed trainable state.")
    checkpoint_sources = audits.source_kind.eq("locked_synthetic_checkpoint")
    if not audits.loc[checkpoint_sources, "source_model_state_sha256"].eq(audits.loc[checkpoint_sources, "loaded_model_state_sha256"]).all():
        raise AssertionError("A target run did not start from its exact locked checkpoint.")


def _primary_success(config, metrics, gates):
    group_gate = gates[gates.row_type.eq("group")].set_index("group").gate_passed
    if not bool(group_gate.get("G1", False)) or not bool(group_gate.get("G2", False)): return False
    final = max(config.probe_checkpoints); source = metrics[(metrics.row_type.eq("seed_mean")) & metrics.checkpoint.eq(final) & metrics.primary_inference_eligible]
    values = source.groupby("group").between_endpoint_fraction.mean()
    return bool(values.get("G1", -np.inf) > values.get("G2", np.inf))


def _run_ablations(config, registry, primary_success):
    if not primary_success or not config.payload["weight_ablations"]["enabled_if_primary_success"]: return {}
    if config.smoke and not config.payload["smoke"]["run_weight_ablations"]: return {}
    runs, manifests = [], []
    for seed in config.seeds:
        payload, path = load_locked_checkpoint(config, "P2_meta", seed)
        for subsystem in config.payload["weight_ablations"]["subsystems"]:
            state, keys = ablated_state(config, payload["model_state_dict"], subsystem, seed)
            model = VariableElmanSRN(42, 128, 1.0); model.load_state_dict(state); state_hash = model_hash(model)
            specification = {"group": f"G1_{subsystem}_reset", "label": f"G1 {subsystem} reset", "source": f"P2_meta_{subsystem}_reset", "hidden_size": 128}
            override = (state, state_hash, str(path), "parent_hash_in_checkpoint_registry", f"single_subsystem_reset:{subsystem}")
            runs.append(train_transfer_group(config, registry, specification, seed, state_override=override))
            manifests.append({"seed": seed, "subsystem": subsystem, "reset_parameter_keys": json.dumps(keys), "ablated_initial_state_sha256": state_hash})
            _progress(f"conditional ablation {subsystem} seed {seed} complete")
    raw = pd.concat([run.probes for run in runs], ignore_index=True); canonical = canonicalize_familiar_profiles(raw); metrics, geometry = transfer_metrics(canonical)
    return {"human_meta_transfer_ablation_training.csv": pd.concat([run.training for run in runs], ignore_index=True),
            "human_meta_transfer_ablation_probe_raw.csv": raw,
            "human_meta_transfer_ablation_metrics.csv": metrics,
            "human_meta_transfer_ablation_hidden_geometry.csv": geometry,
            "human_meta_transfer_ablation_manifest.csv": pd.DataFrame(manifests)}


def _report(output, config, groups, gates, mixed, novel, contrasts, primary_success, ablations):
    group_gate = gates[gates.row_type.eq("group")][["group", "gate_passed", "failure_reason"]]
    final = max(config.probe_checkpoints); means = mixed[(mixed.row_type.eq("seed_mean")) & mixed.checkpoint.eq(final)].groupby("group").between_endpoint_fraction.mean()
    raw_mixed = means; gate_lookup = group_gate.set_index("group").gate_passed
    g1_g2 = raw_mixed.get("G1", -np.inf) > raw_mixed.get("G2", np.inf); g1_g3 = raw_mixed.get("G1", -np.inf) > raw_mixed.get("G3", np.inf)
    nmeans = novel[(novel.row_type.eq("seed_mean")) & novel.checkpoint.eq(final)].groupby("group")[["prediction_NLL_deficit", "RT_deficit"]].mean()
    broad = bool(set(["G1", "G2"]).issubset(nmeans.index) and (nmeans.loc["G1"] < nmeans.loc["G2"]).all())
    if config.smoke: outcome = "Engineering smoke run only; no scientific outcome is assigned."
    elif g1_g2 and not bool(gate_lookup.get("G1", False)): outcome = "Outcome C: raw Mixed transfer improved while G1 failed familiar learning; the intended account fails."
    elif primary_success and broad: outcome = "Outcomes A/E: G1 passed familiar gates, beat G2 on Mixed, and also improved Novel transfer."
    elif primary_success: outcome = "Outcomes A/F: G1 passed familiar gates and beat G2 on Mixed, without corresponding broad Novel improvement."
    elif bool(gate_lookup.get("G1", False)) and g1_g3 and not g1_g2: outcome = "Outcome B: G1 beat structured ordinary but not shuffled meta."
    else: outcome = "Outcome D: familiar behavior passed where indicated, but the primary G1 > G2 transfer criterion was not met."
    lines = ["# Human Meta-Prior Transfer Report", "", "## Outcome", "", outcome, "",
             f"- Smoke mode: `{config.smoke}`", f"- Primary-success trigger: `{primary_success}`",
             f"- Conditional weight ablations run: `{bool(ablations)}`", "", "## Locked groups", "", groups.to_markdown(index=False),
             "", "## Familiar gates", "", group_gate.to_markdown(index=False), "", "## Settled first-exposure Mixed fraction", "",
             means.rename("between_endpoint_fraction").reset_index().to_markdown(index=False) if len(means) else "No gate-eligible estimate.",
             "", "## Planned contrasts", "", contrasts.to_markdown(index=False) if len(contrasts) else "No admissible contrasts.",
             "", "## Firewall", "", "All checkpoint, optimizer, duration, and learning-rate choices were locked before target probes. Mixed and Novel received zero training updates. Empirical profile data were loaded only after every model run and gate computation, solely for unfitted profile correspondence.",
             "", "## Scaling rule", "", "H256/H512 were not run. They remain an optional post hoc robustness experiment only after the registered positive-H128 trigger and require independent synthetic calibration."]
    (output / "HUMAN_META_PRIOR_TRANSFER_REPORT.md").write_text("\n".join(lines) + "\n")


def run_human_meta_transfer(config: HumanMetaTransferConfig):
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True); shutil.copyfile(config.path, output / "human_meta_transfer_config.json")
    _progress(f"validating locked upstream result; smoke={config.smoke}")
    checkpoint_registry, upstream = materialize_locked_checkpoints(config, _progress)
    groups = _group_registry(config); registry = load_coordinate_registry(config.coordinate_registry, config.payload["coordinate_array_assignment"])
    runs = []
    for seed in config.seeds:
        for specification in config.payload["groups"]:
            runs.append(train_transfer_group(config, registry, specification, seed))
            _progress(f"target training {specification['group']} seed {seed} complete")
    training = pd.concat([run.training for run in runs], ignore_index=True); rt_training = pd.concat([run.rt_training for run in runs], ignore_index=True)
    probes = pd.concat([run.probes for run in runs], ignore_index=True); probe_audit = pd.concat([run.probe_audit for run in runs], ignore_index=True)
    audits = pd.DataFrame([run.audit for run in runs]); _assert_training_contract(config, training, probes, audits, probe_audit)
    _progress("training equality and frozen-probe audits passed")
    canonical = canonicalize_familiar_profiles(probes); metrics, geometry = transfer_metrics(canonical)
    separation = hidden_separation(canonical, audits); state_space, state_predictions, classifications = state_space_tables(training)
    gates = familiar_gates(config, training, rt_training, separation, classifications, audits, canonical)
    metrics = enforce_familiar_gate(metrics, gates); geometry = enforce_familiar_gate(geometry, gates)
    mixed = metrics[metrics.probe_context.eq("Mixed")].copy(); novel = metrics[metrics.probe_context.eq("Novel")].copy()
    mixed, human_profile = add_human_profile_correspondence(mixed, config.human_profile_source)
    contrasts, bootstrap = planned_statistics(config, mixed, novel, geometry)
    contrasts = _with_schema(contrasts, ["record_type", "metric", "contrast", "contrast_role", "checkpoint", "n_paired_seeds", "estimate", "standard_error", "p_value", "standardized_effect"])
    bootstrap = _with_schema(bootstrap, ["metric", "checkpoint", "contrast", "contrast_role", "n_paired_seeds", "mean_difference", "CI95_lower", "CI95_upper", "standardized_effect", "bootstrap_draws", "bootstrap_unit"])
    retention_raw, retention = synthetic_prior_retention(config, runs, _progress)
    success = _primary_success(config, mixed, gates); ablations = _run_ablations(config, registry, success)
    tables = {"human_meta_transfer_group_registry.csv": groups, "human_meta_transfer_training.csv": training,
        "human_meta_transfer_rt_training.csv": rt_training, "human_meta_transfer_familiar_gates.csv": gates,
        "human_meta_transfer_state_space.csv": state_space, "human_meta_transfer_state_space_predictions.csv": state_predictions,
        "human_meta_transfer_hidden_separation.csv": separation, "human_meta_transfer_probe_raw.csv": probes,
        "human_meta_transfer_probe_canonicalized.csv": canonical, "human_meta_transfer_first_exposure.csv": probes[probes.probe_exposure_index.eq(1)].copy(),
        "human_meta_transfer_mixed_metrics.csv": mixed, "human_meta_transfer_novel_metrics.csv": novel,
        "human_meta_transfer_hidden_geometry.csv": geometry, "human_meta_transfer_planned_contrasts.csv": contrasts,
        "human_meta_transfer_bootstrap.csv": bootstrap, "human_meta_transfer_training_audit.csv": audits,
        "human_meta_transfer_probe_audit.csv": probe_audit, "human_meta_transfer_synthetic_retention_raw.csv": retention_raw,
        "human_meta_transfer_synthetic_retention.csv": retention, "human_meta_transfer_human_profile_reference.csv": human_profile,
        **ablations}
    for name, frame in tables.items(): frame.to_csv(output / name, index=False)
    hashes = {"analysis_version": config.payload["analysis_version"], "transfer_config_sha256": config.config_sha256,
              "engineering_smoke": config.smoke, "upstream_evidence": upstream,
              "checkpoints": checkpoint_registry.to_dict("records"),
              "target_initial_states": audits[["group", "seed", "source_kind", "source_model_state_sha256", "loaded_model_state_sha256"]].to_dict("records")}
    (output / "human_meta_transfer_checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    make_figures(output, training, canonical, mixed, novel, geometry, separation, state_space, retention)
    _report(output, config, groups, gates, mixed, novel, contrasts, success, ablations)
    missing = [name for name in REQUIRED if not (output / name).exists()]
    if missing: raise AssertionError(f"Missing registered outputs: {missing}")
    _progress("finished"); return tables
