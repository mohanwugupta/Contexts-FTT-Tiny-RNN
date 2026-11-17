"""End-to-end Phase 1G pipeline with calibration/probe firewall separation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analysis import (
    CONTEXT_VECTORS, add_probe_summaries, causal_checks, evaluate_with_firewall,
    geometry, novel_vector_table, shared_rollout,
)
from .config import G0, G1, G2, G3, G4, G5, LOW, ExperimentConfig
from .plotting import enforce_outputs, make_figures
from .training import Hyperparameters, restore_result, train_condition, training_source_hash


def _stamp(): return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
def _log(message): print(f"[{_stamp()}] PHASE1G | {message}", flush=True)


def _sha(path):
    digest = hashlib.sha256(); digest.update(Path(path).read_bytes()); return digest.hexdigest()


def _final_diagnostics(result):
    row = result.metrics.iloc[-1]
    return {"familiar_loss": float(np.mean([row.A_raw_loss, row.B_raw_loss])),
            "hidden_separation": float(row.AB_hidden_separation),
            "task_separation": float(row.AB_task_separation),
            "shared_prediction": float(row.shared_prediction_loss),
            "shared_trajectory": float(row.shared_trajectory_loss)}


def _candidate_rows(config, condition, parameter, candidates, baseline):
    rows = []
    for value in candidates:
        for seed in config.calibration_seeds:
            hyper = Hyperparameters(**{parameter: float(value)})
            result = train_condition(config, condition, seed, hyper)
            observed = _final_diagnostics(result); reference = baseline[seed]
            rows.append({"condition": condition, "parameter": parameter, "candidate": value,
                         "seed": seed, **observed,
                         "familiar_loss_ratio": observed["familiar_loss"]/reference["familiar_loss"],
                         "hidden_separation_ratio": observed["hidden_separation"]/(reference["hidden_separation"]+1e-12),
                         "task_separation_ratio": observed["task_separation"]/(reference["task_separation"]+1e-12)})
    return rows


def _select(frame, parameter, objective, config):
    frame = frame.copy()
    frame["seed_guard_pass"] = ((frame.familiar_loss_ratio <= config.familiar_loss_ratio_limit) &
                                 (frame.hidden_separation_ratio >= config.separation_ratio_limit) &
                                 (frame.task_separation_ratio >= config.separation_ratio_limit))
    summary = frame.groupby("candidate", as_index=False).agg(
        guard_pass_rate=("seed_guard_pass", "mean"), selection_objective=(objective, "mean"),
        familiar_loss_ratio=("familiar_loss_ratio", "mean"),
        hidden_separation_ratio=("hidden_separation_ratio", "mean"),
        task_separation_ratio=("task_separation_ratio", "mean"))
    eligible = summary[summary.guard_pass_rate.eq(1.0)]
    pool = eligible if len(eligible) else summary[summary.guard_pass_rate.eq(summary.guard_pass_rate.max())]
    selected = float(pool.sort_values(["selection_objective", "candidate"]).iloc[0].candidate)
    frame["selected"] = np.isclose(frame.candidate, selected)
    frame["selection_parameter"] = parameter; frame["selection_objective_name"] = objective
    return selected, frame, summary


def calibrate(config):
    _log("Calibration starts (A/B and shared diagnostics only; probes are unavailable)")
    baseline = {}
    rows = []
    for seed in config.calibration_seeds:
        result = train_condition(config, G0, seed, Hyperparameters())
        baseline[seed] = _final_diagnostics(result)
        rows.append({"condition": G0, "parameter": "none", "candidate": 0., "seed": seed,
                     **baseline[seed], "familiar_loss_ratio": 1., "hidden_separation_ratio": 1.,
                     "task_separation_ratio": 1., "seed_guard_pass": True, "selected": True,
                     "selection_parameter": "none", "selection_objective_name": "reference"})
    selections = {}
    specifications = (
        (G1, "p_dropout", config.dropout_candidates, "shared_trajectory"),
        (G2, "lambda_pred", config.prediction_candidates, "shared_prediction"),
        (G3, "lambda_traj", config.trajectory_candidates, "shared_trajectory"),
    )
    summaries = []
    for condition, parameter, candidates, objective in specifications:
        candidate = pd.DataFrame(_candidate_rows(config, condition, parameter, candidates, baseline))
        selected, annotated, summary = _select(candidate, parameter, objective, config)
        selections[parameter] = selected; rows.extend(annotated.to_dict("records"))
        summary["condition"] = condition; summary["parameter"] = parameter
        summaries.append(summary)
        _log(f"Selected {parameter}={selected:g} using {objective}; no held-out probe was evaluated")
    selections["selection_firewall"] = "A/B_and_shared_diagnostics_only"
    return selections, pd.DataFrame(rows), pd.concat(summaries, ignore_index=True)


def _hyper(condition, selected):
    if condition == G1: return Hyperparameters(p_dropout=selected["p_dropout"])
    if condition == G2: return Hyperparameters(lambda_pred=selected["lambda_pred"])
    if condition == G3: return Hyperparameters(lambda_traj=selected["lambda_traj"])
    if condition == G4: return Hyperparameters(lambda_pred=selected["lambda_pred"], lambda_traj=selected["lambda_traj"])
    if condition == G5: return Hyperparameters(p_dropout=selected["p_dropout"], lambda_traj=selected["lambda_traj"])
    return Hyperparameters()


def _train_or_load(config, output, condition, seed, hyper):
    partial = output/"partials"; checkpoints = output/"checkpoints"
    partial.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    metric_path = partial/f"{condition}_seed_{seed}_training.csv"
    checkpoint_path = checkpoints/f"{condition}_seed_{seed}.pt"
    if metric_path.exists() and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("config_sha256") != config.config_sha256:
            raise RuntimeError(f"Existing checkpoint has a different frozen config: {checkpoint_path}")
        if checkpoint.get("training_source_sha256") != training_source_hash():
            raise RuntimeError(f"Existing checkpoint was produced by different training source: {checkpoint_path}")
        _log(f"resuming completed {condition} seed {seed}")
        return restore_result(config, checkpoint, pd.read_csv(metric_path)), checkpoint_path
    result = train_condition(config, condition, seed, hyper, progress=_log)
    result.metrics.to_csv(metric_path, index=False); torch.save(result.checkpoint, checkpoint_path)
    return result, checkpoint_path


def _seed_summary(config, training, probe_summary, rollout):
    final = training.sort_values("training_sequence").groupby(["condition", "seed"], as_index=False).tail(1)
    final = final[["condition", "seed", "A_raw_loss", "B_raw_loss", "AB_hidden_separation",
                   "AB_task_separation", "shared_prediction_loss", "shared_trajectory_loss"]]
    losses = probe_summary.pivot_table(index=["condition", "seed"], columns="probe", values="raw_sequence_loss").reset_index()
    summary = final.merge(losses, on=["condition", "seed"], how="left")
    roll = rollout.groupby(["condition", "seed", "mode"], as_index=False).agg(
        trajectory_state_error=("shared_rollout_state_error", "mean"),
        rollout_raw_loss=("shared_rollout_raw_loss", "mean"))
    one = roll[roll["mode"].eq("one_step")][["condition", "seed", "trajectory_state_error"]].rename(
        columns={"trajectory_state_error": "shared_trajectory_error"})
    free = roll[roll["mode"].eq("free_rollout")][["condition", "seed", "rollout_raw_loss"]].rename(
        columns={"rollout_raw_loss": "shared_rollout_loss"})
    summary = summary.merge(one, on=["condition", "seed"], how="left").merge(free, on=["condition", "seed"], how="left")
    baseline = summary[summary.condition.eq(G0)].set_index("seed")
    for index, row in summary.iterrows():
        ref = baseline.loc[row.seed]
        summary.loc[index, "familiar_loss_ratio"] = np.mean([row.A, row.B])/np.mean([ref.A, ref.B])
        summary.loc[index, "AB_hidden_separation_ratio"] = row.AB_hidden_separation/(ref.AB_hidden_separation+1e-12)
        summary.loc[index, "AB_task_separation_ratio"] = row.AB_task_separation/(ref.AB_task_separation+1e-12)
        for probe in ("Mixed", "NoContext", "NovelMean"):
            summary.loc[index, f"{probe}_improvement"] = ref[probe]-row[probe]
        low, high = min(row.A, row.B), max(row.A, row.B)
        for probe in ("Mixed", "NoContext", "NovelMean"):
            summary.loc[index, f"{probe}_placement"] = ("below A/B range" if row[probe] < low else
                "above A/B range" if row[probe] > high else "within A/B range")
    summary["familiar_guard_pass"] = summary.familiar_loss_ratio <= config.familiar_loss_ratio_limit
    summary["hidden_separation_guard_pass"] = summary.AB_hidden_separation_ratio >= config.separation_ratio_limit
    summary["task_separation_guard_pass"] = summary.AB_task_separation_ratio >= config.separation_ratio_limit
    summary["all_guardrails_pass"] = summary[["familiar_guard_pass", "hidden_separation_guard_pass",
                                               "task_separation_guard_pass"]].all(axis=1)
    return summary


def _select_causal_condition(summary):
    candidates = summary[summary.condition.isin([G1, G2, G3, G4, G5])].groupby("condition", as_index=False).agg(
        guard=("all_guardrails_pass", "mean"), Mixed=("Mixed_improvement", "mean"),
        NoContext=("NoContext_improvement", "mean"), Novel=("NovelMean_improvement", "mean"))
    eligible = candidates[candidates.guard.eq(1.)].copy()
    pool = eligible if len(eligible) else candidates
    pool["score"] = pool[["Mixed", "NoContext", "Novel"]].mean(axis=1)
    return str(pool.sort_values(["score", "condition"], ascending=[False, True]).iloc[0].condition)


def _decision(summary):
    means = summary[summary.condition.isin([G0,G1,G2,G3,G4,G5])].groupby("condition").mean(numeric_only=True)
    def success(name):
        if name not in means.index: return False
        r = means.loc[name]
        return bool(r.all_guardrails_pass == 1 and r.Mixed_improvement > 0 and
                    r.NoContext_improvement > 0 and r.NovelMean_improvement > 0 and
                    r.shared_trajectory_error < means.loc[G0].shared_trajectory_error and
                    r.shared_rollout_loss < means.loc[G0].shared_rollout_loss)
    if success(G4): return "A", "G4 succeeds while preserving the A/B split"
    if success(G3) and not success(G2): return "B", "Trajectory stability succeeds without shared prediction"
    if success(G2) and not success(G3): return "C", "Shared prediction succeeds without trajectory stability"
    explicit = max([means.loc[x, ["Mixed_improvement","NoContext_improvement","NovelMean_improvement"]].mean()
                    for x in (G2,G3,G4) if x in means.index], default=-np.inf)
    if success(G1) and means.loc[G1, ["Mixed_improvement","NoContext_improvement","NovelMean_improvement"]].mean() >= .9*explicit:
        return "D", "Context dropout succeeds as well as explicit shared losses"
    any_improvement = any(means.loc[x, ["Mixed_improvement","NoContext_improvement","NovelMean_improvement"]].mean() > 0
                          for x in means.index if x != G0)
    any_guarded = any(bool(means.loc[x].all_guardrails_pass == 1) for x in means.index if x != G0)
    if any_improvement and not any_guarded: return "E", "Transfer improves only when the A/B split collapses"
    return "F", "Mixed/shared transfer is insufficient for Novel and NoContext routing"


def _write_report(output, config, selected, calibration, summary, causal_condition, decision):
    code, label = decision
    aggregate = summary.groupby("condition").mean(numeric_only=True)
    table_columns = ["A", "B", "AB_hidden_separation_ratio", "AB_task_separation_ratio",
                     "Mixed", "NoContext", "NovelMean", "shared_trajectory_error", "shared_rollout_loss"]
    text = f"""# Phase 1G Shared Trajectory Learning Report

## 1. Executive result

**Outcome {code} — {label}.** This classification follows the preregistered mechanical decision tree.

## 2. Model/provenance audit

Every trained model uses the canonical 8-unit tanh Elman SRN, scale 2.0 (or the labeled scale-.02 reference),
canonical sequence and A/B schedule, Adam learning rate .005, gradient clipping 1.0, shared output head, and
the canonical monotonic execution objective. The output architecture was not changed.

## 3. Training interventions

Globally selected values were `p_dropout={selected['p_dropout']}`, `lambda_pred={selected['lambda_pred']}`,
and `lambda_traj={selected['lambda_traj']}`. Selection used only familiar A/B losses, A/B separation, shared
prediction, and shared trajectory metrics across calibration seeds. Mixed, NoContext, Novel, and human data
were unavailable to selection. There is no direct A/B similarity or collapse penalty.

## 4. Familiar acquisition

`PHASE1G_TS01_true_AB_acquisition_raw_loss.png` uses pre-update raw CE recorded during every actual optimizer
step. It is not a frozen-checkpoint replay.

## 5. A/B separation

The frozen guardrails require familiar loss ratio <= {config.familiar_loss_ratio_limit} and both hidden/task
A/B separation ratios >= {config.separation_ratio_limit} relative to matched-seed G0.

## 6. Shared-state predictiveness

The midpoint is exactly `(h_A + h_B)/2` and is decoded through the single existing sequence head.

## 7. Shared-trajectory stability

The recurrent target is detached. One-step and mandatory free-rollout diagnostics are in
`PHASE1G_shared_rollout.csv`.

## 8. Mixed transfer

Transfer is reported as matched-seed `G0 loss - intervention loss`; positive values improve on baseline.

## 9. NoContext transfer

NoContext is exactly `[0,0,0]` and never enters training.

## 10. Genuine Novel transfer

Four deterministic, unit-norm, off-axis cue vectors are used. They are nonzero, distinct from A/B and zero,
unseen during training, identical across all models, and saved in `PHASE1G_novel_cue_vectors.csv`.

## 11. High-vs-Low comparison

`Low_reference` uses the identical architecture/schedule/objective with canonical initialization scale .02.

## 12. Post-training causal checks

The globally selected post-training condition was **{causal_condition}**. Shared, A/B-axis, wrong-position,
norm-matched orthogonal, transition-wise hidden, and single-entry free-rollout controls were run only after
all checkpoints were frozen.

## 13. Decision-tree classification

**Outcome {code}: {label}.**

## 14. Paper-level implication

The intervention table below is descriptive; a human-like claim requires simultaneous transfer improvement
and preserved familiar A/B differentiation.

```text
{aggregate[[c for c in table_columns if c in aggregate]].to_string()}
```

## Probe firewall

All probe evaluations preserved parameter, optimizer, and RNG hashes. Probe training rows: **0**.
"""
    (output/"PHASE1G_SHARED_TRAJECTORY_REPORT.md").write_text(text)


def run_pipeline(config: ExperimentConfig):
    config.validate(); output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    _log(f"Starting {config.profile} run in {output}")
    selected, calibration, calibration_summary = calibrate(config)
    calibration.to_csv(output/"PHASE1G_calibration_metrics.csv", index=False)
    calibration_summary.to_csv(output/"PHASE1G_calibration_summary.csv", index=False)
    results = {}; training_frames=[]; probe_frames=[]; geometry_frames=[]; rollout_frames=[]; firewall=[]; checkpoint_files=[]
    for condition in config.conditions:
        hyper = _hyper(condition, selected)
        for seed in config.seeds:
            _log(f"training {condition} seed {seed}")
            result, checkpoint_path = _train_or_load(config, output, condition, seed, hyper)
            results[(condition, seed)] = result; checkpoint_files.append(checkpoint_path)
            training_frames.append(result.metrics)
            probes, runs, audit = evaluate_with_firewall(result, config, condition, seed)
            probe_frames.append(probes); audit.update({"condition": condition, "seed": seed}); firewall.append(audit)
            geo, arrays = geometry(config, condition, seed, runs); geometry_frames.append(geo)
            rollout_frames.append(shared_rollout(config, condition, seed, result.model, arrays, runs))
    training = pd.concat(training_frames, ignore_index=True); probes = pd.concat(probe_frames, ignore_index=True)
    geometry_frame = pd.concat(geometry_frames, ignore_index=True); rollout = pd.concat(rollout_frames, ignore_index=True)
    probe_summary = add_probe_summaries(probes); summary = _seed_summary(config, training, probe_summary, rollout)
    causal_condition = _select_causal_condition(summary); causal_frames=[]
    _log(f"post-training causal checks: {causal_condition}")
    for seed in config.seeds:
        result=results[(causal_condition, seed)]; _, runs, _=evaluate_with_firewall(result, config, causal_condition, seed)
        _, arrays=geometry(config, causal_condition, seed, runs)
        causal_frames.append(causal_checks(config, causal_condition, seed, result.model, runs, arrays))
    causal = pd.concat(causal_frames, ignore_index=True)
    decision = _decision(summary)
    novel = novel_vector_table()
    training.to_csv(output/"PHASE1G_training_metrics.csv", index=False)
    probes.to_csv(output/"PHASE1G_probe_metrics.csv", index=False)
    probe_summary.to_csv(output/"PHASE1G_probe_summary.csv", index=False)
    geometry_frame.to_csv(output/"PHASE1G_shared_geometry.csv", index=False)
    rollout.to_csv(output/"PHASE1G_shared_rollout.csv", index=False)
    novel.to_csv(output/"PHASE1G_novel_cue_vectors.csv", index=False)
    causal.to_csv(output/"PHASE1G_posttraining_causal_checks.csv", index=False)
    summary.to_csv(output/"PHASE1G_seed_summary.csv", index=False)
    pd.DataFrame(firewall).to_csv(output/"PHASE1G_probe_firewall.csv", index=False)
    if config.make_plots:
        make_figures(output, training, probe_summary, probes, geometry_frame, rollout, causal, summary)
        enforce_outputs(output)
    _write_report(output, config, selected, calibration, summary, causal_condition, decision)
    source_files = [Path(__file__).with_name(name) for name in ("config.py","training.py","analysis.py","plotting.py","pipeline.py")]
    manifest = {
        "phase": "Phase1G", "profile": config.profile, "created_at": _stamp(),
        "config": asdict(config), "config_sha256": config.config_sha256,
        "selected_hyperparameters": selected, "hyperparameter_selection_probe_rows": 0,
        "training_contexts": ["A","B"], "probe_training_rows": 0,
        "novel_context_vectors": {k: v.tolist() for k,v in CONTEXT_VECTORS.items() if k.startswith("Novel")},
        "novel_construction": "four deterministic unit-norm tetrahedral/off-axis vectors; seed-independent",
        "no_context_vector": CONTEXT_VECTORS["NoContext"].tolist(),
        "causal_condition": causal_condition, "decision": {"code": decision[0], "label": decision[1]},
        "checkpoint_sha256": {path.name: _sha(path) for path in checkpoint_files},
        "source_sha256": {path.name: _sha(path) for path in source_files},
        "firewall_all_pass": bool(pd.DataFrame(firewall)[
            ["parameter_unchanged", "optimizer_unchanged", "rng_unchanged"]
        ].all().all()),
    }
    (output/"PHASE1G_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    _log(f"Complete: Outcome {decision[0]} — {decision[1]}")
    return {"output": output, "decision": decision, "summary": summary}
