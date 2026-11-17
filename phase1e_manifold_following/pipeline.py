from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from phase1b_direct_path.core import clone_for_finetuning, load_source
from phase1c_pareto_path.core import construct_wrong_path
from phase1d_constrained_retrofit.constraints import correct_target
from srn_mixed_novel_diagnostics.core import sha256_file

from .analysis import (
    build_seed_metrics, continuous_summary, decision, evaluate_phase1e,
    load_phase1d_model, load_phase1d_tables, probe_rows, success_rates,
)
from .config import ExperimentConfig
from .firewall import assert_matched_checkpoint, assert_optimizer_rows_allowed
from .optimizer import module_hash, train_manifold
from .plotting import enforce_outputs, make_primary, make_timeseries


PARTIALS = ("correct_metric", "wrong_metric", "correct_trace", "wrong_trace",
            "correct_familiar", "wrong_familiar", "correct_full", "wrong_full",
            "correct_probe", "wrong_probe", "summaries")


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message):
    print(f"[{_stamp()}] PHASE1E | {message}", flush=True)


def _hash_payload(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _config_hash(config):
    return _hash_payload(config.manifest())


def _write_optimizer_audit(output):
    text = """# Phase 1E Optimizer Audit

## Phase 1D behavior

| Quantity | Recomputed on every attempted Phase 1D update? | Finding |
|---|---:|---|
| Endpoint constraint residual | Yes | Evaluated at the current parameters. |
| Endpoint constraint Jacobian, J_C | Yes | Rebuilt before every proposal, including after an accepted update. |
| Task-path residual | Yes | Evaluated at the current parameters. |
| Exact task-path Jacobian, J_P | No | The full J_P was formed only for the initial local-reachability audit. Training used the current scalar path gradient. |
| Tangent/null geometry | Yes for J_C | The SVD geometry was rebuilt inside each equality-constrained proposal. |
| Damping scale | Partly | Absolute damping was recomputed from current J_C, but relative damping was globally fixed. |
| Trust-region scale | No | A single source-relative radius was fixed for the run. |
| Equality correction | Yes | Recomputed for each proposal. |
| Nonlinear restoration loop | No | Failed proposals were reverted and learning rate halved; no relinearized restoration iterations were used. |

**Answer:** J_C was recomputed after every accepted update, but an exact J_P was not. Therefore both
Jacobians were not recomputed as a pair after every accepted update. Phase 1E implements full
relinearization plus adaptive trust-region acceptance and nonlinear feasibility restoration, so it is not a
duplicate of Phase 1D.

## Phase 1E frozen implementation decision

Every attempted step starts by recomputing scaled residuals c and r and exact J_C and J_P at the current
accepted parameter vector. Any accepted step invalidates the geometry ledger. Each nonlinear restoration
iteration independently recomputes c and J_C. The trust radius changes only by the registered 0.25/0.75
ratio rules. Mixed, Novel, NoContextProbe, and lambda=.5 are excluded from optimization and selection.
"""
    path = Path(output) / "PHASE1E_OPTIMIZER_AUDIT.md"
    path.write_text(text)
    return path


def _save_checkpoint(path, result, source_hash, config_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"phase": "Phase1E", "condition": result.summary["condition"],
                "seed": result.summary["seed"], "model_state_dict": result.model.state_dict(),
                "summary": result.summary, "source_checkpoint_sha256": source_hash,
                "config_sha256": config_hash}, path)


def _checkpoint_probe_frames(config, result, cache, seed, condition):
    frames = []
    scratch = clone_for_finetuning(result.model)
    for update in sorted(result.checkpoint_states):
        scratch.load_state_dict(result.checkpoint_states[update], strict=True)
        checkpoint_id = f"{condition}_seed_{seed}_update_{update}"
        frames.append(probe_rows(scratch, cache, config, seed, condition, update, checkpoint_id))
    return pd.concat(frames, ignore_index=True)


def _run_seed(config, seed, output, config_hash):
    network, cache = load_source(config, "D0_frozen_high", seed)
    source_hash = sha256_file(network.path)
    target = correct_target(cache, config.lambda_train)
    wrong = construct_wrong_path(network.model, cache, config.sequence, config.lambda_train,
                                 seed, config.wrong_path_seed_offset)
    progress = lambda message: _log(message)
    correct = train_manifold(config, "phase1e_correct", seed, network.model, cache,
                             target, progress=progress)
    wrong_result = train_manifold(config, "phase1e_wrong", seed, network.model, cache,
                                  wrong.target, progress=progress)
    if correct.summary["optimizer_updates"] != wrong_result.summary["optimizer_updates"]:
        raise AssertionError("Correct and wrong paths received unequal optimization exposure.")
    c_metric, c_familiar, c_full, _ = evaluate_phase1e(
        config, seed, correct, network.model, cache)
    w_metric, w_familiar, w_full, _ = evaluate_phase1e(
        config, seed, wrong_result, network.model, cache)
    c_probe = _checkpoint_probe_frames(config, correct, cache, seed, "Phase 1E correct")
    w_probe = _checkpoint_probe_frames(config, wrong_result, cache, seed, "Phase 1E wrong")
    _save_checkpoint(output / "checkpoints" / f"phase1e_correct_seed_{seed}.pt",
                     correct, source_hash, config_hash)
    _save_checkpoint(output / "checkpoints" / f"phase1e_wrong_seed_{seed}.pt",
                     wrong_result, source_hash, config_hash)
    if sha256_file(network.path) != source_hash:
        raise AssertionError("Frozen source checkpoint changed during Phase 1E.")
    return {
        "correct_metric": [c_metric], "wrong_metric": [w_metric],
        "correct_trace": correct.trace, "wrong_trace": wrong_result.trace,
        "correct_familiar": c_familiar.to_dict("records"),
        "wrong_familiar": w_familiar.to_dict("records"),
        "correct_full": c_full.to_dict("records"),
        "wrong_full": w_full.to_dict("records"),
        "correct_probe": c_probe.to_dict("records"),
        "wrong_probe": w_probe.to_dict("records"),
        "summaries": [correct.summary, wrong_result.summary],
    }


def _write_partial(path, data, config_hash):
    path.mkdir(parents=True, exist_ok=True)
    for name in PARTIALS:
        pd.DataFrame(data[name]).to_csv(path / f"{name}.csv", index=False)
    (path / "COMPLETE.json").write_text(json.dumps(
        {"complete": True, "config_sha256": config_hash}, indent=2))


def _load_partial(path, config_hash):
    marker = path / "COMPLETE.json"
    if not marker.exists():
        return None
    metadata = json.loads(marker.read_text())
    if metadata.get("config_sha256") != config_hash:
        raise RuntimeError(f"Stale Phase 1E partial has a different configuration: {path}")
    return {name: pd.read_csv(path / f"{name}.csv", low_memory=False).to_dict("records")
            for name in PARTIALS}


def _prior_raw_tables(config, prior):
    rename = {
        "Baseline High": "Baseline High", "Baseline Low": "Baseline Low",
        "Previous Phase 1C soft penalty": "Phase 1C soft penalty",
        "Constrained correct path": "Phase 1D constrained",
    }
    familiar = prior["familiar"]
    full = prior["full"]
    familiar = familiar[familiar.seed.isin(config.seeds) & familiar.method.isin(rename)].copy()
    full = full[full.seed.isin(config.seeds) & full.method.isin(rename)].copy()
    familiar["method"] = familiar.method.map(rename)
    full["method"] = full.method.map(rename)
    return familiar, full


def _static_reference_probes(config):
    frames = []
    for seed in config.seeds:
        high, cache = load_source(config, "D0_frozen_high", seed)
        frames.append(probe_rows(high.model, cache, config, seed, "Baseline High", 0,
                                 f"baseline_high_seed_{seed}"))
        p1d, cache, payload = load_phase1d_model(config, seed, "correct")
        update = int(payload["summary"]["selected_update"])
        frames.append(probe_rows(p1d, cache, config, seed, "Phase 1D constrained", update,
                                 f"phase1d_seed_{seed}_update_{update}"))
    return pd.concat(frames, ignore_index=True)


def _source_hashes(config):
    result = {}
    for seed in config.seeds:
        network, _ = load_source(config, "D0_frozen_high", seed)
        result[str(network.path)] = sha256_file(network.path)
    return result


def _write_report(output, seed, rates, continuous, label, code):
    rate = rates.set_index("method")
    base = continuous.set_index("contrast")
    gain = base.loc["Baseline High minus Phase 1E"]
    versus = base.loc["Phase 1D minus Phase 1E"]
    specificity = base.loc["Phase 1E wrong minus correct"]
    report = f"""# Phase 1E Manifold-Following Report

## Executive result

**Outcome {code} — {label}.** This label was selected mechanically from the frozen decision rule.

## Separate scientific claims

- **Acute state-level sufficiency:** inherited from the frozen Phase 1D acute-patch audit; it was not re-estimated or tuned here.
- **Local parameter accessibility:** mean Q_local was {seed.Q_local.mean():.3f} using the frozen Phase 1D Jacobian audit.
- **Finite learnability:** Phase 1E correct passed {int(rate.loc['Phase 1E correct'].success_n)}/{int(rate.loc['Phase 1E correct'].n_seeds)} seeds, versus Phase 1D {int(rate.loc['Phase 1D constrained'].success_n)}/{int(rate.loc['Phase 1D constrained'].n_seeds)}.
- **Endpoint preservation:** checkpoint eligibility used the unchanged Phase 1C/1D endpoint, separation, and familiar-loss gates.
- **Correct-vs-wrong specificity:** wrong-minus-correct Mixed loss averaged {specificity['mean']:.3f}, bootstrap 95% CI [{specificity.bootstrap_low:.3f}, {specificity.bootstrap_high:.3f}].

## Continuous outcome

Baseline-minus-Phase1E Mixed loss averaged {gain['mean']:.3f} (median {gain['median']:.3f};
95% CI [{gain.bootstrap_low:.3f}, {gain.bootstrap_high:.3f}]); {100*gain.fraction_positive:.1f}% of seeds improved.
Phase1D-minus-Phase1E averaged {versus['mean']:.3f} (95% CI [{versus.bootstrap_low:.3f},
{versus.bootstrap_high:.3f}]).

## Optimization and firewall

Both exact Jacobians were recomputed from scratch at each current accepted parameter state. Proposals used
normal restoration plus tangent least squares, nonlinear constraint restoration, and adaptive trust radii.
Only familiar A/B endpoint diagnostics and the registered nonmidpoint path entered training or checkpoint
selection. Mixed, Novel, NoContextProbe, lambda=.5, and human data were read only after selection.

## Failure decomposition

{seed.failure_category.value_counts().to_string()}

## Raw-loss interpretation

TS01–TS04 retain cross-entropy in raw model-loss units. TS02 and TS03 are central report figures and show
where Mixed, Novel, and NoContextProbe lie relative to directly evaluated familiar A/B losses.

This experiment distinguishes causal/state sufficiency, local access, and finite endpoint-preserving
learnability. It does not establish necessity, universality, or literal human neural implementation.
"""
    path = Path(output) / "PHASE1E_MANIFOLD_FOLLOWING_REPORT.md"
    path.write_text(report)
    return path


def run_pipeline(config: ExperimentConfig | None = None):
    config = config or ExperimentConfig()
    config.validate()
    output = Path(config.output_dir)
    for directory in ("partials", "checkpoints", "figures", "timeseries", "audit"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    audit_path = _write_optimizer_audit(output)
    assert_optimizer_rows_allowed(pd.DataFrame({"context": ["A", "B"], "is_probe": [False, False]}))
    phase1d = load_phase1d_tables(config)
    _, _, _, phase1c_selected = __import__(
        "phase1d_constrained_retrofit.analysis", fromlist=["load_phase1c_provenance"]
    ).load_phase1c_provenance(config)
    config_hash = _config_hash(config)
    collections = {name: [] for name in PARTIALS}
    for seed in config.seeds:
        partial = output / "partials" / f"seed_{seed}"
        data = _load_partial(partial, config_hash)
        if data is None:
            _log(f"starting seed {seed}")
            data = _run_seed(config, seed, output, config_hash)
            _write_partial(partial, data, config_hash)
            _log(f"seed {seed} safely checkpointed")
        else:
            _log(f"resuming completed seed {seed}")
        for name in PARTIALS:
            collections[name].extend(data[name])
    correct = pd.DataFrame(collections["correct_metric"])
    wrong = pd.DataFrame(collections["wrong_metric"])
    summaries = pd.DataFrame(collections["summaries"])
    trace = pd.concat([pd.DataFrame(collections["correct_trace"]),
                       pd.DataFrame(collections["wrong_trace"])], ignore_index=True)
    local = phase1d["local"][phase1d["local"].seed.isin(config.seeds)]
    master = phase1d["master"][phase1d["master"].seed.isin(config.seeds)]
    selected = phase1c_selected[phase1c_selected.seed.isin(config.seeds)]
    seed = build_seed_metrics(config, master, local, selected, correct, wrong, summaries, trace)
    rates = success_rates(seed)
    continuous = continuous_summary(config, seed)
    code, label = decision(seed, continuous)

    prior_familiar, prior_full = _prior_raw_tables(config, phase1d)
    e_familiar = pd.concat([
        pd.DataFrame(collections["correct_familiar"]).assign(method="Phase 1E correct"),
        pd.DataFrame(collections["wrong_familiar"]).assign(method="Phase 1E wrong")], ignore_index=True)
    e_full = pd.concat([
        pd.DataFrame(collections["correct_full"]).assign(method="Phase 1E correct"),
        pd.DataFrame(collections["wrong_full"]).assign(method="Phase 1E wrong")], ignore_index=True)
    familiar = pd.concat([prior_familiar, e_familiar], ignore_index=True, sort=False)
    full = pd.concat([prior_full, e_full], ignore_index=True, sort=False)
    probes = pd.concat([
        _static_reference_probes(config), pd.DataFrame(collections["correct_probe"]),
        pd.DataFrame(collections["wrong_probe"])], ignore_index=True)
    assert_matched_checkpoint(probes)
    assert_optimizer_rows_allowed(trace.assign(context="A", is_probe=False))
    if trace.midpoint_accessed.fillna(False).any():
        raise AssertionError("Held-out lambda=.5 entered Phase 1E optimization.")

    seed.to_csv(output / "PHASE1E_seed_metrics.csv", index=False)
    trace.to_csv(output / "PHASE1E_optimizer_trace.csv", index=False)
    probes.to_csv(output / "PHASE1E_probe_raw_losses.csv", index=False)
    familiar.to_csv(output / "PHASE1E_familiar_raw_losses.csv", index=False)
    rates.to_csv(output / "PHASE1E_success_rates.csv", index=False)
    continuous.to_csv(output / "PHASE1E_continuous_summary.csv", index=False)
    full.to_csv(output / "PHASE1E_full_timeline_raw_losses.csv", index=False)
    if not config.make_plots:
        raise RuntimeError("Mandatory Phase 1E figures cannot be disabled.")
    make_primary(output, seed, rates)
    make_timeseries(output, familiar, full, probes, trace, config.updates)
    enforce_outputs(output)
    report = _write_report(output, seed, rates, continuous, label, code)
    manifest = {
        "config": config.manifest(), "config_sha256": config_hash,
        "phase1d_artifact_dir": str(config.phase1d_dir),
        "phase1d_optimizer_audit": str(audit_path), "decision_code": code,
        "decision_label": label, "report": str(report),
        "heldout_midpoint_used_for_training": False,
        "heldout_midpoint_used_for_selection": False,
        "probe_data_used_for_training": False, "probe_data_used_for_selection": False,
        "human_data_used": False, "output_head_frozen": True,
        "geometry_recomputed_after_every_accepted_update": True,
        "source_checkpoint_hashes": _source_hashes(config),
    }
    (output / "PHASE1E_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"complete: {report}")
    return {"seed_metrics": seed, "optimizer_trace": trace, "probe_raw_losses": probes,
            "familiar_raw_losses": familiar, "rates": rates, "continuous": continuous,
            "manifest": manifest}
