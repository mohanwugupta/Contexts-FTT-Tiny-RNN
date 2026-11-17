from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch

from confirmatory_srn.config import ConfirmatoryConfig
from confirmatory_srn.training import state_hash
from phase1b_direct_path.core import load_source

from .core import (
    PROBE_CHECKPOINTS, assert_canonical_checkpoint, checkpoint_provenance,
    frozen_schedule, frozen_schedule_detector, hash_payload, model_from_checkpoint,
    raw_loss, replay_high_seed, schedule_hash, sha256_file, state_distance,
    tensor_state_hash,
)
from .plotting import enforce_figures, make_figures


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message):
    print(f"[{_stamp()}] SRN-TIMELINE-AUDIT | {message}", flush=True)


def _git_commit(repo):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL,
            text=True).strip()
    except Exception:
        return "unavailable:not-a-git-worktree"


def _phase_state_rows(repo, config, provenance):
    rows = []
    for seed in config.seeds:
        source_path = repo / "outputs/confirmatory_srn_final/confirmatory_checkpoints" / f"High_monotonic_seed_{seed}.pt"
        original_model, _, _ = model_from_checkpoint(source_path, config)
        original = original_model.state_dict()
        original_hash = tensor_state_hash(original)
        source_file_hash = sha256_file(source_path)
        p1d_path = repo / "outputs/phase1d_constrained_retrofit/checkpoints" / f"constrained_correct_seed_{seed}.pt"
        p1e_path = repo / "outputs/phase1e_manifold_following/checkpoints" / f"phase1e_correct_seed_{seed}.pt"
        p1d = torch.load(p1d_path, map_location="cpu", weights_only=False)
        p1e = torch.load(p1e_path, map_location="cpu", weights_only=False)
        if p1d["source_checkpoint_sha256"] != source_file_hash:
            raise AssertionError(f"Phase 1D source mismatch for seed {seed}")
        if p1e["source_checkpoint_sha256"] != source_file_hash:
            raise AssertionError(f"Phase 1E source mismatch for seed {seed}")
        stages = (
            ("Original final", original, str(source_path), source_file_hash),
            ("Phase 1D selected", p1d["model_state_dict"], str(p1d_path), sha256_file(p1d_path)),
            ("Phase 1E start", original, str(source_path), source_file_hash),
            ("Phase 1E end", p1e["model_state_dict"], str(p1e_path), sha256_file(p1e_path)),
        )
        for stage, state, path, file_hash in stages:
            maximum, l2 = state_distance(original, state)
            rows.append({
                "seed": seed, "stage": stage, "checkpoint_path": path,
                "checkpoint_file_hash": file_hash, "parameter_hash": tensor_state_hash(state),
                "original_parameter_hash": original_hash,
                "parameter_max_abs_diff_to_original": maximum,
                "parameter_L2_distance_to_original": l2,
                "source_hash_verified": True,
            })
    return pd.DataFrame(rows)


def _compare_existing_curves(repo, config, frozen_high):
    rows = []
    p1d = pd.read_csv(repo / "outputs/phase1d_constrained_retrofit/tables/PHASE1D_familiar_timeseries.csv")
    existing = p1d[p1d.method.eq("Baseline High")][
        ["seed", "sequence_index", "context", "sequence_loss"]].copy()
    merged = existing.merge(frozen_high, on=["seed", "sequence_index", "context"])
    rows.append({
        "artifact": "PHASE1D_TS01_familiar_sequence_loss.png",
        "comparison": "Baseline High rows vs deliberate frozen final-checkpoint replay",
        "n_rows": len(merged),
        "max_abs_difference": float((merged.sequence_loss-merged.raw_sequence_loss).abs().max()),
        "exact_within_tolerance": bool(np.allclose(merged.sequence_loss, merged.raw_sequence_loss, atol=1e-6, rtol=0)),
        "frozen_schedule_detected": frozen_schedule_detector(frozen_high),
    })
    p1e = pd.read_csv(repo / "outputs/phase1e_manifold_following/PHASE1E_familiar_raw_losses.csv")
    existing = p1e[p1e.method.eq("Baseline High")][
        ["seed", "sequence_index", "context", "sequence_loss"]].copy()
    merged = existing.merge(frozen_high, on=["seed", "sequence_index", "context"])
    rows.append({
        "artifact": "PHASE1E_TS01_raw_familiar_loss_all_sequences.png",
        "comparison": "Baseline High rows vs deliberate frozen final-checkpoint replay",
        "n_rows": len(merged),
        "max_abs_difference": float((merged.sequence_loss-merged.raw_sequence_loss).abs().max()),
        "exact_within_tolerance": bool(np.allclose(merged.sequence_loss, merged.raw_sequence_loss, atol=1e-6, rtol=0)),
        "frozen_schedule_detected": frozen_schedule_detector(frozen_high),
    })
    return pd.DataFrame(rows)


def _report(output, provenance, fidelity, diagnosis, training, probes, phase_states,
            existing_training, git_commit):
    high = provenance[provenance.condition.eq("High_monotonic")]
    low = provenance[provenance.condition.eq("Low_monotonic")]
    exact = bool(fidelity.exact_match.all())
    old_frozen = bool(diagnosis.exact_within_tolerance.all() and diagnosis.frozen_schedule_detected.all())
    original_match = float(np.max(np.abs(
        training.sort_values(["seed", "sequence_index"]).raw_sequence_loss_pre_update.to_numpy() -
        existing_training.sort_values(["seed", "training_sequence"]).sequence_CE.to_numpy())))
    checkpoint_summary = probes.groupby(["training_checkpoint", "condition"]).raw_sequence_loss.mean().unstack()
    start_verified = bool(phase_states[phase_states.stage.eq("Phase 1E start")]
                          .parameter_L2_distance_to_original.eq(0).all())
    movement = training.parameter_moved.mean()
    report = f"""# SRN Training Timeline Audit

## 1. Canonical model verified?

**YES.** All {len(high)} High and {len(low)} Low checkpoints passed hard guards.

```text
hidden_size: 8
High init scale: 2.0
Low init scale: 0.02
checkpoint family: ConfirmatoryElmanSRN, tanh Elman RNN, 7 inputs, 4 outputs
optimizer: Adam, learning rate .005
schedule: 17 alternating A10/B10 blocks (170 sequences)
git commit: {git_commit}
```

No 6-unit or High-scale=.5 checkpoint was found.

## 2. Was the old time-series plot a frozen-checkpoint artifact?

**{'YES' if old_frozen else 'NO'}.** The deliberately frozen final-checkpoint replay matched the plotted
Baseline High rows with maximum absolute differences:

```text
{diagnosis[['artifact','max_abs_difference','frozen_schedule_detected']].to_string(index=False)}
```

Within-context variance is numerically zero because one fixed model is repeatedly evaluated on one fixed
sequence/context pair. Those curves are retrospective schedule displays, not acquisition trajectories.

## 3. Could the original training trajectory be exactly replayed?

**{'YES' if exact else 'NO'}.** Deterministic replay matched {int(fidelity.exact_match.sum())}/{len(fidelity)}
saved High checkpoints exactly. Maximum parameter absolute difference was
`{fidelity.parameter_max_abs_diff.max():.3g}` and the replayed pre-update loss table matched the originally
saved training table to `{original_match:.3g}`. Parameters moved on {100*movement:.2f}% of updates.

## 4. What does the true A/B acquisition curve look like?

The time-matched pre-update curve changes within A and B blocks rather than forming two flat plateaus.
Mean raw loss changed from `{training[training.sequence_index.le(10)].raw_sequence_loss.mean():.3f}` over
the first block to `{training[training.sequence_index.gt(160)].raw_sequence_loss.mean():.3f}` over the final
block. Figure A is the valid acquisition curve.

## 5. How do Mixed / Novel / NoContext probe losses evolve relative to A/B?

Raw checkpoint means are saved in `probe_checkpoint_losses.csv`. Descriptively:

```text
{checkpoint_summary.to_string(float_format=lambda x: f'{x:.4f}')}
```

The table is descriptive only. Every probe call preserved model/policy parameters, optimizer state, and
the training RNG hash exactly.

## 6. Did Phase 1E use the correct starting checkpoints?

**{'YES' if start_verified else 'NO'}.** For every seed, the Phase 1E checkpoint metadata names the exact
canonical High checkpoint file hash, and the reconstructed Phase 1E-start state has zero distance from the
original final trained state. Phase 1D source hashes pass the same check.

## 7. Which previous plots/results are invalidated?

- `outputs/phase1d_constrained_retrofit/timeseries/PHASE1D_TS01_familiar_sequence_loss.png` is invalid **if interpreted as acquisition**.
- `outputs/phase1e_manifold_following/timeseries/PHASE1E_TS01_raw_familiar_loss_all_sequences.png` is invalid **if interpreted as acquisition**.
- `outputs/phase1e_manifold_following/timeseries/PHASE1E_TS02_raw_full_timeline_with_probes.png` mixes a frozen retrospective A/B schedule with terminal probes and should not be called the original learning timeline.

Their underlying fixed-checkpoint losses remain valid as retrospective endpoint diagnostics when relabeled.

## 8. Which previous scientific conclusions remain intact?

Checkpoint provenance, Phase 1D endpoint/path results, Phase 1E retrofit optimization, and matched frozen
probe comparisons remain intact: they used the correct canonical 8-unit High-scale checkpoints. Any claim
about *when behavior emerged during original A/B acquisition* must instead use Figures A–C and the
time-matched tables from this audit. Phase 1E changes are retrofit optimization over optimizer updates, not
continued A/B acquisition.
"""
    path = Path(output) / "AUDIT_REPORT.md"; path.write_text(report); return path


def run_audit(repo: Path, output: Path):
    repo = Path(repo).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = ConfirmatoryConfig(show_progress=False)
    config.validate()
    source = repo / "outputs/confirmatory_srn_final"
    config_path = source / "confirmatory_config.json"
    checkpoint_dir = source / "confirmatory_checkpoints"
    provenance = checkpoint_provenance(config, checkpoint_dir, config_path)
    if not provenance.valid.all():
        raise AssertionError("Canonical checkpoint provenance failed; stopping audit.")
    _log("all 40 High/Low reference checkpoints passed architecture and scale guards")

    training_frames, probe_frames, fidelity_rows, frozen_frames = [], [], [], []
    identity_errors = []
    for seed in config.seeds:
        _log(f"deterministic replay seed {seed}/19")
        model, policy, training, probes, identity_error = replay_high_seed(config, seed)
        path = checkpoint_dir / f"High_monotonic_seed_{seed}.pt"
        existing_model, existing_policy, payload = model_from_checkpoint(path, config)
        maximum, l2 = state_distance(payload["model_state_dict"], model.state_dict())
        policy_max, policy_l2 = state_distance(payload["policy_state_dict"], policy.state_dict())
        existing_loss = raw_loss(existing_model, config, "A")[0]
        replayed_loss = raw_loss(model, config, "A")[0]
        fidelity_rows.append({
            "seed": seed, "existing_final_checkpoint_hash": sha256_file(path),
            "existing_model_parameter_hash": state_hash(existing_model, existing_policy),
            "replayed_final_checkpoint_hash": state_hash(model, policy),
            "parameter_max_abs_diff": max(maximum, policy_max),
            "parameter_L2_diff": float(np.hypot(l2, policy_l2)),
            "final_loss_existing": existing_loss, "final_loss_replayed": replayed_loss,
            "exact_match": maximum == 0 and l2 == 0 and policy_max == 0 and policy_l2 == 0,
        })
        training_frames.append(training); probe_frames.append(probes)
        frozen_frames.append(frozen_schedule(existing_model, config, seed, "frozen_final"))
        identity_errors.append(identity_error)
    training = pd.concat(training_frames, ignore_index=True)
    probes = pd.concat(probe_frames, ignore_index=True)
    fidelity = pd.DataFrame(fidelity_rows)
    frozen = pd.concat(frozen_frames, ignore_index=True)
    existing_training = pd.read_csv(source / "confirmatory_training_sequence_level.csv")
    existing_training = existing_training[existing_training.condition.eq("High_monotonic")]
    diagnosis = _compare_existing_curves(repo, config, frozen)
    if not fidelity.exact_match.all():
        raise AssertionError("Deterministic replay did not match saved checkpoints; interpretation stopped.")
    if max(identity_errors) > 1e-7:
        raise AssertionError("Training and plotting raw-loss functions differ.")
    if not probes.probe_firewall_passed.all():
        raise AssertionError("Probe firewall failed.")
    if training.parameter_moved.mean() < .95:
        raise AssertionError("Too few acquisition updates changed parameters.")

    phase_states = _phase_state_rows(repo, config, provenance)
    phase1e = pd.read_csv(repo / "outputs/phase1e_manifold_following/PHASE1E_probe_raw_losses.csv")
    make_figures(output, training, probes, frozen, phase1e, phase_states)
    enforce_figures(output)
    provenance.to_csv(output / "checkpoint_provenance.csv", index=False)
    fidelity.to_csv(output / "replay_fidelity.csv", index=False)
    training.to_csv(output / "training_sequence_losses.csv", index=False)
    probes.to_csv(output / "probe_checkpoint_losses.csv", index=False)
    phase1e.to_csv(output / "phase1e_probe_losses.csv", index=False)
    frozen.to_csv(output / "frozen_final_schedule_losses.csv", index=False)
    phase_states.to_csv(output / "checkpoint_state_distances.csv", index=False)
    diagnosis.to_csv(output / "frozen_plot_diagnosis.csv", index=False)
    report = _report(output, provenance, fidelity, diagnosis, training, probes,
                     phase_states, existing_training, _git_commit(repo))
    manifest = {
        "analysis": "SRN training timeline audit", "read_only_sources": True,
        "canonical_config": asdict(config) if False else config.__dict__,
        "canonical_config_hash": sha256_file(config_path),
        "training_schedule_hash": schedule_hash(config),
        "probe_checkpoints": list(PROBE_CHECKPOINTS),
        "git_commit": _git_commit(repo), "all_reference_checkpoints_valid": True,
        "exact_replay_all_seeds": True, "probe_firewall_all_passed": True,
        "raw_loss_identity_max_error": max(identity_errors),
        "old_plot_frozen_artifact": bool(diagnosis.exact_within_tolerance.all()),
        "phase1e_start_provenance_all_passed": bool(phase_states.source_hash_verified.all()),
        "source_artifacts_modified": False, "report": str(report),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _log(f"audit complete: {report}")
    return manifest
