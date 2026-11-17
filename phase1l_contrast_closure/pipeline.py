from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from phase1i_context_composition.cue_encoding import CUES

from .analysis import bootstrap_summary, classify, novel_within_seed, outcome_matrix, paired_comparisons, seed_summaries
from .config import C0, C1, C2, C3, ExperimentConfig
from .evaluation import add_novel_mean, evaluate_checkpoint
from .plotting import enforce_figures, make_figures
from .projection import cue_projection_audit
from .training import load_trained_model, train_run


def _stamp(): return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
def _log(message): print(f"[{_stamp()}] PHASE1L | {message}", flush=True)


def _source_hash():
    digest = hashlib.sha256()
    for name in ("config.py", "projection.py", "model.py", "training.py", "evaluation.py", "analysis.py", "plotting.py", "pipeline.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _git_commit(repo):
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception: return "unavailable:not-a-git-worktree"


def _train_or_load(config, output, condition, family, seed):
    checkpoint = output / "checkpoints" / f"{condition}_{family}_seed_{seed}.pt"
    partial = output / "partials"; partial.mkdir(parents=True, exist_ok=True); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = partial / f"{condition}_{family}_seed_{seed}_training.csv"
    metadata_path = partial / f"{condition}_{family}_seed_{seed}_metadata.json"
    if checkpoint.exists() and metrics_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_sha256") != config.config_sha256 or metadata.get("source_sha256") != _source_hash():
            raise RuntimeError(f"Phase 1L partial uses another frozen specification: {condition} {family} seed {seed}")
        model, payload = load_trained_model(config, checkpoint, CUES)
        _log(f"resuming trained {condition} {family} seed {seed}")
        return model, pd.read_csv(metrics_path), payload, checkpoint
    model, metrics, payload = train_run(config, condition, family, seed, CUES, checkpoint)
    metrics.to_csv(metrics_path, index=False)
    metadata_path.write_text(json.dumps({"config_sha256": config.config_sha256, "source_sha256": _source_hash()}, indent=2, sort_keys=True))
    return model, metrics, payload, checkpoint


def _write_report(output, config, summary, bootstrap, outcome):
    code, label = outcome; high = summary[summary.family.eq("High_monotonic")].groupby("condition").mean(numeric_only=True)
    focus = high[["familiar_mean_ce", "AB_hidden_separation", "AB_task_separation", "Mixed_seq_ce", "NoContext_seq_ce", "NovelMean_seq_ce",
                  "Mixed_task_midpoint_residual", "NoContext_task_midpoint_residual", "NovelMean_task_midpoint_residual", "neutral_hidden_closure_residual"]]
    stopping = "The confirmatory computational program is complete; proceed to manuscript writing." if config.profile == "confirmatory" else "Smoke output is engineering validation only; the stopping decision awaits all 20 matched High seeds."
    text = f"""# Phase 1L Constructive Recovery Report

## 1. Executive summary

**Outcome {code} — {label}.** {stopping}

## 2. Frozen specification and provenance

C0--C3 used matched seeds, identical A/B sequence streams, 170 online updates, the canonical optimizer and learning rate,
and eight-unit tanh recurrence. Pre-update losses and changing parameter hashes are retained.

## 3. Cue projection audit

The scalar context coordinate was fit from A and B cue vectors only. Protected cue coordinates are consequences of
their original vectors; no probe label is assigned a coordinate.

## 4. Probe firewall

Only A/B labels, tokens, and next-token losses enter training. Primary closure weight is fixed at one. Protected probes
are evaluated only after every requested checkpoint has been trained and frozen.

## 5. Familiar performance preservation

Raw A/B transition profiles and per-seed sequence losses are exported. Probe recovery cannot qualify as full success
if familiar performance is catastrophically degraded.

## 6. A/B representational separation

Hidden and centered-logit A/B distances are retained at all seven transitions. The prospective substantial-separation
criterion is at least half of the C0 mean in both spaces.

## 7. Three-probe behavioral recovery

Mixed, NoContext, and NovelMean are reported separately; the unweighted mean never substitutes for an individual
probe component. Novel1--4 remain available in the raw and within-seed tables.

## 8. Shared-trajectory geometry

Hidden/task midpoint residuals, endpoint coordinate, and off-axis residual are evaluated naturally without patches,
clamps, adapters, or target substitution.

## 9. Neutral-trajectory closure audit

The neutral rollout is recomputed from scratch after training and recursively inherits its own previous hidden state.
Transition residuals, cumulative drift, centered-logit residual, endpoint geometry, and raw CE are exported.

## 10. Component ablations

```text
{focus.to_string()}
```

C1 and C2 are compared with both C0 and C3 using matched seed identities.

## 11. Seed-level inference

Bootstrap intervals resample paired seed effects only. Transitions and Novel variants are never treated as independent.

## 12. Frozen decision-table outcome

**Outcome {code}: {label}.** Criteria were encoded before the full confirmatory probe reveal.

## 13. Final interpretation and stop rule

{stopping} The SRN establishes constructive sufficiency or a clean computational boundary; it does not establish a
literal biological implementation in humans.
"""
    (output / "PHASE1L_CONSTRUCTIVE_RECOVERY_REPORT.md").write_text(text)


def run_pipeline(config: ExperimentConfig):
    config.validate(); output = config.resolve(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    audit = cue_projection_audit(CUES, CUES["A"], CUES["B"], config.epsilon)
    alpha = audit.set_index("cue_name").alpha
    if not np.isclose(alpha["Mixed"], 0.0, atol=1e-7): raise AssertionError("FAIL: MIXED_PROJECTION_GEOMETRY")
    projection_hash = hashlib.sha256(audit.to_csv(index=False).encode()).hexdigest()
    _log(f"starting {config.profile}; training all A/B-only checkpoints before probe evaluation")
    registry = {}; training_frames = []
    for family in config.families:
        for condition in config.conditions:
            for seed in config.seeds:
                _log(f"training {condition} {family} seed {seed}")
                model, metrics, payload, checkpoint = _train_or_load(config, output, condition, family, seed)
                registry[(condition, family, seed)] = (payload, checkpoint); training_frames.append(metrics)
    _log("all checkpoints frozen; beginning one-time familiar-first evaluation")
    transition_frames = []; neutral_frames = []; provenance = []
    for family in config.families:
        for condition in config.conditions:
            for seed in config.seeds:
                payload, checkpoint = registry[(condition, family, seed)]
                model, loaded = load_trained_model(config, checkpoint, CUES)
                transition, neutral, record = evaluate_checkpoint(config, condition, family, seed, model, loaded, checkpoint, CUES, projection_hash)
                transition_frames.append(transition); neutral_frames.append(neutral); provenance.append(record)
    training = pd.concat(training_frames, ignore_index=True); transition = add_novel_mean(pd.concat(transition_frames, ignore_index=True)); neutral = pd.concat(neutral_frames, ignore_index=True)
    if not training.parameters_changed.all() or training.groupby(["condition", "family", "seed"]).parameter_hash.nunique().min() != 170:
        raise AssertionError("FAIL: ACQUISITION_PROVENANCE")
    summary = seed_summaries(transition, neutral); novel = novel_within_seed(transition)
    comparisons = paired_comparisons(summary); bootstrap = bootstrap_summary(comparisons, config); matrix = outcome_matrix(summary); outcome = classify(summary)
    geometry_columns = ["condition", "family", "seed", "probe", "novel_id", "transition", "hidden_midpoint_residual", "task_midpoint_residual",
                        "alpha_task", "off_axis_task", "AB_hidden_separation", "AB_task_separation", "cue_alpha", "parameter_hash"]
    git_commit = _git_commit(Path(config.repo_dir).resolve())
    outputs = {
        "phase1l_provenance.csv": pd.DataFrame(provenance), "phase1l_cue_projection_audit.csv": audit,
        "phase1l_training_sequence_metrics.csv": training, "phase1l_final_transition_metrics.csv": transition,
        "phase1l_neutral_rollout_metrics.csv": neutral, "phase1l_geometry_metrics.csv": transition[geometry_columns].copy(),
        "phase1l_seed_summary.csv": summary, "phase1l_novel_within_seed.csv": novel,
        "phase1l_bootstrap_summary.csv": bootstrap, "phase1l_primary_comparisons.csv": comparisons,
    }
    for frame in outputs.values(): frame["git_commit"] = git_commit
    for name, frame in outputs.items(): frame.to_csv(output / name, index=False)
    if config.make_plots:
        make_figures(output, training, transition, neutral, summary, audit, matrix); enforce_figures(output)
    _write_report(output, config, summary, bootstrap, outcome)
    manifest = {"phase": "Phase1L", "profile": config.profile, "created_at": _stamp(), "config": asdict(config),
                "config_sha256": config.config_sha256, "source_sha256": _source_hash(), "git_commit": git_commit,
                "training_probe_labels": [], "lambda_closure": config.lambda_closure, "hidden_state_interventions": 0,
                "novel_mean_members": ["Novel1", "Novel2", "Novel3", "Novel4"],
                "outcome": {"code": outcome[0], "label": outcome[1]}}
    (output / "PHASE1L_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    _log(f"complete: Outcome {outcome[0]} — {outcome[1]}")
    return {"output": output, "outcome": outcome, "seed_summary": summary}
