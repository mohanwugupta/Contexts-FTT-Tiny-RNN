from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from srn_training_timeline_audit.core import sha256_file

from .acquisition import replay_acquisition
from .analysis import (
    add_all_novel_means, bootstrap_summary, classify, local_generation_table,
    novel_within_seed, primary_comparisons, seed_summary, trajectory_summary,
)
from .config import ExperimentConfig
from .interventions import evaluate_seed
from .plotting import enforce_figures, make_figures


PARTS = ("natural", "local", "takeover", "pulse", "endpoint", "convergence", "acquisition")


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message):
    print(f"[{_stamp()}] PHASE1K | {message}", flush=True)


def _source_hash():
    digest = hashlib.sha256()
    for name in ("config.py", "acquisition.py", "interventions.py", "analysis.py", "plotting.py", "pipeline.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _git_commit(repo):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unavailable:not-a-git-worktree"


def _run_or_load(config, output, family, seed):
    partial = output / "partials"; partial.mkdir(parents=True, exist_ok=True); stem = f"{family}_seed_{seed}"
    paths = {name: partial / f"{stem}_{name}.csv" for name in PARTS}
    meta_path = partial / f"{stem}_metadata.json"
    if meta_path.exists() and all(path.exists() for path in paths.values()):
        metadata = json.loads(meta_path.read_text())
        if metadata.get("config_sha256") != config.config_sha256 or metadata.get("source_sha256") != _source_hash():
            raise RuntimeError(f"Partial Phase 1K result uses another frozen specification: {stem}")
        _log(f"resuming completed {family} seed {seed}")
        return {name: pd.read_csv(path) for name, path in paths.items()}, metadata["provenance"]
    frames, provenance = evaluate_seed(config, family, seed)
    frames["acquisition"] = replay_acquisition(config, family, seed)
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False)
    meta_path.write_text(json.dumps({"config_sha256": config.config_sha256, "source_sha256": _source_hash(), "provenance": provenance}, indent=2, sort_keys=True))
    return frames, provenance


def _write_report(output, config, seed_level, bootstrap, outcome):
    code, label = outcome
    high = seed_level[seed_level.family.eq("High_monotonic")].mean(numeric_only=True)
    focus = high[[
        "A_to_M_local_task_generation", "B_to_M_local_task_generation", "AB_to_M_mean_convergence",
        "NoContext_local_task_generation", "NoContext_takeover_remaining_rescue",
        "NovelMean_local_task_generation", "NovelMean_takeover_remaining_rescue",
        "pulse_vs_takeover_difference", "mixed_midpoint_specificity",
    ]]
    stopping = "Confirmatory Outcome A/B satisfies the frozen stop rule; proceed to manuscript writing." if config.profile == "confirmatory" and code in ("A", "B") else (
        "The confirmatory result is a mechanistic boundary to report; reopen modeling only for a demonstrated implementation error." if config.profile == "confirmatory" else
        "Smoke output is engineering validation only; the stop rule awaits all 20 seeds per family."
    )
    text = f"""# Phase 1K Cue-Only Composite Generation Report

## 1. Executive summary

**Outcome {code} — {label}.** {stopping}

## 2. Provenance audit

All analyzed checkpoints were matched to the registered canonical High/Low sources. Independent pre-update replay
reproduced each final checkpoint exactly and retained changing parameter hashes across all 170 training sequences.

## 3. Natural familiar and probe behavior

Natural A, B, Mixed, NoContext, Novel1--4, and within-seed NovelMean retain all seven raw transition losses,
hidden states, raw/centered logits, endpoint coordinates, and checkpoint identities.

## 4. Local cue-generation test

Native, physical Mixed, A, B, and zero cue branches were evaluated from bit-identical natural prestates. Positive
generation scores mean the Mixed cue moved the next state closer to the frozen familiar A/B midpoint.

## 5. A→Mixed and B→Mixed convergence

The normalized distance between A→Mixed and B→Mixed branches is compared with the untouched A/B separation.
Positive `AB_to_M_mean_convergence` means that the common Mixed cue reduced the separation.

## 6. NoContext cue takeover

Every takeover position retains natural, one-step Mixed pulse, persistent Mixed, persistent A, and persistent B
raw CE profiles and output-relevant geometry.

## 7. Novel cue takeover

Novel1--4 are retained separately. NovelMean is calculated within seed from all four preregistered variants before
any family statistic or seed bootstrap.

## 8. One-step pulse versus persistent takeover

`pulse_vs_takeover_difference` is persistent minus pulse mean remaining-sequence rescue. Positive values support
continued contextual stabilization beyond a one-update cue perturbation.

## 9. Endpoint-direction controls

A, Mixed, and B cue branches are compared by endpoint coordinate, off-axis residual, and actual distances to A/B.
Behavioral improvement alone is not used as evidence of compositional specificity.

## 10. High-versus-Low comparison

Low uses the identical code path as a geometric control; near-floor Low CE is not interpreted as weaker rescue.

## 11. Per-seed heterogeneity

```text
{focus.to_string()}
```

All inferential intervals resample seeds, never transitions or Novel variants.

## 12. Frozen decision-table outcome

**Outcome {code}: {label}.** The outcome uses the sign-based qualitative table frozen in the implementation,
without post-hoc magnitude thresholds.

## 13. Mechanistic interpretation

Phase 1K distinguishes cue-driven state generation from direct state sufficiency and from continued maintenance.
The SRN result is a sufficient computational mechanism; it is not a claim that humans literally implement it.

## 14. Limitations

Teacher-forced external tokens remain fixed. Cue takeover is causal within the frozen trained SRN but does not
identify a unique biological mechanism. Midpoint geometry is defined relative to the familiar A/B trajectories.

## 15. Final stopping decision

{stopping}
"""
    (output / "PHASE1K_CUE_GENERATION_REPORT.md").write_text(text)


def run_pipeline(config: ExperimentConfig):
    config.validate(); output = config.resolve(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    _log(f"starting {config.profile} frozen cue-generation diagnostic")
    collected = {name: [] for name in PARTS}; provenance = []
    for family in config.families:
        for seed in config.seeds:
            _log(f"analyzing {family} seed {seed}")
            frames, metadata = _run_or_load(config, output, family, seed)
            for name in PARTS:
                collected[name].append(frames[name])
            provenance.append(metadata)
    frames = {name: pd.concat(values, ignore_index=True) for name, values in collected.items()}
    natural, local, takeover, pulse, endpoint = add_all_novel_means(
        frames["natural"], frames["local"], frames["takeover"], frames["pulse"], frames["endpoint"]
    )
    summary = trajectory_summary(takeover, pulse, endpoint)
    generation = local_generation_table(local)
    novel = novel_within_seed(generation, summary)
    seed_level = seed_summary(local, summary, frames["convergence"])
    comparisons = primary_comparisons(seed_level); bootstrap = bootstrap_summary(comparisons, config); outcome = classify(seed_level)
    acquisition = frames["acquisition"]
    if not acquisition.parameters_changed.all() or acquisition.groupby(["family", "seed"]).parameter_hash.nunique().min() != 170:
        raise AssertionError("FAIL: ACQUISITION_PROVENANCE")
    git_commit = _git_commit(Path(config.repo_dir).resolve())
    outputs = {
        "phase1k_provenance.csv": pd.DataFrame(provenance),
        "phase1k_natural_rollouts.csv": natural,
        "phase1k_local_cue_branches.csv": local,
        "phase1k_takeover_transition_metrics.csv": takeover,
        "phase1k_takeover_summary.csv": summary,
        "phase1k_pulse_transition_metrics.csv": pulse,
        "phase1k_endpoint_control_metrics.csv": endpoint,
        "phase1k_ab_to_mixed_convergence.csv": frames["convergence"],
        "phase1k_novel_within_seed.csv": novel,
        "phase1k_seed_summary.csv": seed_level,
        "phase1k_bootstrap_summary.csv": bootstrap,
        "phase1k_primary_comparisons.csv": comparisons,
        "phase1k_true_acquisition.csv": acquisition,
    }
    for frame in outputs.values():
        frame["git_commit"] = git_commit
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    if config.make_plots:
        make_figures(output, acquisition, natural, local, takeover, pulse, endpoint, frames["convergence"], seed_level)
        enforce_figures(output)
    _write_report(output, config, seed_level, bootstrap, outcome)
    manifest = {
        "phase": "Phase1K", "profile": config.profile, "created_at": _stamp(), "config": asdict(config),
        "config_sha256": config.config_sha256, "source_sha256": _source_hash(),
        "source_config_sha256": sha256_file(config.resolve(config.source_config)), "git_commit": git_commit,
        "frozen_checkpoints": True, "parameter_updates_during_interventions": 0,
        "direct_hidden_state_patches": 0, "teacher_forced_tokens": True,
        "novel_mean_members": ["Novel1", "Novel2", "Novel3", "Novel4"],
        "outcome": {"code": outcome[0], "label": outcome[1]},
    }
    (output / "PHASE1K_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    _log(f"complete: Outcome {outcome[0]} — {outcome[1]}")
    return {"output": output, "outcome": outcome, "seed_summary": seed_level}
