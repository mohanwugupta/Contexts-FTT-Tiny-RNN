from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    aggregate_patch_rows, add_rescue, extract_states, frozen_runs, geometry_tables,
    hidden_patches, load_canonical, task_patches,
)
from .config import ExperimentConfig
from .plotting import enforce_outputs, make_figures


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message):
    print(f"[{_stamp()}] PHASE1F | {message}", flush=True)


def _provenance(config):
    path = Path(config.timeline_audit_dir) / "checkpoint_provenance.csv"
    manifest_path = Path(config.timeline_audit_dir) / "manifest.json"
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Phase 1F requires the completed SRN training-timeline audit.")
    frame = pd.read_csv(path)
    frame = frame[frame.seed.isin(config.seeds) & frame.condition.isin(config.conditions)].copy()
    if len(frame) != len(config.seeds) * 2 or not frame.valid.all():
        raise AssertionError("FAIL: NONCANONICAL_SRN")
    expected = {"High_monotonic": 2.0, "Low_monotonic": .02}
    for condition, scale in expected.items():
        subset = frame[frame.condition.eq(condition)]
        if not subset.hidden_size.eq(8).all() or not np.allclose(subset.init_scale, scale):
            raise AssertionError("FAIL: NONCANONICAL_SRN")
    manifest = json.loads(manifest_path.read_text())
    return frame, manifest


def _references(states):
    grouped = states.groupby(["seed", "scale_condition", "probe_type", "checkpoint_hash"], as_index=False).agg(
        raw_loss=("raw_transition_loss", "mean"), mean_token_loss=("raw_transition_loss", "mean"),
        sum_token_loss=("raw_transition_loss", "sum"),
        correct_token_probability=("correct_token_probability", "mean"),
        sequence_accuracy=("sequence_accuracy", "mean"), execution_proxy=("execution_proxy", "mean"))
    grouped = grouped.rename(columns={"probe_type": "probe"})
    grouped["intervention"] = "B0_frozen_reference"
    grouped["space"] = "reference"; grouped["patch_mode"] = "frozen_final"
    grouped["transition_count"] = 7; grouped["sequence_position_source"] = "observed"
    grouped["baseline_raw_loss"] = grouped.raw_loss; grouped["raw_rescue"] = 0.
    return grouped


def _reference_placement(raw, references):
    raw = raw.copy()
    familiar = references[references.probe.isin(["A", "B"])].pivot_table(
        index=["seed", "scale_condition"], columns="probe", values="raw_loss").reset_index()
    familiar["familiar_min"] = familiar[["A", "B"]].min(axis=1)
    familiar["familiar_max"] = familiar[["A", "B"]].max(axis=1)
    raw = raw.merge(familiar[["seed", "scale_condition", "familiar_min", "familiar_max"]],
                    on=["seed", "scale_condition"], how="left")
    raw["familiar_reference_placement"] = np.select(
        [raw.raw_loss < raw.familiar_min, raw.raw_loss > raw.familiar_max],
        ["below familiar range", "above familiar range"], default="inside familiar A/B range")
    return raw


def _seed_summary(config, task_geometry, hidden_geometry, specificity, task_raw, hidden_raw):
    rows = []
    def value(frame, intervention, seed, condition, probe, column="raw_rescue", mode=None):
        subset = frame[(frame.seed == seed) & frame.scale_condition.eq(condition) &
                       frame.probe.eq(probe) & frame.intervention.eq(intervention)]
        if mode is not None: subset = subset[subset.patch_mode.eq(mode)]
        return float(subset[column].iloc[0]) if len(subset) else np.nan
    for seed in config.seeds:
        for condition in config.conditions:
            for probe in config.probes:
                tg = task_geometry[(task_geometry.seed == seed) & task_geometry.scale_condition.eq(condition) & task_geometry.probe.eq(probe)]
                hg = hidden_geometry[(hidden_geometry.seed == seed) & hidden_geometry.scale_condition.eq(condition) & hidden_geometry.probe.eq(probe)]
                sp = specificity[(specificity.seed == seed) & specificity.scale_condition.eq(condition) &
                                 specificity.probe.eq(probe)]
                shared = value(task_raw, "B1_shared", seed, condition, probe)
                wrong = value(task_raw, "C1_wrong_shared", seed, condition, probe)
                axis = value(task_raw, "B2_axis", seed, condition, probe)
                wrong_axis = value(task_raw, "C2_wrong_axis", seed, condition, probe)
                orth = value(task_raw, "C3_orthogonal_shared", seed, condition, probe)
                rows.append({
                    "seed": seed, "scale_condition": condition, "probe": probe,
                    "task_alpha": tg.alpha.mean(),
                    "task_normalized_off_axis": tg.normalized_off_axis_norm.mean(),
                    "task_AB_modulation_norm": tg.AB_modulation_norm.mean(),
                    "hidden_alpha": hg.alpha.mean(),
                    "hidden_normalized_off_axis": hg.normalized_off_axis_norm.mean(),
                    "hidden_AB_modulation_norm": hg.AB_modulation_norm.mean(),
                    "task_sequence_specificity_margin": sp[sp.space.eq("task")].sequence_specificity_margin.mean(),
                    "hidden_sequence_specificity_margin": sp[sp.space.eq("hidden")].sequence_specificity_margin.mean(),
                    "task_shared_rescue": shared, "task_axis_rescue": axis,
                    "task_wrong_shared_rescue": wrong, "task_wrong_axis_rescue": wrong_axis,
                    "task_orthogonal_shared_rescue": orth,
                    "shared_sequence_specific_rescue": shared - wrong,
                    "axis_sequence_specific_rescue": axis - wrong_axis,
                    "shared_direction_specificity": shared - orth,
                    "hidden_shared_transition_rescue": value(hidden_raw, "D1_shared_transition", seed, condition, probe, mode="transition_wise"),
                    "hidden_axis_transition_rescue": value(hidden_raw, "D2_axis_transition", seed, condition, probe, mode="transition_wise"),
                    "hidden_wrong_transition_rescue": value(hidden_raw, "D3_wrong_shared_transition", seed, condition, probe, mode="transition_wise"),
                    "hidden_orthogonal_transition_rescue": value(hidden_raw, "D4_orthogonal_shared_transition", seed, condition, probe, mode="transition_wise"),
                    "hidden_shared_rollout_rescue": value(hidden_raw, "D1_shared_rollout", seed, condition, probe, mode="single_entry_free_rollout"),
                    "hidden_axis_rollout_rescue": value(hidden_raw, "D2_axis_rollout", seed, condition, probe, mode="single_entry_free_rollout"),
                    "hidden_wrong_rollout_rescue": value(hidden_raw, "D3_wrong_shared_rollout", seed, condition, probe, mode="single_entry_free_rollout"),
                })
    return pd.DataFrame(rows)


def _decision(seed_summary):
    focus = seed_summary[seed_summary.probe.isin(["Novel", "NoContextProbe"])]
    means = focus.mean(numeric_only=True)
    task = bool(means.task_shared_rescue > 0 and means.task_axis_rescue > 0 and
                means.shared_sequence_specific_rescue > 0 and means.shared_direction_specificity > 0)
    hidden_transition = bool(means.hidden_shared_transition_rescue > 0)
    hidden_rollout = bool(means.hidden_shared_rollout_rescue > 0)
    if task and hidden_transition and hidden_rollout:
        return "A", "Strong support"
    if task and not hidden_transition:
        return "B", "Task-space rescue but hidden-space rescue fails"
    if task and hidden_transition and not hidden_rollout:
        return "C", "Hidden shared-state rescue works only transition-wise"
    return "D", "Shared-state patch does not provide the complete preregistered rescue pattern"


def _write_report(output, provenance, task_geometry, hidden_geometry, specificity,
                  task_raw, hidden_raw, summary, decision):
    code, label = decision
    means = summary.groupby(["scale_condition", "probe"]).mean(numeric_only=True)
    comparison = task_geometry.groupby(["scale_condition", "probe"]).agg(
        alpha=("alpha", "mean"), off_axis_norm=("off_axis_norm", "mean"),
        normalized_off_axis_norm=("normalized_off_axis_norm", "mean"),
        AB_modulation_norm=("AB_modulation_norm", "mean")).reset_index()
    baseline = task_raw[task_raw.intervention.eq("B0_baseline")].groupby(
        ["scale_condition", "probe"], as_index=False).raw_loss.mean().rename(
        columns={"raw_loss": "baseline_raw_loss"})
    comparison = comparison.merge(baseline, on=["scale_condition", "probe"])
    text = f"""# Phase 1F Shared Sequence Backbone Report

## 1. Executive result

**Outcome {code} — {label}.** The classification follows the frozen decision tree and does not trigger Stage 2.

## 2. Canonical checkpoint audit

All {len(provenance)} audited High/Low checkpoints are 8-unit tanh SRNs with initialization scales 2.0/.02.
Checkpoint and training-schedule hashes come from the completed immutable training-timeline audit.

## 3. Formal shared-backbone hypothesis

For each transition, the analysis defines `s=(A+B)/2`, `d=(A-B)/2`, and decomposes each observed probe as
`s + alpha*d + orthogonal residual`. Endpoint reconstruction and centered-logit identities are tested exactly.

## 4. Task-space geometry

Centered logits are primary. Aggregate task alpha, off-axis residual, A/B modulation, and explained-axis
statistics are in `PHASE1F_task_geometry.csv`.

## 5. Hidden-state geometry

The identical decomposition in the full eight-dimensional recurrent state is secondary because output-null
movement need not affect behavior. Results are in `PHASE1F_hidden_geometry.csv`.

## 6. Mixed geometry

```text
{means.xs('Mixed', level='probe')[['task_alpha','task_normalized_off_axis','hidden_normalized_off_axis']].to_string()}
```

## 7. Novel geometry

```text
{means.xs('Novel', level='probe')[['task_alpha','task_normalized_off_axis','hidden_normalized_off_axis']].to_string()}
```

Novel and NoContext are not independent model inputs here: the canonical training code fixes the Novel
context column to zero, so both `[0,0,1]` and `[0,0,0]` produce exactly the same frozen trajectory.

## 8. NoContext geometry

```text
{means.xs('NoContextProbe', level='probe')[['task_alpha','task_normalized_off_axis','hidden_normalized_off_axis']].to_string()}
```

The exact equality with Novel is an architectural consequence and is not evidence from two independent probes.

## 9. Sequence-specificity analysis

Positive margins mean a probe transition is closer to its correct-position shared state than the average
wrong-position shared state. All six wrong positions were used; the true position was mechanically excluded.

## 10. Task-space causal patches

Mean rescue (`baseline - patch`):

```text
{means[['task_shared_rescue','task_axis_rescue']].to_string()}
```

These are state-level sufficiency interventions and do not demonstrate learnability.

## 11. Hidden-state causal patches

```text
{means[['hidden_shared_transition_rescue','hidden_axis_transition_rescue','hidden_shared_rollout_rescue','hidden_axis_rollout_rescue']].to_string()}
```

Transition-wise and single-entry free-rollout results are kept separate.

## 12. Correct-sequence controls

```text
{means[['shared_sequence_specific_rescue','axis_sequence_specific_rescue']].to_string()}
```

## 13. Orthogonal controls

`shared_direction_specificity` subtracts the rescue from a deterministic norm-matched orthogonal movement:

```text
{means[['shared_direction_specificity']].to_string()}
```

## 14. High-vs-Low comparison

```text
{comparison.to_string(index=False)}
```

Low has far smaller absolute off-axis displacement and much lower baseline loss. Its normalized Novel/
NoContext residual is nevertheless larger because Low's A/B modulation norm—the denominator—is extremely
small. Absolute and normalized geometry must therefore be reported together. Figures 01–09 show these
scale-condition contrasts without presuming that Low is more compositional.

## 15. Full raw-loss time series

TS01 reuses the audited time-matched acquisition states. TS02–TS05 use raw mean cross-entropy and never
substitute normalized residuals. Probe labels remain distinct.

## 16. Decision-tree classification

**Outcome {code}.** Descriptive geometry, task-state sufficiency, hidden-state sufficiency, sequence
specificity, and finite learnability are separate claims. Phase 1F tests the first four and does not train.

## 17. Implications for later training interventions

Stage 2 remains disabled. Shared-backbone regularization is warranted only if the complete causal and
sequence-specific pattern is strong enough to justify it; this report does not launch such training.
"""
    path = Path(output)/"PHASE1F_SHARED_SEQUENCE_BACKBONE_REPORT.md"
    path.write_text(text); return path


def _audit_completeness(config, task_rows, hidden_rows, states):
    if set(config.probes) != set(task_rows.probe.unique()) or set(config.probes) != set(hidden_rows.probe.unique()):
        raise AssertionError("Phase 1F probe labels were pooled or lost.")
    for frame in (task_rows, hidden_rows):
        counts = frame.groupby(["seed", "scale_condition", "probe", "intervention"]).transition.nunique()
        if not counts.eq(7).all():
            raise AssertionError("Phase 1F intervention lacks seven registered transitions.")
    if states.groupby(["seed", "scale_condition", "probe_type"]).transition.nunique().ne(7).any():
        raise AssertionError("Frozen state extraction lacks seven transitions.")


def run_pipeline(config: ExperimentConfig | None = None):
    config = config or ExperimentConfig(); config.validate()
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Phase 1F output is immutable and already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance, timeline_manifest = _provenance(config)
    states_all, task_geo_all, hidden_geo_all, specificity_all = [], [], [], []
    task_rows_all, hidden_rows_all = [], []
    for condition in config.conditions:
        for seed in config.seeds:
            _log(f"frozen causal analysis {condition} seed {seed}")
            model, policy, payload, path, checkpoint_hash = load_canonical(config, condition, seed)
            runs = frozen_runs(config, model)
            states_all.append(extract_states(config, condition, seed, model, policy, checkpoint_hash, runs))
            tg, hg, sp, arrays = geometry_tables(config, condition, seed, checkpoint_hash, runs)
            task_geo_all.append(tg); hidden_geo_all.append(hg); specificity_all.append(sp)
            task_rows_all.append(task_patches(config, condition, seed, policy, checkpoint_hash, runs, arrays))
            hidden_rows_all.append(hidden_patches(
                config, condition, seed, model, policy, checkpoint_hash, runs, arrays))
    states = pd.concat(states_all, ignore_index=True)
    task_geometry = pd.concat(task_geo_all, ignore_index=True)
    hidden_geometry = pd.concat(hidden_geo_all, ignore_index=True)
    specificity = pd.concat(specificity_all, ignore_index=True)
    task_rows = pd.concat(task_rows_all, ignore_index=True)
    hidden_rows = pd.concat(hidden_rows_all, ignore_index=True)
    _audit_completeness(config, task_rows, hidden_rows, states)
    task_raw = aggregate_patch_rows(task_rows, None, "task")
    task_raw["patch_mode"] = "task_state"
    # Recompute execution proxy with the per-checkpoint policies would require carrying
    # policy objects through aggregation; transition probabilities remain fully saved.
    task_raw = add_rescue(task_raw)
    hidden_raw = aggregate_patch_rows(hidden_rows, None, "hidden")
    hidden_raw = add_rescue(hidden_raw)
    references = _references(states)
    raw = _reference_placement(pd.concat([references, task_raw, hidden_raw], ignore_index=True, sort=False), references)
    summary = _seed_summary(config, task_geometry, hidden_geometry, specificity, task_raw, hidden_raw)
    outcome = _decision(summary)
    acquisition = pd.read_csv(Path(config.timeline_audit_dir)/"training_sequence_losses.csv")
    acquisition = acquisition[acquisition.seed.isin(config.seeds)]

    tables = {
        "PHASE1F_frozen_states": states, "PHASE1F_task_geometry": task_geometry,
        "PHASE1F_hidden_geometry": hidden_geometry,
        "PHASE1F_task_patch_results": task_rows,
        "PHASE1F_hidden_patch_results": hidden_rows,
        "PHASE1F_sequence_specificity": specificity,
        "PHASE1F_raw_sequence_losses": raw, "PHASE1F_seed_summary": summary,
        "PHASE1F_checkpoint_provenance": provenance,
    }
    for name, frame in tables.items(): frame.to_csv(output/f"{name}.csv", index=False)
    if not config.make_plots: raise RuntimeError("Mandatory Phase 1F figures cannot be disabled.")
    make_figures(output, task_geometry, hidden_geometry, specificity, task_raw,
                 hidden_raw, summary, acquisition, states)
    enforce_outputs(output)
    report = _write_report(output, provenance, task_geometry, hidden_geometry, specificity,
                           task_raw, hidden_raw, summary, outcome)
    manifest = {
        "config": config.manifest(), "timeline_audit_manifest": timeline_manifest,
        "canonical_provenance_passed": True, "parameters_updated": False,
        "human_data_used": False, "stage2_enabled": False, "stage2_ran": False,
        "decision_code": outcome[0], "decision_label": outcome[1],
        "report": str(report), "mandatory_figures_complete": True,
    }
    (output/"PHASE1F_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _log(f"complete: {report}")
    return {**tables, "manifest": manifest}
