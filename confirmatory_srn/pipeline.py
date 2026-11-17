"""End-to-end frozen confirmatory SRN analysis with no human fitting."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .causal import causal_interventions
from .config import (
    ConfirmatoryConfig, HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY, LOW_MONOTONIC,
)
from .diagnostics import (
    hidden_scale_comparison, hidden_separation_and_pca, seed_training_summary,
    switch_diagnostics,
)
from .low_rank import low_rank_bridge
from .plotting import make_figures
from .probes import evaluate_frozen_probes
from .readout_controls import evaluate_readout_controls
from .responsibility import probe_responsibility_tables
from .state_space import fit_confirmatory_state_space
from .training import train_condition


REQUIRED_OUTPUTS = (
    "confirmatory_config.json",
    "confirmatory_training_sequence_level.csv",
    "confirmatory_training_transition_level.csv",
    "confirmatory_seed_summary.csv",
    "confirmatory_hidden_separation.csv",
    "confirmatory_hidden_states.npz",
    "confirmatory_state_space_fits.csv",
    "confirmatory_state_space_seed_classification.csv",
    "confirmatory_probe_transition_level.csv",
    "confirmatory_probe_responsibility.csv",
    "confirmatory_probe_midpoint_deviation.csv",
    "confirmatory_low_rank_results.csv",
    "confirmatory_low_rank_predictions.csv",
    "confirmatory_causal_patching.csv",
    "confirmatory_causal_controls.csv",
    "srn_human_bridge_table.csv",
    "CONFIRMATORY_SRN_REPORT.md",
)


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] CONFIRMATORY-SRN | {message}", flush=True)


def _hash_sources(repo: Path) -> dict[str, str]:
    directories = (
        "ftt_srn_execution_v2", "execution_objective_search",
        "execution_readout_diagnostic", "commitment_rt",
    )
    result = {}
    for directory in directories:
        for path in sorted((repo / directory).rglob("*.py")):
            result[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _seed_summary(sequence, separation, switch_costs, classifications, midpoint):
    summary = seed_training_summary(sequence)
    hidden = separation[separation.row_type.eq("seed_aggregate")][[
        "condition", "seed", "aggregate_AB_separation"
    ]]
    summary = summary.merge(hidden, on=["condition", "seed"], how="left")
    if len(switch_costs):
        switch = switch_costs.groupby(["condition", "seed"], as_index=False).agg(
            mean_switch_cost=("switch_cost", "mean"),
            fraction_positive_switches=("switch_cost_positive", "mean"),
        )
        summary = summary.merge(switch, on=["condition", "seed"], how="left")
    seed_class = classifications[classifications.row_type.eq("seed")][[
        "condition", "seed", "model_label", "BIC", "RMSE", "R2"
    ]].rename(columns={
        "model_label": "state_space_best_model", "BIC": "state_space_BIC",
        "RMSE": "state_space_RMSE", "R2": "state_space_R2",
    })
    summary = summary.merge(seed_class, on=["condition", "seed"], how="left")
    ab = midpoint.groupby(["condition", "seed"], as_index=False).agg(
        mean_AB_RT_difference=("AB_difference", "mean"),
        mean_absolute_AB_RT_difference=("AB_difference", lambda x: float(np.abs(x).mean())),
    )
    return summary.merge(ab, on=["condition", "seed"], how="left")


def _interpretation(config, scale_comparison, seed_summary, patching, controls):
    comparison = scale_comparison.iloc[0]
    representation = bool(
        comparison.mean_paired_difference > 0
        and comparison.fraction_seeds_high_greater_low >= .8
    )
    high = seed_summary[seed_summary.condition.eq(HIGH_MONOTONIC)]
    behavior = bool(high.mean_absolute_AB_RT_difference.mean() >= .01)
    # The paper-level causal statement is anchored to the High-scale mechanism.
    # Preserve the project's pre-existing .90 sequence criterion and additionally
    # require high correct-target accuracy, as specified in this PRD.  Small CE
    # disruption is operationalized conservatively as smaller than every control.
    high_patching = patching[patching.condition.eq(HIGH_MONOTONIC)]
    high_controls = controls[controls.condition.eq(HIGH_MONOTONIC)]
    matched_endpoint = float(high_patching.endpoint_proximity_improvement.mean())
    control_endpoint = high_controls.groupby(
        "intervention_type"
    ).endpoint_proximity_improvement.mean()
    matched_alignment = float(high_patching.RT_direction_aligned.mean())
    control_alignment = high_controls.groupby("intervention_type").RT_direction_aligned.mean()
    preservation = float(high_patching.prediction_identity_preserved.mean())
    accuracy = float(high_patching.intervened_next_token_correct.mean())
    matched_ce = float(high_patching.delta_sequence_CE.abs().mean())
    control_ce = high_controls.assign(
        absolute_CE_change=high_controls.delta_sequence_CE.abs()
    ).groupby("intervention_type").absolute_CE_change.mean()
    causal = bool(
        matched_endpoint > 0
        and matched_endpoint > float(control_endpoint.max())
        and matched_alignment > float(control_alignment.max())
        and preservation >= config.sequence_preservation_threshold
        and accuracy >= config.sequence_preservation_threshold
        and matched_ce < float(control_ce.min())
    )
    if representation and causal:
        return "A", "representational scale produces context-specific states that causally organize RT"
    if representation:
        return "B", "representational splitting exists but does not reliably control RT"
    if behavior:
        return "C", "RT differences exist without strong representational correspondence"
    return "D", "neither representational nor behavioral context splitting replicates"


def _write_report(output: Path, config: ConfirmatoryConfig, sequence, seed_summary,
                  scale_comparison, fits, classifications, midpoint, responsibility,
                  low_rank_results, patching, controls):
    condition_training = seed_summary.groupby("condition", as_index=False).agg(
        CE_early=("CE_early", "mean"), CE_late=("CE_late", "mean"),
        confidence_early=("confidence_early", "mean"),
        confidence_late=("confidence_late", "mean"),
        fraction_CE_decreased=("CE_decreased", "mean"),
        RT_early=("RT_early", "mean"), RT_late=("RT_late", "mean"),
        fraction_RT_decreased=("RT_decreased", "mean"),
        AB_hidden_separation=("aggregate_AB_separation", "mean"),
        mean_switch_cost=("mean_switch_cost", "mean"),
        mean_absolute_AB_RT_difference=("mean_absolute_AB_RT_difference", "mean"),
    )
    mean_winners = fits[fits.fit_scope.eq("seed_average") & fits.best_model][[
        "condition", "model_label", "BIC", "AIC", "RMSE", "R2", "converged"
    ]]
    fractions = classifications[
        classifications.row_type.eq("classification_fraction")
    ][["condition", "model_label", "classified_seed_count", "classification_fraction"]]
    probe_summary = midpoint[midpoint.probe_context.isin(["Mixed_norm", "Novel"])].groupby(
        ["condition", "probe_context"], as_index=False
    ).agg(
        A_RT=("A_RT", "mean"), B_RT=("B_RT", "mean"),
        probe_RT=("probe_RT", "mean"),
        midpoint_deviation=("probe_midpoint_deviation", "mean"),
    )
    q_summary = responsibility[
        responsibility.probe_context.isin(["Mixed_norm", "Novel"])
    ].groupby(["condition", "probe_context"], as_index=False).agg(
        global_q=("q_global_unclipped", "mean"),
        transition_q_mean=("q_unclipped", "mean"),
        transition_q_sd=("q_unclipped", "std"),
        usable_transitions=("responsibility_denominator_usable", "sum"),
    )
    selected_rank = low_rank_results[low_rank_results.selected][[
        "condition", "selected_rank", "AB_only_CV_MSE", "probe_heldout_MSE",
        "global_scalar_MSE", "low_rank_supported",
    ]]
    causal_summary = pd.concat([
        patching.assign(table="matched"), controls.assign(table="control")
    ]).groupby(["table", "condition", "intervention_type"], as_index=False).agg(
        mean_delta_RT=("delta_normalized_RT", "mean"),
        mean_endpoint_improvement=("endpoint_proximity_improvement", "mean"),
        direction_aligned=("RT_direction_aligned", "mean"),
        identity_preserved=("prediction_identity_preserved", "mean"),
        next_token_accuracy=("intervened_next_token_correct", "mean"),
        mean_absolute_CE_change=("delta_sequence_CE", lambda x: float(np.abs(x).mean())),
    )
    code, interpretation = _interpretation(
        config, scale_comparison, seed_summary, patching, controls
    )
    report = f"""# Confirmatory SRN report

## 1. Frozen model specification

- Architecture: 8-unit tanh Elman SRN, four-way next-token output, 3D context basis.
- High/Low initialization scales: `2.0` / `.02`.
- Canonical seeds: `{list(config.seeds)}`.
- Curriculum: 17 alternating A10/B10 blocks; 170 external sequences.
- Registered sequence tuple: `{list(config.fixed_sequence)}`.
- Primary readout: `v=sigmoid(softplus(alpha)*logit(stopgrad(p))+b)`; `RT=1-v`.
- Execution objective retained from the frozen working readout run:
  `{config.execution_candidate.equation}`, `lambda={config.lambda_execution:g}`.
- Contexts: A `[1,0,0]`, B `[0,1,0]`, Mixed_norm `[.5,.5,0]`, Novel `[0,0,1]`.
- Human observations, RT targets, and probe training rows: `0`.

The PRD's written physical-key sequence differs from the repository's registered tuple.
The registered tuple was retained because the same PRD freezes the existing working run.

## 2. Did all conditions learn the sequence?

```text
{condition_training.to_string(index=False)}
```

## 3. Did representational scale control A/B splitting?

```text
{scale_comparison.to_string(index=False)}
```

Token-matched Euclidean distance is the primary measure; the shared PCA is visualization only.

## 4. Did representational scale change emergent RT dynamics?

Learning-related RT, switch costs, and frozen A/B endpoint differences are shown in the
condition table in Section 2 and at seed level in `confirmatory_seed_summary.csv`.

## 5. Which canonical state-space model described each condition?

Seed-averaged fits:

```text
{mean_winners.to_string(index=False)}
```

Seed-level classifications:

```text
{fractions.to_string(index=False)}
```

These are the unchanged no-fatigue Lump/Cross/Split equations from `state-space_SRN.Rmd`.

## 6. What happened on frozen Mixed and Novel probes?

```text
{probe_summary.to_string(index=False)}
```

Probe evaluation made zero optimizer steps and all before/after parameter hashes matched.

## 7. Is probe recombination scalar or transition structured?

```text
{q_summary.to_string(index=False)}
```

A/B-only leave-seed-and-transition-out rank selection, followed by transition-held-out
probe prediction:

```text
{selected_rank.to_string(index=False)}
```

Unclipped q values are retained; transitions with `|A_RT-B_RT| <
{config.responsibility_min_abs_denominator:g}` are marked unusable rather than clipped.

## 8. Do representational interventions causally shift RT?

```text
{causal_summary.to_string(index=False)}
```

Every matched intervention is accompanied by an orthogonal random direction, a wrong-token
A/B component, and an unrestricted magnitude-matched random vector. RT, confidence, CE,
prediction identity, direction alignment, and endpoint proximity are reported jointly.
Strong causal success additionally requires at least
`{config.sequence_preservation_threshold:.0%}` prediction-identity preservation and
correct-target accuracy, following the repository's established sequence-preservation rule.

## 9. Final interpretation

**{code}. {interpretation}.**

This statement follows the frozen analysis criteria and was not selected for manuscript fit.
"""
    (output / "CONFIRMATORY_SRN_REPORT.md").write_text(report, encoding="utf-8")


def run_confirmatory_pipeline(
    output_dir: str | Path, *, config: ConfirmatoryConfig,
) -> dict[str, pd.DataFrame]:
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "confirmatory_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    frozen_before = _hash_sources(repo)
    sequence_frames, transition_frames, audit_rows = [], [], []
    probe_frames, probe_audits, trained = [], [], {}
    total = len(config.conditions) * len(config.seeds)
    completed = 0
    _progress(config.show_progress, f"Starting {total}-network frozen confirmatory run")
    for condition in config.conditions:
        _progress(config.show_progress, f"Training {condition}")
        for seed in config.seeds:
            result = train_condition(
                condition, config, seed=seed,
                progress=(lambda message, name=condition: _progress(
                    config.show_progress, f"{name} | {message}"
                )),
            )
            if result.optimizer_steps != config.training_sequences:
                raise AssertionError("External training exposure changed.")
            trained[(condition, seed)] = result
            sequence_frames.append(result.sequence_level)
            transition_frames.append(result.transition_level)
            audit_rows.append(result.audit)
            torch.save(
                {**result.checkpoint, "config": asdict(config)},
                checkpoints / f"{condition}_seed_{seed}.pt",
            )
            probe, probe_audit = evaluate_frozen_probes(result, config)
            probe_frames.append(probe)
            probe_audits.append(probe_audit)
            completed += 1
            _progress(config.show_progress, f"Completed network {completed}/{total}")

    sequence = pd.concat(sequence_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True)
    probes = pd.concat(probe_frames, ignore_index=True)
    training_audit = pd.DataFrame(audit_rows)
    probe_audit = pd.DataFrame(probe_audits)
    if len(training_audit) != total or len(list(checkpoints.glob("*.pt"))) != total:
        raise AssertionError("Not every primary network/checkpoint was saved.")
    high = training_audit[training_audit.condition.eq(HIGH_MONOTONIC)].set_index("seed")
    sequence_only = training_audit[
        training_audit.condition.eq(HIGH_SEQUENCE_ONLY)
    ].set_index("seed")
    if not high.initial_model_sha256.equals(sequence_only.initial_model_sha256):
        raise AssertionError(
            "Matched High monotonic and sequence-only initializations diverged; "
            "the paired-seed contract was violated."
        )

    _progress(config.show_progress, "Computing switch and frozen representation diagnostics")
    switch_aligned, switch_costs = switch_diagnostics(sequence)
    separation, pca_scores, pca_components = hidden_separation_and_pca(probes, config)
    scale_comparison = hidden_scale_comparison(separation)
    fits, classifications, state_predictions = fit_confirmatory_state_space(sequence)
    midpoint, responsibility, bridge = probe_responsibility_tables(probes, config)
    low_rank_results, low_rank_predictions = low_rank_bridge(responsibility, config)
    seed_summary = _seed_summary(
        sequence, separation, switch_costs, classifications, midpoint
    )
    _progress(config.show_progress, "Evaluating frozen readout controls")
    readout_controls = evaluate_readout_controls(transitions, config)

    _progress(config.show_progress, "Running matched causal patches and specificity controls")
    patch_frames, control_frames, causal_audits = [], [], []
    for condition in (HIGH_MONOTONIC, LOW_MONOTONIC):
        for seed in config.seeds:
            result = trained[(condition, seed)]
            seed_probes = probes[
                probes.condition.eq(condition) & probes.seed.eq(seed)
            ]
            patching, controls, audit = causal_interventions(
                result, seed_probes, config
            )
            patch_frames.append(patching)
            control_frames.append(controls)
            causal_audits.append(audit)
    patching = pd.concat(patch_frames, ignore_index=True)
    causal_controls = pd.concat(control_frames, ignore_index=True)
    causal_audit = pd.DataFrame(causal_audits)

    hidden_columns = [f"hidden_state_{index}" for index in range(1, 9)]
    np.savez_compressed(
        output / "confirmatory_hidden_states.npz",
        hidden_states=probes[hidden_columns].to_numpy(dtype=np.float32),
        condition=probes.condition.to_numpy(dtype=str),
        seed=probes.seed.to_numpy(dtype=np.int64),
        context=probes.probe_context.to_numpy(dtype=str),
        transition_position=probes.transition_position.to_numpy(dtype=np.int64),
    )
    outputs = {
        "confirmatory_training_sequence_level": sequence,
        "confirmatory_training_transition_level": transitions,
        "confirmatory_seed_summary": seed_summary,
        "confirmatory_hidden_separation": separation,
        "confirmatory_hidden_pca_scores": pca_scores,
        "confirmatory_hidden_pca_components": pca_components,
        "confirmatory_hidden_scale_comparison": scale_comparison,
        "confirmatory_state_space_fits": fits,
        "confirmatory_state_space_seed_classification": classifications,
        "confirmatory_state_space_predictions": state_predictions,
        "confirmatory_probe_transition_level": probes,
        "confirmatory_probe_responsibility": responsibility,
        "confirmatory_probe_midpoint_deviation": midpoint,
        "confirmatory_low_rank_results": low_rank_results,
        "confirmatory_low_rank_predictions": low_rank_predictions,
        "confirmatory_causal_patching": patching,
        "confirmatory_causal_controls": causal_controls,
        "confirmatory_readout_controls": readout_controls,
        "confirmatory_switch_aligned": switch_aligned,
        "confirmatory_switch_costs": switch_costs,
        "confirmatory_training_audit": training_audit,
        "confirmatory_probe_audit": probe_audit,
        "confirmatory_causal_audit": causal_audit,
        "srn_human_bridge_table": bridge,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    frozen_after = _hash_sources(repo)
    if frozen_before != frozen_after:
        raise AssertionError("A frozen earlier analysis source changed during confirmation.")
    config_record = {
        **asdict(config),
        "analysis_version": "confirmatory-srn-1.0.0",
        "execution_candidate": asdict(config.execution_candidate),
        "execution_equation": config.execution_candidate.equation,
        "monotonic_readout": "sigmoid(softplus(alpha)*logit(stopgrad(p))+b)",
        "normalized_RT": "1-execution_speed",
        "canonical_seed_source": "repository existing 0--19 seed list",
        "sequence_source": "ftt_srn_execution_v2.config.FIXED_SEQUENCE",
        "PRD_sequence_label_discrepancy_recorded": True,
        "probe_contexts_used_for_training": [],
        "human_rows_used": 0,
        "all_checkpoints_saved": True,
        "frozen_source_hashes_before": frozen_before,
        "frozen_source_hashes_after": frozen_after,
        "required_outputs": list(REQUIRED_OUTPUTS),
    }
    (output / "confirmatory_config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        output, config, sequence, seed_summary, scale_comparison, fits,
        classifications, midpoint, responsibility, low_rank_results,
        patching, causal_controls,
    )
    if config.make_plots:
        _progress(config.show_progress, "Rendering confirmatory figures")
        make_figures(
            output / "figures_confirmatory_srn", sequence=sequence,
            separation=separation, pca=pca_scores, state_fits=fits,
            state_classification=classifications, state_predictions=state_predictions,
            probes=probes, responsibility=responsibility,
            low_rank_results=low_rank_results,
            low_rank_predictions=low_rank_predictions, patching=patching,
            controls=causal_controls, switch_aligned=switch_aligned,
        )
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Required confirmatory outputs missing: {missing}")
    _progress(config.show_progress, "Confirmatory SRN pipeline complete")
    return outputs
