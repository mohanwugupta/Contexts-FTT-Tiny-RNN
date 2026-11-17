"""Gate-controlled Stage 0--5 structured-context ecology pipeline."""

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
    baseline_bundle, evaluate_gate, factorization_diagnostics, probe_geometry,
    representation_diagnostics, responsibility_table, state_space_tables,
    training_summary,
)
from .config import (
    CONDITIONS, EcologyConfig, HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY, LOW_MONOTONIC,
)
from .firewall import StageFirewall
from .probes import evaluate_frozen_contexts
from .training import EcologyTrainingResult, train_ecology_condition


REQUIRED_OUTPUTS = (
    "structured_context_config.json",
    "stage0_baseline_summary.csv",
    "stage1_shared_item_training.csv",
    "stage1_shared_item_representation.csv",
    "stage1_shared_item_state_space.csv",
    "stage2_structured_training.csv",
    "stage2_structured_representation.csv",
    "stage2_factorization_diagnostics.csv",
    "stage2_state_space.csv",
    "stage3_mixed_probe.csv",
    "stage3_mixed_geometry.csv",
    "stage3_responsibility.csv",
    "stage4_salience_sweep.csv",
    "stage4_salience_representation.csv",
    "stage4_salience_mixed.csv",
    "stage5_novel_probe.csv",
    "stage5_novel_geometry.csv",
    "structured_context_stage_summary.csv",
    "STRUCTURED_CONTEXT_ECOLOGY_REPORT.md",
)


TABLE_NAMES = tuple(
    name.removesuffix(".csv") for name in REQUIRED_OUTPUTS if name.endswith(".csv")
)


def _progress(config: EcologyConfig, message: str):
    if config.show_progress:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] STRUCTURED-ECOLOGY | {message}", flush=True)


def _source_hashes(repo: Path) -> dict[str, str]:
    result = {}
    for directory in ("confirmatory_srn", "ftt_srn_execution_v2"):
        for path in sorted((repo / directory).rglob("*.py")):
            result[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def _relabel(frame: pd.DataFrame, *, stage: str, cohort: str | None = None,
             sigma: float | None = None) -> pd.DataFrame:
    result = frame.copy()
    result["stage"] = stage
    if cohort is not None:
        result["cohort"] = cohort
    if sigma is not None:
        result["sigma_context"] = float(sigma)
    return result


def _train_cohort(
    output: Path,
    stage: str,
    sigma: float,
    cohort: str,
    seeds: tuple[int, ...],
    config: EcologyConfig,
) -> dict:
    models: dict[tuple[str, int], EcologyTrainingResult] = {}
    sequence_parts, transition_parts, probe_parts, audits, probe_audits = [], [], [], [], []
    checkpoint_dir = output / "structured_context_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total = len(CONDITIONS) * len(seeds)
    completed = 0
    for condition in CONDITIONS:
        for seed in seeds:
            result = train_ecology_condition(
                stage, condition, config, seed=seed, sigma_context=sigma,
                cohort=cohort,
                progress=lambda text, c=condition: _progress(config, f"{stage} | {c} | {text}"),
            )
            models[(condition, seed)] = result
            sequence_parts.append(result.sequence_level)
            transition_parts.append(result.transition_level)
            audits.append(result.audit)
            probe, probe_audit = evaluate_frozen_contexts(
                result, config, contexts=("A", "B")
            )
            probe_parts.append(probe)
            probe_audits.append(probe_audit)
            sigma_label = f"{sigma:g}".replace(".", "p")
            name = f"{stage}_{cohort}_sigma_{sigma_label}_{condition}_seed_{seed}.pt"
            torch.save(
                {**result.checkpoint, "config": asdict(config)}, checkpoint_dir / name
            )
            completed += 1
            _progress(config, f"{stage} {cohort}: completed {completed}/{total}")
    audit = pd.DataFrame(audits)
    high = audit[audit.condition.eq(HIGH_MONOTONIC)].set_index("seed")
    seq = audit[audit.condition.eq(HIGH_SEQUENCE_ONLY)].set_index("seed")
    if not high.initial_model_sha256.equals(seq.initial_model_sha256):
        raise AssertionError("Paired High execution/sequence-only initialization diverged.")
    sequence = _concat(sequence_parts)
    probes = _concat(probe_parts)
    representation = representation_diagnostics(probes)
    fits, classifications, predictions = state_space_tables(sequence)
    return {
        "models": models,
        "sequence": sequence,
        "transitions": _concat(transition_parts),
        "probes": probes,
        "representation": representation,
        "factorization": factorization_diagnostics(probes),
        "fits": fits,
        "classifications": classifications,
        "predictions": predictions,
        "training_summary": training_summary(sequence),
        "audit": audit,
        "probe_audit": pd.DataFrame(probe_audits),
    }


def _baseline_gate(baseline: dict, config: EcologyConfig) -> dict:
    summary = baseline["summary"]
    pivot = summary.pivot(index="seed", columns="condition",
                          values="aggregate_AB_separation").dropna()
    high_specific = int(summary[
        summary.condition.eq(HIGH_MONOTONIC)
    ].model_label.isin(["Cross", "Split"]).sum())
    low_lump = int(summary[
        summary.condition.eq(LOW_MONOTONIC)
    ].model_label.eq("Lump").sum())
    n = int(summary.seed.nunique())
    required = int(np.ceil(config.gate_fraction * n))
    learning = all(
        int(summary[summary.condition.eq(condition)].CE_decreased.sum()) >= required
        for condition in CONDITIONS
    )
    rt = all(
        int(summary[summary.condition.eq(condition)].RT_decreased.eq(True).sum()) >= required
        for condition in (HIGH_MONOTONIC, LOW_MONOTONIC)
    )
    high_greater = int((pivot[HIGH_MONOTONIC] > pivot[LOW_MONOTONIC]).sum())
    passed = bool(
        learning and rt and high_greater >= required and low_lump >= required
        and high_specific > n - low_lump
    )
    return {
        "stage": "stage0_categorical", "cohort": "reference", "sigma_context": 0.0,
        "n_seeds": n, "required_passing_seeds": required,
        "G1_sequence_learning": learning, "G2_high_greater_low": high_greater >= required,
        "G3_nontrivial_high_splitting": True,
        "G4_state_space_behavior": low_lump >= required and high_specific > n - low_lump,
        "G5_RT_sanity": rt, "gate_passed": passed,
        "high_greater_low_seed_count": high_greater,
        "mean_high_low_separation_difference": baseline["stage0_paired_difference"],
        "stage0_difference_retained_fraction": 1.0,
        "low_lump_seed_count": low_lump,
        "high_cross_split_seed_count": high_specific,
        "low_cross_split_seed_count": n - low_lump,
        "high_RT_decrease_seed_count": int(summary[
            summary.condition.eq(HIGH_MONOTONIC)
        ].RT_decreased.eq(True).sum()),
        "low_RT_decrease_seed_count": int(summary[
            summary.condition.eq(LOW_MONOTONIC)
        ].RT_decreased.eq(True).sum()),
        "high_negative_confidence_RT_slope_count": np.nan,
        "low_negative_confidence_RT_slope_count": np.nan,
        "failure_reason": "" if passed else "frozen_baseline_regression",
    }


def _gate(batch: dict, stage: str, cohort: str, baseline: dict,
          config: EcologyConfig) -> dict:
    return evaluate_gate(
        stage, cohort, batch["training_summary"], batch["representation"],
        batch["classifications"],
        stage0_paired_difference=float(baseline["stage0_paired_difference"]),
        config=config,
    )


def _probe_models(batch: dict, config: EcologyConfig, contexts: tuple[str, ...]):
    frames, audits = [], []
    for result in batch["models"].values():
        frame, audit = evaluate_frozen_contexts(result, config, contexts=contexts)
        frames.append(frame)
        audits.append(audit)
    return _concat(frames), pd.DataFrame(audits)


def _mixed_metrics(responsibility: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    if responsibility.empty or geometry.empty:
        return pd.DataFrame(columns=[
            "stage", "cohort", "sigma_context", "condition", "seed",
            "usable_mixed_transition_count", "mixed_between_endpoint_fraction",
            "mean_mixed_segment_distance", "mean_mixed_manifold_distance",
        ])
    keys = ["stage", "cohort", "sigma_context", "condition", "seed"]
    usable = responsibility[responsibility.responsibility_denominator_usable]
    behavior = usable.groupby(keys, as_index=False).agg(
        usable_mixed_transition_count=("transition_position", "size"),
        mixed_between_endpoint_fraction=("probe_between_A_B", "mean"),
    )
    geometric = geometry.groupby(keys, as_index=False).agg(
        mean_mixed_segment_distance=("distance_to_AB_segment", "mean"),
        mean_mixed_manifold_distance=("distance_to_nearest_trained_state", "mean"),
    )
    return behavior.merge(geometric, on=keys, how="outer")


def _combine_responsibility_geometry(responsibility, geometry):
    keys = [
        "stage", "cohort", "sigma_context", "condition", "seed",
        "probe_context", "transition_position",
    ]
    return responsibility.merge(geometry, on=keys, how="outer")


def _write_report(output: Path, tables: dict[str, pd.DataFrame], stage_rows: list[dict],
                  selected_sigma, selected_sigma_confirmed: bool, blocked_by: str):
    stages = pd.DataFrame(stage_rows)
    baseline = tables["stage0_baseline_summary"]
    stage1 = tables["stage1_shared_item_representation"]
    stage2 = tables["stage2_structured_representation"]
    stage3 = tables["stage3_responsibility"]
    stage4 = tables["stage4_salience_sweep"]
    stage5 = tables["stage5_novel_geometry"]
    factorization = tables["stage2_factorization_diagnostics"]
    novel_probe = tables["stage5_novel_probe"]

    def aggregate_rep(frame):
        if frame.empty or "row_type" not in frame:
            return "Not run because of the stage firewall."
        value = frame[frame.row_type.eq("seed_aggregate")].groupby(
            ["cohort", "condition"], as_index=False
        ).aggregate_AB_separation.mean()
        return value.to_string(index=False)

    stage2_improved = False
    if not stage3.empty and "stage" in stage3:
        summary = stage3[
            stage3.condition.eq(HIGH_MONOTONIC)
            & stage3.responsibility_denominator_usable.eq(True)
        ].groupby(
            "stage", as_index=False
        ).agg(
            between=("probe_between_A_B", "mean"),
            segment=("distance_to_AB_segment", "mean"),
        )
        if {"stage0_categorical", "stage2_structured"}.issubset(summary.stage):
            indexed = summary.set_index("stage")
            stage2_improved = bool(
                indexed.loc["stage2_structured", "between"]
                > indexed.loc["stage0_categorical", "between"]
                and indexed.loc["stage2_structured", "segment"]
                < indexed.loc["stage0_categorical", "segment"]
            )
    if not stage4.empty:
        salience_report = stage4.groupby(
            ["cohort", "sigma_context", "condition"], as_index=False
        ).agg(
            seeds=("seed", "nunique"), CE_change=("CE_change", "mean"),
            RT_change=("RT_change", "mean"),
            AB_separation=("aggregate_AB_separation", "mean"),
            Lump_fraction=("model_label", lambda x: float(x.eq("Lump").mean())),
            Mixed_between_fraction=("mixed_between_endpoint_fraction", "mean"),
            Mixed_segment_distance=("mean_mixed_segment_distance", "mean"),
            sigma_gate_passed=("sigma_gate_passed", "first"),
            selected_for_confirmation=("selected_for_confirmation", "first"),
        ).to_string(index=False)
    else:
        salience_report = "Not run because of the stage firewall."
    if not stage3.empty and "responsibility_denominator_usable" in stage3:
        mixed_report = stage3[
            stage3.condition.eq(HIGH_MONOTONIC)
            & stage3.responsibility_denominator_usable.eq(True)
        ].groupby("stage", as_index=False).agg(
            usable_transitions=("transition_position", "size"),
            between_endpoint_fraction=("probe_between_A_B", "mean"),
            midpoint_deviation=("probe_midpoint_deviation", "mean"),
            global_q=("global_q_unclipped", "mean"),
            segment_distance=("distance_to_AB_segment", "mean"),
            manifold_distance=("distance_to_nearest_trained_state", "mean"),
        ).to_string(index=False)
    else:
        mixed_report = "Not run because of the stage firewall."
    if not factorization.empty:
        factorization_report = factorization.groupby(
            ["stage", "cohort", "condition"], as_index=False
        ).agg(
            matched_AB_cosine=("matched_position_AB_cosine", "mean"),
            cross_position_cosine=("cross_position_AB_cosine", "mean"),
            displacement_consistency=("context_displacement_pairwise_cosine", "mean"),
            additive_RMSE=("additive_position_context_RMSE", "mean"),
            additive_variance_explained=(
                "additive_position_context_variance_explained", "mean"
            ),
        ).to_string(index=False)
    else:
        factorization_report = "Not run because of the stage firewall."
    if not novel_probe.empty and "probe_context" in novel_probe:
        novel_report = novel_probe[
            novel_probe.probe_context.eq("Novel")
            & novel_probe.condition.isin([HIGH_MONOTONIC, LOW_MONOTONIC])
        ].groupby(["stage", "condition"], as_index=False).agg(
            normalized_RT=("normalized_RT", "mean"),
            sequence_CE=("sequence_CE", "mean"),
            next_token_accuracy=("next_token_correct", "mean"),
        ).merge(
            stage5.groupby(["stage", "condition"], as_index=False).agg(
                segment_distance=("distance_to_AB_segment", "mean"),
                manifold_distance=("distance_to_nearest_trained_state", "mean"),
            ), on=["stage", "condition"], how="left",
        ).to_string(index=False)
    else:
        novel_report = "Not run because of the stage firewall."
    stage2_gate = any(
        row["stage"] == "stage2_structured" and row["cohort"] == "confirmatory"
        and row["gate_passed"] for row in stage_rows
    )
    if stage2_improved:
        code = "A. shared structured context is sufficient"
    elif selected_sigma_confirmed:
        code = "B. structured context + salience variability is required"
    elif stage2_gate:
        code = "C. splitting and recombination remain in tension"
    else:
        code = "D. structured environmental overlap does not solve the problem"

    report = f"""# Structured context ecology report

## 1. Baseline replication

The frozen categorical baseline was loaded without retraining or modification.

```text
{baseline.groupby('condition', as_index=False).agg(seeds=('seed','nunique'), CE_late=('CE_late','mean'), RT_late=('RT_late','mean'), AB_separation=('aggregate_AB_separation','mean')).to_string(index=False)}
```

## 2. Shared-item context

Stage 1 screening and confirmatory gates:

```text
{stages[stages.stage.eq('stage1_shared_item')].to_string(index=False)}
```

Representational separation:

```text
{aggregate_rep(stage1)}
```

## 3. Position-structured context

Stage 2 gates:

```text
{stages[stages.stage.eq('stage2_structured')].to_string(index=False)}
```

Representational separation:

```text
{aggregate_rep(stage2)}
```

Shared-position similarity, displacement consistency, and additive factorization:

```text
{factorization_report}
```

## 4. Mixed probe

Mixed was evaluated only after the Stage 2 confirmatory gate passed. The saved table
reports unclipped q, midpoint deviation, between-endpoint status, A/B-segment distance,
and nearest-trained-state distance. Structured-context improvement on both prespecified
behavioral and geometric dimensions: `{stage2_improved}`.

```text
{mixed_report}
```

## 5. Salience variability

Selected sigma: `{selected_sigma}`. Confirmatory recombination improvement replicated:
`{selected_sigma_confirmed}`. Selection used only the mechanistic rule frozen in
`structured_context_config.json`, never human RT.

```text
{salience_report}
```

## 6. Novel probe

Novel was evaluated last using the shared S scaffold plus position-matched N1--N3 items.
The categorical unused-axis probe is retained as the direct comparison.

```text
{novel_report}
```

## 7. Final interpretation

**{code}.**

Blocked after: `{blocked_by or 'none'}`. This interpretation follows the fixed gates and
mechanistic selection rule; no human observations were loaded or fitted.
"""
    (output / "STRUCTURED_CONTEXT_ECOLOGY_REPORT.md").write_text(report, encoding="utf-8")


def run_structured_context_ecology(
    output_dir: str | Path, *, config: EcologyConfig,
) -> dict[str, pd.DataFrame]:
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    frozen_before = _source_hashes(repo)
    baseline = baseline_bundle(repo, config.baseline_output)
    firewall = StageFirewall()
    stage_rows: list[dict] = []
    tables = {name: pd.DataFrame() for name in TABLE_NAMES}
    audits, probe_audits = [], []
    predictions_for_figures, fits_for_figures = [], []
    mixed_probes_for_figures = []
    blocked_by = ""
    selected_sigma = None
    selected_sigma_confirmed = False

    tables["stage0_baseline_summary"] = baseline["summary"]
    stage0 = _baseline_gate(baseline, config)
    stage_rows.append(stage0)
    firewall.record("stage0_baseline", stage0["gate_passed"])
    _progress(config, f"Stage 0 baseline gate: {stage0['gate_passed']}")

    stage1_screen = stage1_confirm = stage2_screen = stage2_confirm = None
    stage3_probe = stage3_geometry = stage3_responsibility = pd.DataFrame()
    stage4_batches = {}
    selected_confirm = None

    if firewall.may_run("stage1_screen"):
        _progress(config, "Stage 1: five-seed shared-item screen")
        stage1_screen = _train_cohort(
            output, "stage1_shared_item", 0.0, "screen", config.screen_seeds, config
        )
        audits.append(stage1_screen["audit"]); probe_audits.append(stage1_screen["probe_audit"])
        gate = _gate(stage1_screen, "stage1_shared_item", "screen", baseline, config)
        stage_rows.append(gate); firewall.record("stage1_screen", gate["gate_passed"])
        if not gate["gate_passed"]:
            blocked_by = "stage1_screen: " + gate["failure_reason"]
    else:
        blocked_by = "stage0_baseline"

    if not blocked_by and firewall.may_run("stage1_confirmatory"):
        _progress(config, "Stage 1 passed; running 20-seed confirmation")
        stage1_confirm = _train_cohort(
            output, "stage1_shared_item", 0.0, "confirmatory",
            config.confirmatory_seeds, config,
        )
        audits.append(stage1_confirm["audit"]); probe_audits.append(stage1_confirm["probe_audit"])
        gate = _gate(stage1_confirm, "stage1_shared_item", "confirmatory", baseline, config)
        stage_rows.append(gate); firewall.record("stage1_confirmatory", gate["gate_passed"])
        if not gate["gate_passed"]:
            blocked_by = "stage1_confirmatory: " + gate["failure_reason"]

    for batch in (stage1_screen, stage1_confirm):
        if batch:
            tables["stage1_shared_item_training"] = _concat([
                tables["stage1_shared_item_training"], batch["sequence"]
            ])
            tables["stage1_shared_item_representation"] = _concat([
                tables["stage1_shared_item_representation"], batch["representation"]
            ])
            tables["stage1_shared_item_state_space"] = _concat([
                tables["stage1_shared_item_state_space"], batch["fits"]
            ])
            fits_for_figures.append(batch["fits"]); predictions_for_figures.append(batch["predictions"])

    if not blocked_by and firewall.may_run("stage2_screen"):
        _progress(config, "Stage 2: five-seed ordinal structured-context screen")
        stage2_screen = _train_cohort(
            output, "stage2_structured", 0.0, "screen", config.screen_seeds, config
        )
        audits.append(stage2_screen["audit"]); probe_audits.append(stage2_screen["probe_audit"])
        gate = _gate(stage2_screen, "stage2_structured", "screen", baseline, config)
        stage_rows.append(gate); firewall.record("stage2_screen", gate["gate_passed"])
        if not gate["gate_passed"]:
            blocked_by = "stage2_screen: " + gate["failure_reason"]

    if not blocked_by and firewall.may_run("stage2_confirmatory"):
        _progress(config, "Stage 2 passed; running 20-seed confirmation")
        stage2_confirm = _train_cohort(
            output, "stage2_structured", 0.0, "confirmatory",
            config.confirmatory_seeds, config,
        )
        audits.append(stage2_confirm["audit"]); probe_audits.append(stage2_confirm["probe_audit"])
        gate = _gate(stage2_confirm, "stage2_structured", "confirmatory", baseline, config)
        stage_rows.append(gate); firewall.record("stage2_confirmatory", gate["gate_passed"])
        if not gate["gate_passed"]:
            blocked_by = "stage2_confirmatory: " + gate["failure_reason"]

    factor_parts = []
    baseline_factor = factorization_diagnostics(baseline["probes"])
    factor_parts.append(baseline_factor)
    for batch in (stage2_screen, stage2_confirm):
        if batch:
            tables["stage2_structured_training"] = _concat([
                tables["stage2_structured_training"], batch["sequence"]
            ])
            tables["stage2_structured_representation"] = _concat([
                tables["stage2_structured_representation"], batch["representation"]
            ])
            tables["stage2_state_space"] = _concat([
                tables["stage2_state_space"], batch["fits"]
            ])
            factor_parts.append(batch["factorization"])
            fits_for_figures.append(batch["fits"]); predictions_for_figures.append(batch["predictions"])
    tables["stage2_factorization_diagnostics"] = _concat(factor_parts)

    if not blocked_by and firewall.may_run("stage3_mixed"):
        _progress(config, "Stage 3: frozen familiar-item Mixed probe")
        structured_probe, audit = _probe_models(
            stage2_confirm, config, ("A", "B", "Mixed")
        )
        probe_audits.append(audit)
        categorical_probe = baseline["probes"][
            baseline["probes"].probe_context.isin(["A", "B", "Mixed"])
        ].copy()
        stage3_probe = _concat([categorical_probe, structured_probe])
        stage3_geometry = _concat([
            probe_geometry(categorical_probe, "Mixed"),
            probe_geometry(structured_probe, "Mixed"),
        ])
        stage3_responsibility = _concat([
            responsibility_table(categorical_probe, "Mixed", config),
            responsibility_table(structured_probe, "Mixed", config),
        ])
        tables["stage3_mixed_probe"] = stage3_probe
        tables["stage3_mixed_geometry"] = stage3_geometry
        tables["stage3_responsibility"] = _combine_responsibility_geometry(
            stage3_responsibility, stage3_geometry
        )
        mixed_probes_for_figures.append(stage3_probe)
        firewall.record("stage3_mixed", True)

    if not blocked_by and firewall.may_run("stage4_salience"):
        _progress(config, "Stage 4: contextual salience screen")
        # Sigma 0 is exactly Stage 2 and is reused rather than redundantly retrained.
        stage4_batches[0.0] = stage2_screen
        salience_representations, salience_mixed_parts, sweep_parts = [], [], []
        zero_screen_probe, zero_audit = _probe_models(
            stage2_screen, config, ("A", "B", "Mixed")
        )
        probe_audits.append(zero_audit)
        mixed_by_sigma = {0.0: _relabel(
            zero_screen_probe, stage="stage4_salience", cohort="screen", sigma=0.0
        )}
        gate_by_sigma = {}
        for sigma in config.salience_sigmas:
            if sigma > 0:
                batch = _train_cohort(
                    output, "stage4_salience", sigma, "screen",
                    config.screen_seeds, config,
                )
                stage4_batches[sigma] = batch
                audits.append(batch["audit"]); probe_audits.append(batch["probe_audit"])
            else:
                batch = stage2_screen
            summary = _relabel(batch["training_summary"], stage="stage4_salience", sigma=sigma)
            representation = _relabel(
                batch["representation"], stage="stage4_salience", sigma=sigma
            )
            classification = _relabel(
                batch["classifications"], stage="stage4_salience", sigma=sigma
            )
            gate = evaluate_gate(
                "stage4_salience", "screen", summary, representation, classification,
                stage0_paired_difference=float(baseline["stage0_paired_difference"]),
                config=config,
            )
            gate_by_sigma[sigma] = gate
            stage_rows.append(gate)
            salience_representations.append(representation)
            fits_for_figures.append(_relabel(batch["fits"], stage="stage4_salience", sigma=sigma))
            predictions_for_figures.append(_relabel(
                batch["predictions"], stage="stage4_salience", sigma=sigma
            ))
            if gate["gate_passed"]:
                if sigma not in mixed_by_sigma:
                    probe, audit = _probe_models(batch, config, ("A", "B", "Mixed"))
                    probe_audits.append(audit)
                    mixed_by_sigma[sigma] = probe
                probe = mixed_by_sigma[sigma]
                responsibility = responsibility_table(probe, "Mixed", config)
                geometry = probe_geometry(probe, "Mixed")
                combined = _combine_responsibility_geometry(responsibility, geometry)
                salience_mixed_parts.append(combined)
                metric = _mixed_metrics(responsibility, geometry)
                high_metric = metric[metric.condition.eq(HIGH_MONOTONIC)]
                gate["mixed_between_endpoint_fraction"] = float(
                    high_metric.mixed_between_endpoint_fraction.mean()
                )
                gate["mixed_segment_distance"] = float(
                    high_metric.mean_mixed_segment_distance.mean()
                )
            else:
                metric = pd.DataFrame()
                gate["mixed_between_endpoint_fraction"] = np.nan
                gate["mixed_segment_distance"] = np.nan
            per_seed = summary.merge(
                representation[representation.row_type.eq("seed_aggregate")][[
                    "stage", "cohort", "sigma_context", "condition", "seed",
                    "aggregate_AB_separation",
                ]], on=["stage", "cohort", "sigma_context", "condition", "seed"],
                how="left",
            )
            winners = classification[classification.row_type.eq("seed")][[
                "condition", "seed", "model_label"
            ]]
            per_seed = per_seed.merge(winners, on=["condition", "seed"], how="left")
            if not metric.empty:
                per_seed = per_seed.merge(metric, on=[
                    "stage", "cohort", "sigma_context", "condition", "seed"
                ], how="left")
            per_seed["sigma_gate_passed"] = gate["gate_passed"]
            per_seed["sigma_failure_reason"] = gate["failure_reason"]
            sweep_parts.append(per_seed)

        zero_resp = responsibility_table(mixed_by_sigma[0.0], "Mixed", config)
        zero_geom = probe_geometry(mixed_by_sigma[0.0], "Mixed")
        zero_metric = _mixed_metrics(zero_resp, zero_geom)
        zero_high = zero_metric[zero_metric.condition.eq(HIGH_MONOTONIC)]
        zero_between = float(zero_high.mixed_between_endpoint_fraction.mean())
        zero_distance = float(zero_high.mean_mixed_segment_distance.mean())
        eligible = []
        for sigma in config.salience_sigmas[1:]:
            if not gate_by_sigma[sigma]["gate_passed"] or sigma not in mixed_by_sigma:
                continue
            probe = mixed_by_sigma[sigma]
            metric = _mixed_metrics(
                responsibility_table(probe, "Mixed", config),
                probe_geometry(probe, "Mixed"),
            )
            high = metric[metric.condition.eq(HIGH_MONOTONIC)]
            between = float(high.mixed_between_endpoint_fraction.mean())
            distance = float(high.mean_mixed_segment_distance.mean())
            if between > zero_between and distance < zero_distance:
                eligible.append(sigma)
            gate_by_sigma[sigma]["mechanistic_improvement_vs_sigma0"] = bool(
                between > zero_between and distance < zero_distance
            )
        gate_by_sigma[0.0]["mechanistic_improvement_vs_sigma0"] = False
        selected_sigma = min(eligible) if eligible else None

        if selected_sigma is not None:
            _progress(config, f"Stage 4 selected sigma={selected_sigma:g}; confirming 20 seeds")
            selected_confirm = _train_cohort(
                output, "stage4_salience", selected_sigma, "confirmatory",
                config.confirmatory_seeds, config,
            )
            audits.append(selected_confirm["audit"]); probe_audits.append(selected_confirm["probe_audit"])
            confirm_gate = _gate(
                selected_confirm, "stage4_salience", "confirmatory", baseline, config
            )
            stage_rows.append(confirm_gate)
            if confirm_gate["gate_passed"]:
                probe, audit = _probe_models(
                    selected_confirm, config, ("A", "B", "Mixed")
                )
                probe_audits.append(audit)
                responsibility = responsibility_table(probe, "Mixed", config)
                geometry = probe_geometry(probe, "Mixed")
                salience_mixed_parts.append(
                    _combine_responsibility_geometry(responsibility, geometry)
                )
                confirm_metric = _mixed_metrics(responsibility, geometry)
                confirm_high = confirm_metric[
                    confirm_metric.condition.eq(HIGH_MONOTONIC)
                ]
                stage2_probe_for_comparison, comparison_audit = _probe_models(
                    stage2_confirm, config, ("A", "B", "Mixed")
                )
                probe_audits.append(comparison_audit)
                stage2_metric = _mixed_metrics(
                    responsibility_table(stage2_probe_for_comparison, "Mixed", config),
                    probe_geometry(stage2_probe_for_comparison, "Mixed"),
                )
                stage2_high = stage2_metric[
                    stage2_metric.condition.eq(HIGH_MONOTONIC)
                ]
                confirm_between = float(
                    confirm_high.mixed_between_endpoint_fraction.mean()
                )
                confirm_distance = float(
                    confirm_high.mean_mixed_segment_distance.mean()
                )
                reference_between = float(
                    stage2_high.mixed_between_endpoint_fraction.mean()
                )
                reference_distance = float(
                    stage2_high.mean_mixed_segment_distance.mean()
                )
                selected_sigma_confirmed = bool(
                    confirm_between > reference_between
                    and confirm_distance < reference_distance
                )
                confirm_gate["mixed_between_endpoint_fraction"] = confirm_between
                confirm_gate["mixed_segment_distance"] = confirm_distance
                confirm_gate["mechanistic_improvement_vs_sigma0"] = selected_sigma_confirmed
                salience_representations.append(selected_confirm["representation"])
                fits_for_figures.append(selected_confirm["fits"])
                predictions_for_figures.append(selected_confirm["predictions"])
                mixed_probes_for_figures.append(probe)
                confirm_summary = selected_confirm["training_summary"].merge(
                    selected_confirm["representation"][
                        selected_confirm["representation"].row_type.eq("seed_aggregate")
                    ][[
                        "stage", "cohort", "sigma_context", "condition", "seed",
                        "aggregate_AB_separation",
                    ]], on=["stage", "cohort", "sigma_context", "condition", "seed"],
                    how="left",
                )
                confirm_winners = selected_confirm["classifications"][
                    selected_confirm["classifications"].row_type.eq("seed")
                ][["condition", "seed", "model_label"]]
                confirm_summary = confirm_summary.merge(
                    confirm_winners, on=["condition", "seed"], how="left"
                ).merge(confirm_metric, on=[
                    "stage", "cohort", "sigma_context", "condition", "seed"
                ], how="left")
                confirm_summary["sigma_gate_passed"] = True
                confirm_summary["sigma_failure_reason"] = ""
                sweep_parts.append(confirm_summary)
            else:
                selected_sigma = None
        salience_sweep = _concat(sweep_parts)
        salience_sweep["selected_for_confirmation"] = (
            salience_sweep.sigma_context.eq(selected_sigma)
            if selected_sigma is not None else False
        )
        salience_sweep["selection_rule"] = config.salience_selection_rule
        tables["stage4_salience_sweep"] = salience_sweep
        tables["stage4_salience_representation"] = _concat(salience_representations)
        tables["stage4_salience_mixed"] = _concat(salience_mixed_parts)
        firewall.record("stage4_salience", True)

    if not blocked_by and firewall.may_run("stage5_novel"):
        _progress(config, "Stage 5: structured Novel probe")
        novel_parts, geometry_parts = [], []
        categorical = baseline["probes"][
            baseline["probes"].probe_context.isin(["A", "B", "Novel"])
        ].copy()
        categorical = _relabel(categorical, stage="stage5_novel_categorical")
        categorical["training_ecology"] = "categorical_unused_axis"
        novel_parts.append(categorical)
        geometry_parts.append(probe_geometry(categorical, "Novel"))

        structured, audit = _probe_models(stage2_confirm, config, ("A", "B", "Novel"))
        probe_audits.append(audit)
        structured = _relabel(structured, stage="stage5_novel_structured")
        structured["training_ecology"] = "position_structured_sigma0"
        novel_parts.append(structured)
        geometry_parts.append(probe_geometry(structured, "Novel"))
        if selected_confirm is not None and selected_sigma is not None:
            salience, audit = _probe_models(
                selected_confirm, config, ("A", "B", "Novel")
            )
            probe_audits.append(audit)
            salience = _relabel(salience, stage="stage5_novel_salience")
            salience["training_ecology"] = f"position_structured_sigma{selected_sigma:g}"
            novel_parts.append(salience)
            geometry_parts.append(probe_geometry(salience, "Novel"))
        tables["stage5_novel_probe"] = _concat(novel_parts)
        tables["stage5_novel_geometry"] = _concat(geometry_parts)
        firewall.record("stage5_novel", True)

    tables["structured_context_stage_summary"] = pd.DataFrame(stage_rows)
    tables["structured_context_training_audit"] = _concat(audits)
    tables["structured_context_probe_audit"] = _concat(probe_audits)
    all_ecology_fits = _concat(fits_for_figures)
    all_ecology_predictions = _concat(predictions_for_figures)
    tables["stage4_salience_state_space"] = (
        all_ecology_fits[all_ecology_fits.stage.eq("stage4_salience")].copy()
        if not all_ecology_fits.empty else pd.DataFrame()
    )
    tables["structured_context_state_space_predictions"] = all_ecology_predictions
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    _write_report(
        output, tables, stage_rows, selected_sigma,
        selected_sigma_confirmed, blocked_by,
    )
    if config.make_plots:
        from .plotting import make_figures
        _progress(config, "Rendering six ecology figures")
        make_figures(
            output / "figures_structured_context_ecology",
            baseline=baseline,
            stage_summary=tables["structured_context_stage_summary"],
            stage1_representation=tables["stage1_shared_item_representation"],
            stage2_representation=tables["stage2_structured_representation"],
            salience_representation=tables["stage4_salience_representation"],
            fits=all_ecology_fits, predictions=all_ecology_predictions,
            mixed_probe=_concat(mixed_probes_for_figures),
            stage3_responsibility=tables["stage3_responsibility"],
            salience_mixed=tables["stage4_salience_mixed"],
        )

    frozen_after = _source_hashes(repo)
    if frozen_before != frozen_after:
        raise AssertionError("A frozen predecessor source changed during the ecology run.")
    config_record = {
        **asdict(config),
        "analysis_version": "structured-context-ecology-1.0.0",
        "trained_stage2_context_channels": ["S", "A1", "A2", "A3", "B1", "B2", "B3"],
        "reserved_stage5_channels": ["N1", "N2", "N3"],
        "reserved_stage5_weights_frozen_zero_during_training": True,
        "selected_salience_sigma": selected_sigma,
        "selected_salience_sigma_confirmed": selected_sigma_confirmed,
        "blocked_by": blocked_by or None,
        "human_rows_used": 0,
        "stage_firewall_outcomes": firewall.outcomes,
        "stage_firewall_attempted": firewall.attempted,
        "frozen_source_hashes_before": frozen_before,
        "frozen_source_hashes_after": frozen_after,
        "required_outputs": list(REQUIRED_OUTPUTS),
    }
    (output / "structured_context_config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Required ecology outputs missing: {missing}")
    _progress(config, f"Ecology complete; blocked_by={blocked_by or 'none'}")
    return tables
