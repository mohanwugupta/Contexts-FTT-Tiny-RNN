"""Hard-gated ordinal scaffold plus spatial-layout ecology pipeline."""

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
    baseline_bundle,
    baseline_gate,
    combine_responsibility_geometry,
    evaluate_gate,
    factorization_diagnostics,
    mixed_metrics,
    probe_geometry,
    representation_diagnostics,
    responsibility_table,
    state_space_tables,
    switch_effects,
    training_summary,
)
from .config import (
    CONDITIONS, HIGH_MONOTONIC, HIGH_SEQUENCE_ONLY, LOW_MONOTONIC,
    LayoutEcologyConfig,
)
from .firewall import OrdinalLayoutFirewall
from .inputs import GLOBAL_CUES, feature_dim, layout_metadata
from .probes import evaluate_frozen_layouts
from .training import LayoutTrainingResult, train_layout_condition


REQUIRED_OUTPUTS = (
    "ordinal_layout_config.json",
    "stage0_reference_summary.csv",
    "stage1_ordinal_training.csv",
    "stage1_ordinal_representation.csv",
    "stage1_ordinal_state_space.csv",
    "stage2_layout_training.csv",
    "stage2_layout_representation.csv",
    "stage2_layout_factorization.csv",
    "stage2_layout_state_space.csv",
    "stage2_layout_switch_effects.csv",
    "stage3_mixed_probe.csv",
    "stage3_mixed_geometry.csv",
    "stage3_mixed_responsibility.csv",
    "layout_overlap_sweep.csv",
    "stage5_novel_probe.csv",
    "ORDINAL_LAYOUT_ECOLOGY_REPORT.md",
)

EXTRA_TABLES = (
    "ordinal_layout_stage_summary",
    "stage1_ordinal_state_space_classification",
    "stage2_layout_transition_training",
    "stage2_layout_state_space_classification",
    "ordinal_layout_state_space_predictions",
    "stage5_novel_geometry",
    "ordinal_layout_training_audit",
    "ordinal_layout_probe_audit",
)

TABLE_NAMES = tuple(
    name.removesuffix(".csv") for name in REQUIRED_OUTPUTS if name.endswith(".csv")
) + EXTRA_TABLES


def _progress(config: LayoutEcologyConfig, message: str):
    if config.show_progress:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] ORDINAL-LAYOUT | {message}", flush=True)


def _source_hashes(repo: Path) -> dict[str, str]:
    result = {}
    for directory in (
        "confirmatory_srn", "ftt_srn_execution_v2", "structured_context_ecology",
    ):
        for path in sorted((repo / directory).rglob("*.py")):
            result[str(path.relative_to(repo))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


def _concat(parts) -> pd.DataFrame:
    frames = [frame for frame in parts if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _train_cohort(
    output: Path,
    stage: str,
    cohort: str,
    seeds: tuple[int, ...],
    config: LayoutEcologyConfig,
    *,
    overlap_count: int = 0,
) -> dict:
    models: dict[tuple[str, int], LayoutTrainingResult] = {}
    sequence_parts, transition_parts, probe_parts = [], [], []
    audits, probe_audits = [], []
    checkpoint_dir = output / "ordinal_layout_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total, completed = len(CONDITIONS) * len(seeds), 0
    for condition in CONDITIONS:
        for seed in seeds:
            result = train_layout_condition(
                stage, condition, config, seed=seed, cohort=cohort,
                overlap_count=overlap_count,
                progress=lambda text, c=condition: _progress(
                    config, f"{stage} | {c} | {text}"
                ),
            )
            models[(condition, seed)] = result
            sequence_parts.append(result.sequence_level)
            transition_parts.append(result.transition_level)
            audits.append(result.audit)
            probe, probe_audit = evaluate_frozen_layouts(
                result, config, contexts=("A", "B")
            )
            probe_parts.append(probe)
            probe_audits.append(probe_audit)
            name = (
                f"{stage}_{cohort}_overlap_{overlap_count}_"
                f"{condition}_seed_{seed}.pt"
            )
            torch.save(
                {**result.checkpoint, "config": asdict(config)},
                checkpoint_dir / name,
            )
            completed += 1
            _progress(config, f"{stage} {cohort}: completed {completed}/{total}")
    audit = pd.DataFrame(audits)
    high = audit[audit.condition.eq(HIGH_MONOTONIC)].set_index("seed")
    sequence_only = audit[audit.condition.eq(HIGH_SEQUENCE_ONLY)].set_index("seed")
    if not high.initial_model_sha256.equals(sequence_only.initial_model_sha256):
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
        "switch_effects": switch_effects(sequence, classifications),
        "audit": audit,
        "probe_audit": pd.DataFrame(probe_audits),
    }


def _probe_models(batch: dict, config: LayoutEcologyConfig, contexts):
    frames, audits = [], []
    for result in batch["models"].values():
        frame, audit = evaluate_frozen_layouts(result, config, contexts=contexts)
        frames.append(frame)
        audits.append(audit)
    return _concat(frames), pd.DataFrame(audits)


def _gate(batch: dict, stage: str, cohort: str, config: LayoutEcologyConfig):
    return evaluate_gate(
        stage, cohort, batch["training_summary"], batch["representation"],
        batch["classifications"], config,
    )


def _winner_table(batch: dict) -> pd.DataFrame:
    return batch["classifications"][
        batch["classifications"].row_type.eq("seed")
    ][["condition", "seed", "model_label"]].drop_duplicates()


def _mixed_tables(probes: pd.DataFrame, context: str, config: LayoutEcologyConfig):
    responsibility = responsibility_table(probes, context, config)
    geometry = probe_geometry(probes, context)
    return responsibility, geometry, combine_responsibility_geometry(
        responsibility, geometry
    )


def _high_primary_summary(combined: pd.DataFrame, stage: str) -> tuple[float, float]:
    one = combined[
        combined.stage.eq(stage)
        & combined.condition.eq(HIGH_MONOTONIC)
        & combined.probe_context.eq("Mixed")
    ]
    usable = one[one.responsibility_denominator_usable.eq(True)]
    return (
        float(usable.probe_between_A_B.mean()),
        float(one.distance_to_AB_segment.mean()),
    )


def _overlap_row_table(
    batch: dict,
    gate: dict,
    metrics: pd.DataFrame,
    overlap_count: int,
) -> pd.DataFrame:
    summary = batch["training_summary"].merge(
        batch["representation"][
            batch["representation"].row_type.eq("seed_aggregate")
        ][[
            "stage", "cohort", "sigma_context", "condition", "seed",
            "aggregate_AB_separation", "AB_hidden_cosine",
            "matched_minus_different_cosine",
        ]],
        on=["stage", "cohort", "sigma_context", "condition", "seed"],
        how="left",
    ).merge(_winner_table(batch), on=["condition", "seed"], how="left")
    if not metrics.empty:
        primary = metrics[metrics.probe_context.eq("Mixed")].drop(
            columns="probe_context"
        )
        summary = summary.merge(
            primary,
            on=["stage", "cohort", "condition", "seed"], how="left",
        )
    for column in (
        "usable_transition_count", "between_endpoint_fraction",
        "midpoint_deviation", "global_q", "mean_segment_distance",
        "mean_manifold_distance", "inside_segment_fraction",
    ):
        if column not in summary:
            summary[column] = np.nan
    summary["overlap_count"] = overlap_count
    summary["stage_gate_passed"] = gate["gate_passed"]
    summary["stage_failure_reason"] = gate["failure_reason"]
    return summary


def _report_table(frame: pd.DataFrame, columns=None) -> str:
    if frame.empty:
        return "Not run because of the stage firewall."
    value = frame if columns is None else frame[columns]
    return value.to_string(index=False)


def _write_report(
    output: Path,
    tables: dict[str, pd.DataFrame],
    *,
    mixed_improved: bool,
    overlap_invoked: bool,
    selected_overlap_count: int | None,
    interpretation: str,
    blocked_by: str,
):
    stages = tables["ordinal_layout_stage_summary"]
    baseline = tables["stage0_reference_summary"]
    representations = _concat([
        tables["stage1_ordinal_representation"],
        tables["stage2_layout_representation"],
    ])
    rep = representations[
        representations.row_type.eq("seed_aggregate")
    ].groupby(["stage", "cohort", "condition"], as_index=False).agg(
        AB_separation=("aggregate_AB_separation", "mean"),
        matched_position_cosine=("AB_hidden_cosine", "mean"),
        different_position_cosine=("different_position_AB_cosine", "mean"),
        matched_minus_different=("matched_minus_different_cosine", "mean"),
    ) if not representations.empty else pd.DataFrame()
    factor = tables["stage2_layout_factorization"]
    factor_summary = factor.groupby(
        ["stage", "cohort", "condition"], as_index=False
    ).agg(
        matched_cosine=("matched_position_AB_cosine", "mean"),
        different_cosine=("cross_position_AB_cosine", "mean"),
        factorization_RMSE=("additive_position_context_RMSE", "mean"),
        factorization_variance_explained=(
            "additive_position_context_variance_explained", "mean"
        ),
    ) if not factor.empty else pd.DataFrame()
    mixed = tables["stage3_mixed_responsibility"]
    mixed_summary = mixed[
        mixed.responsibility_denominator_usable.eq(True)
        & mixed.condition.eq(HIGH_MONOTONIC)
    ].groupby(["stage", "cohort", "probe_context"], as_index=False).agg(
        observations=("transition_position", "size"),
        between_endpoint_fraction=("probe_between_A_B", "mean"),
        absolute_midpoint_deviation=(
            "probe_midpoint_deviation", lambda x: float(np.mean(np.abs(x)))
        ),
        global_q=("global_q_unclipped", "mean"),
        segment_distance=("distance_to_AB_segment", "mean"),
        manifold_distance=("distance_to_nearest_trained_state", "mean"),
    ) if not mixed.empty else pd.DataFrame()
    novel = tables["stage5_novel_probe"]
    novel_summary = novel[
        novel.probe_context.eq("Novel")
        & novel.condition.isin([HIGH_MONOTONIC, LOW_MONOTONIC])
    ].groupby("condition", as_index=False).agg(
        normalized_RT=("normalized_RT", "mean"),
        sequence_CE=("sequence_CE", "mean"),
        next_token_accuracy=("next_token_correct", "mean"),
    ) if not novel.empty else pd.DataFrame()
    overlap = tables["layout_overlap_sweep"]
    overlap_summary = overlap.groupby(
        ["overlap_count", "condition"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        Cross_Split_fraction=(
            "model_label", lambda x: float(x.isin(["Cross", "Split"]).mean())
        ),
        AB_separation=("aggregate_AB_separation", "mean"),
        Mixed_between_fraction=("between_endpoint_fraction", "mean"),
        Mixed_segment_distance=("mean_segment_distance", "mean"),
        stage_gate_passed=("stage_gate_passed", "first"),
        joint_improvement_vs_categorical=(
            "joint_improvement_vs_categorical", "first"
        ),
        selected_followup=("selected_followup", "first"),
    ) if not overlap.empty else pd.DataFrame()

    report = f"""# Ordinal sequence + spatial layout ecology report

## 1. Reference phenotype

The successful categorical Stage 0 result was loaded without retraining.

```text
{baseline.groupby('condition', as_index=False).agg(seeds=('seed','nunique'), CE_late=('CE_late','mean'), RT_late=('RT_late','mean'), AB_separation=('aggregate_AB_separation','mean')).to_string(index=False)}
```

## 2. Shared ordinal scaffold

Stage 1 used token + P1--P8 + the stable global A/B cue. Its 5-seed gate was:

```text
{_report_table(stages[stages.stage.eq('stage1_ordinal')])}
```

## 3. Context-specific full layouts

Stage 2 used two layout banks present from training onset. A activated only the
A bank, B only the B bank. The screen and confirmation gates were:

```text
{_report_table(stages[stages.stage.eq('stage2_layout')])}
```

Seed-level fits, seed-average fits, switch costs, endpoint differences, and raw
trajectories are saved separately and plotted in Figures 2--3.

## 4. Representation

Matched ordinal positions remain distinguishable from different positions as
reported below. Factorization is descriptive, not a pass/fail criterion.

```text
{_report_table(rep)}
```

```text
{_report_table(factor_summary)}
```

## 5. Mixed

Mixed was evaluated only after the Stage 2 Cross/Split confirmation passed.
Primary Mixed used both familiar layouts and C=`[1,1]`; `Mixed_half` is the
input-norm-matched C=`[.5,.5]` control. Improvement on both frozen criteria:
`{mixed_improved}`.

```text
{_report_table(mixed_summary)}
```

Sparse shared-item sweep invoked: `{overlap_invoked}`.
Smallest gate-passing follow-up improving both categorical-reference criteria:
`{selected_overlap_count}`. This is a 5-seed follow-up screen, not a new
20-seed confirmation.

```text
{_report_table(overlap_summary)}
```

Novel was probe-only, used the trained layout banks, and had C=`[0,0]`:

```text
{_report_table(novel_summary)}
```

## 6. Final interpretation

**{interpretation}.**

Blocked after: `{blocked_by or 'none'}`. No human observations were loaded or
fitted, no salience/noise branch was run, and no Mixed or Novel probe updated
model parameters.
"""
    (output / "ORDINAL_LAYOUT_ECOLOGY_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def run_ordinal_layout_ecology(
    output_dir: str | Path, *, config: LayoutEcologyConfig,
) -> dict[str, pd.DataFrame]:
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    frozen_before = _source_hashes(repo)
    baseline = baseline_bundle(repo, config.baseline_output)
    firewall = OrdinalLayoutFirewall()
    tables = {name: pd.DataFrame() for name in TABLE_NAMES}
    stage_rows, audits, probe_audits = [], [], []
    fits_for_figures, predictions_for_figures = [], []
    blocked_by = ""
    overlap_invoked = False
    selected_overlap_count = None
    mixed_improved = False

    tables["stage0_reference_summary"] = baseline["summary"]
    gate0 = baseline_gate(baseline, config)
    stage_rows.append(gate0)
    firewall.record("stage0_reference", gate0["gate_passed"])
    _progress(config, f"Stage 0 frozen reference gate: {gate0['gate_passed']}")
    if not gate0["gate_passed"]:
        blocked_by = "stage0_reference"

    stage1 = stage2_screen = stage2_confirm = None
    if not blocked_by and firewall.may_run("stage1_screen"):
        _progress(config, "Stage 1: five-seed shared ordinal scaffold screen")
        stage1 = _train_cohort(
            output, "stage1_ordinal", "screen", config.screen_seeds, config
        )
        audits.append(stage1["audit"]); probe_audits.append(stage1["probe_audit"])
        gate1 = _gate(stage1, "stage1_ordinal", "screen", config)
        stage_rows.append(gate1)
        firewall.record("stage1_screen", gate1["gate_passed"])
        if not gate1["gate_passed"]:
            blocked_by = "stage1_screen: " + gate1["failure_reason"]

    if stage1:
        tables["stage1_ordinal_training"] = stage1["sequence"]
        tables["stage1_ordinal_representation"] = stage1["representation"]
        tables["stage1_ordinal_state_space"] = stage1["fits"]
        tables["stage1_ordinal_state_space_classification"] = stage1["classifications"]
        fits_for_figures.append(stage1["fits"])
        predictions_for_figures.append(stage1["predictions"])

    if not blocked_by and firewall.may_run("stage2_screen"):
        _progress(config, "Stage 2: five-seed full-layout screen")
        stage2_screen = _train_cohort(
            output, "stage2_layout", "screen", config.screen_seeds, config
        )
        audits.append(stage2_screen["audit"])
        probe_audits.append(stage2_screen["probe_audit"])
        gate2s = _gate(stage2_screen, "stage2_layout", "screen", config)
        stage_rows.append(gate2s)
        firewall.record("stage2_screen", gate2s["gate_passed"])
        if not gate2s["gate_passed"]:
            blocked_by = "stage2_screen: " + gate2s["failure_reason"]

    if not blocked_by and firewall.may_run("stage2_confirmatory"):
        _progress(config, "Stage 2 screen passed; confirming 20 seeds")
        stage2_confirm = _train_cohort(
            output, "stage2_layout", "confirmatory",
            config.confirmatory_seeds, config,
        )
        audits.append(stage2_confirm["audit"])
        probe_audits.append(stage2_confirm["probe_audit"])
        gate2c = _gate(stage2_confirm, "stage2_layout", "confirmatory", config)
        stage_rows.append(gate2c)
        firewall.record("stage2_confirmatory", gate2c["gate_passed"])
        if not gate2c["gate_passed"]:
            blocked_by = "stage2_confirmatory: " + gate2c["failure_reason"]

    for batch in (stage2_screen, stage2_confirm):
        if batch:
            tables["stage2_layout_training"] = _concat([
                tables["stage2_layout_training"], batch["sequence"]
            ])
            # Persist the transition history already produced during training.
            # This logging-only table supports preregistered/posthoc recovery
            # analyses without checkpoint replay or additional optimizer steps.
            tables["stage2_layout_transition_training"] = _concat([
                tables["stage2_layout_transition_training"],
                batch["transitions"],
            ])
            tables["stage2_layout_representation"] = _concat([
                tables["stage2_layout_representation"], batch["representation"]
            ])
            tables["stage2_layout_state_space"] = _concat([
                tables["stage2_layout_state_space"], batch["fits"]
            ])
            tables["stage2_layout_state_space_classification"] = _concat([
                tables["stage2_layout_state_space_classification"],
                batch["classifications"],
            ])
            tables["stage2_layout_switch_effects"] = _concat([
                tables["stage2_layout_switch_effects"], batch["switch_effects"]
            ])
            fits_for_figures.append(batch["fits"])
            predictions_for_figures.append(batch["predictions"])

    factor_parts = [factorization_diagnostics(baseline["probes"])]
    for batch in (stage1, stage2_screen, stage2_confirm):
        if batch:
            factor_parts.append(batch["factorization"])
    tables["stage2_layout_factorization"] = _concat(factor_parts)

    if not blocked_by and firewall.may_run("stage3_mixed"):
        _progress(config, "Stage 3: frozen dual-layout Mixed probes")
        layout_probe, audit = _probe_models(
            stage2_confirm, config, ("A", "B", "Mixed", "Mixed_half")
        )
        probe_audits.append(audit)
        categorical_probe = baseline["probes"][
            baseline["probes"].probe_context.isin(["A", "B", "Mixed"])
        ].copy()
        tables["stage3_mixed_probe"] = _concat([categorical_probe, layout_probe])
        responsibility_parts, geometry_parts, combined_parts = [], [], []
        for probe, context in (
            (categorical_probe, "Mixed"),
            (layout_probe, "Mixed"),
            (layout_probe, "Mixed_half"),
        ):
            responsibility, geometry, combined = _mixed_tables(probe, context, config)
            responsibility_parts.append(responsibility)
            geometry_parts.append(geometry)
            combined_parts.append(combined)
        tables["stage3_mixed_geometry"] = _concat(geometry_parts)
        tables["stage3_mixed_responsibility"] = _concat(combined_parts)
        categorical_between, categorical_distance = _high_primary_summary(
            combined_parts[0], "stage0_categorical"
        )
        layout_between, layout_distance = _high_primary_summary(
            combined_parts[1], "stage2_layout"
        )
        mixed_improved = bool(
            layout_between > categorical_between
            and layout_distance < categorical_distance
        )
        firewall.record("stage3_mixed", True)
        _progress(
            config,
            "Mixed characterized: "
            f"between {categorical_between:.3f}->{layout_between:.3f}, "
            f"segment {categorical_distance:.3f}->{layout_distance:.3f}, "
            f"joint improvement={mixed_improved}",
        )

    if (
        not blocked_by and not mixed_improved
        and firewall.may_run("stage4_overlap")
    ):
        overlap_invoked = True
        _progress(config, "Primary Mixed did not improve both criteria; running fixed overlap sweep")
        sweep_parts = []
        for overlap_count in config.overlap_levels:
            batch = _train_cohort(
                output, "stage4_overlap", "screen", config.screen_seeds, config,
                overlap_count=overlap_count,
            )
            audits.append(batch["audit"]); probe_audits.append(batch["probe_audit"])
            gate = _gate(batch, "stage4_overlap", "screen", config)
            gate["overlap_count"] = overlap_count
            stage_rows.append(gate)
            metrics = pd.DataFrame()
            if gate["gate_passed"]:
                probe, audit = _probe_models(batch, config, ("A", "B", "Mixed"))
                probe_audits.append(audit)
                responsibility, geometry, _ = _mixed_tables(probe, "Mixed", config)
                metrics = mixed_metrics(responsibility, geometry)
            sweep_parts.append(_overlap_row_table(batch, gate, metrics, overlap_count))
        sweep = _concat(sweep_parts)
        improvement_by_count = {}
        for overlap_count in config.overlap_levels:
            high = sweep[
                sweep.overlap_count.eq(overlap_count)
                & sweep.condition.eq(HIGH_MONOTONIC)
            ]
            gate_passed = bool(high.stage_gate_passed.all())
            improvement_by_count[overlap_count] = bool(
                gate_passed
                and high.between_endpoint_fraction.mean() > categorical_between
                and high.mean_segment_distance.mean() < categorical_distance
            )
        eligible = [
            count for count, improved in improvement_by_count.items() if improved
        ]
        selected_overlap_count = min(eligible) if eligible else None
        sweep["joint_improvement_vs_categorical"] = sweep.overlap_count.map(
            improvement_by_count
        )
        sweep["selected_followup"] = (
            sweep.overlap_count.eq(selected_overlap_count)
            if selected_overlap_count is not None else False
        )
        tables["layout_overlap_sweep"] = sweep
        firewall.record("stage4_overlap", True)

    if not blocked_by and firewall.may_run("stage5_novel"):
        _progress(config, "Stage 5: frozen new-layout probe")
        novel_probe, audit = _probe_models(
            stage2_confirm, config, ("A", "B", "Novel")
        )
        probe_audits.append(audit)
        tables["stage5_novel_probe"] = novel_probe
        tables["stage5_novel_geometry"] = probe_geometry(novel_probe, "Novel")
        firewall.record("stage5_novel", True)

    tables["ordinal_layout_stage_summary"] = pd.DataFrame(stage_rows)
    tables["ordinal_layout_training_audit"] = _concat(audits)
    tables["ordinal_layout_probe_audit"] = _concat(probe_audits)
    tables["ordinal_layout_state_space_predictions"] = _concat(
        predictions_for_figures
    )
    all_fits = _concat(fits_for_figures)

    stage2_gate_passed = bool(
        stage2_confirm is not None
        and any(
            row["stage"] == "stage2_layout"
            and row["cohort"] == "confirmatory"
            and row["gate_passed"] for row in stage_rows
        )
    )
    if stage2_gate_passed and mixed_improved:
        interpretation = "A. shared ordinal scaffold + spatial realization is sufficient"
    elif stage2_gate_passed:
        interpretation = "B. it preserves splitting but not recombination"
    elif any(
        row["stage"].startswith("stage1") or row["stage"].startswith("stage2")
        for row in stage_rows
    ) and any(
        row.get("G2_high_greater_low_splitting", False)
        and not row.get("G3_high_cross_split", False)
        for row in stage_rows
    ):
        interpretation = "C. it weakens context-specific behavior too much"
    else:
        interpretation = "D. it does not preserve the original phenotype"

    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    _write_report(
        output, tables, mixed_improved=mixed_improved,
        overlap_invoked=overlap_invoked,
        selected_overlap_count=selected_overlap_count,
        interpretation=interpretation,
        blocked_by=blocked_by,
    )

    if config.make_plots:
        from .plotting import make_figures
        _progress(config, "Rendering six ordinal/layout figures")
        make_figures(
            output / "figures_ordinal_layout_ecology",
            baseline=baseline,
            stage1=stage1,
            stage2=stage2_confirm or stage2_screen,
            stage1_representation=tables["stage1_ordinal_representation"],
            stage2_representation=tables["stage2_layout_representation"],
            fits=all_fits,
            predictions=tables["ordinal_layout_state_space_predictions"],
            mixed_probe=tables["stage3_mixed_probe"],
            mixed_geometry=tables["stage3_mixed_geometry"],
        )

    frozen_after = _source_hashes(repo)
    if frozen_before != frozen_after:
        raise AssertionError("A frozen predecessor source changed during this run.")
    config_record = {
        **asdict(config),
        **layout_metadata(),
        "analysis_version": "ordinal-layout-ecology-1.0.0",
        "stage1_feature_dim": feature_dim("stage1_ordinal", config),
        "stage2_feature_dim": feature_dim("stage2_layout", config),
        "stage2_total_input_dim": config.token_dim + feature_dim("stage2_layout", config),
        "global_context_cues": {
            name: value.tolist() for name, value in GLOBAL_CUES.items()
        },
        "primary_mixed_layout_banks": "[Layout_A, Layout_B]",
        "mixed_half_is_exact_complete_input_norm_control": True,
        "novel_uses_only_trained_layout_bank_dimensions": True,
        "layout_or_context_noise_sigma": 0.0,
        "mixed_improved_both_frozen_criteria": mixed_improved,
        "overlap_sweep_invoked": overlap_invoked,
        "selected_overlap_count": selected_overlap_count,
        "final_interpretation": interpretation,
        "blocked_by": blocked_by or None,
        "human_rows_used": 0,
        "firewall_outcomes": firewall.outcomes,
        "firewall_attempted": firewall.attempted,
        "frozen_source_hashes_before": frozen_before,
        "frozen_source_hashes_after": frozen_after,
        "required_outputs": list(REQUIRED_OUTPUTS),
    }
    (output / "ordinal_layout_config.json").write_text(
        json.dumps(config_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Missing ordinal/layout outputs: {missing}")
    _progress(config, f"Complete; blocked_by={blocked_by or 'none'}")
    return tables
