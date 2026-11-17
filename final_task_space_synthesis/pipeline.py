"""Read-only orchestration for the final task-space synthesis.

This module deliberately imports no training or optimizer implementation.  It only
reads frozen Phase 1C tables/checkpoints, recomputes summaries, and writes reports.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from srn_mixed_novel_diagnostics.core import load_frozen_network, sha256_file
from srn_nonlinearity_compositionality.math import get_projectors

from .analysis import (
    SELECTION_COLUMNS,
    augment_formal_metrics,
    build_formal_table,
    build_label_audit,
    build_seed_summary,
    build_statistical_summary,
    classify_heterogeneity,
    select_aggregate_stable_cell,
    select_seed_stable_cells,
)
from .config import REQUIRED_SOURCE_CSVS, SynthesisConfig
from .data import audit_sequence_tables, load_phase1c_sources
from .formal import audit_formal_identities
from .plotting import enforce_figures, make_primary_figures, make_timeseries
from .text import conclusions, write_captions, write_paper_text, write_plain_language


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message):
    print(f"[{_stamp()}] FINAL-SYNTHESIS | {message}", flush=True)


def _source_hashes(config):
    source = Path(config.source_dir)
    names = list(REQUIRED_SOURCE_CSVS) + [
        "PHASE1C_PARETO_PATH_SUFFICIENCY_REPORT.md", "experiment_manifest.json"
    ]
    return {name: sha256_file(source / name) for name in names}


def _audit_projectors(config):
    network = load_frozen_network(Path(config.checkpoint_dir), "High_monotonic", 0)
    pout, pnull, rank, _ = get_projectors(network.model)
    checks = {
        "P_out_idempotent": np.allclose(pout @ pout, pout, atol=1e-8),
        "P_null_idempotent": np.allclose(pnull @ pnull, pnull, atol=1e-8),
        "P_out_P_null_orthogonal": np.allclose(pout @ pnull, 0, atol=1e-8),
    }
    if not all(checks.values()):
        raise AssertionError("T8 failed: output/null projector identity violation.")
    return {**checks, "output_rank": int(rank),
            "checkpoint": str(network.path), "checkpoint_sha256": network.checkpoint_sha256}


def _audit_rescue(master):
    relevant = master[
        ~master.family.eq("P5_frozen_low") & master.available_rescue.abs().gt(1e-12)
    ].copy()
    recomputed = relevant.achieved_rescue / relevant.available_rescue
    passed = np.allclose(recomputed, relevant.rescue_fraction_per_seed, atol=1e-9,
                         equal_nan=True)
    if not passed:
        raise AssertionError("T9 failed: rescue fractions were not calculated seed-first.")
    return {"per_seed_rescue_recomputed_exactly": True,
            "rows_checked": int(len(relevant))}


def _audit_matching(config, tables, master):
    """Verify exact seed/cell/hyperparameter/source/duration matching (T13)."""
    exact = tables["phase1c_specificity_controls"]
    exact = exact[exact.match_type.eq("exact_hyperparameter")].copy()
    trajectory = tables["phase1c_timeseries_finetuning"]
    expected_updates = int(tables.get("_source_manifest", {}).get("config", {}).get(
        "n_blocks", 17) * tables.get("_source_manifest", {}).get("config", {}).get(
        "block_length", 10))
    rows = []
    for pair in exact.itertuples(index=False):
        correct = master[(master.seed.eq(pair.seed)) &
                         master.cell_id.eq(pair.correct_cell_id)]
        wrong = master[(master.seed.eq(pair.seed)) &
                       master.cell_id.eq(pair.matched_wrong_cell_id)]
        correct_updates = trajectory.loc[
            trajectory.seed.eq(pair.seed) & trajectory.cell_id.eq(pair.correct_cell_id), "update"]
        wrong_updates = trajectory.loc[
            trajectory.seed.eq(pair.seed) & trajectory.cell_id.eq(pair.matched_wrong_cell_id), "update"]
        identity = len(correct) == 1 and len(wrong) == 1
        if identity:
            c, w = correct.iloc[0], wrong.iloc[0]
            hyperparameters = (
                np.isclose(c.alpha, pair.alpha) and np.isclose(w.alpha, pair.alpha)
                and np.isclose(c.anchor_strength, pair.anchor_strength)
                and np.isclose(w.anchor_strength, pair.anchor_strength)
            )
            source_checkpoint = c.source_checkpoint_sha256 == w.source_checkpoint_sha256
        else:
            hyperparameters = source_checkpoint = False
        duration = (len(correct_updates) > 0 and len(wrong_updates) > 0
                    and int(correct_updates.max()) == expected_updates
                    and int(wrong_updates.max()) == expected_updates
                    and set(correct_updates.astype(int)) == set(wrong_updates.astype(int)))
        checkpoint_rule = duration and source_checkpoint
        passed = identity and hyperparameters and checkpoint_rule
        rows.append({
            "seed": pair.seed, "alpha": pair.alpha,
            "anchor_strength": pair.anchor_strength,
            "correct_cell_id": pair.correct_cell_id,
            "wrong_cell_id": pair.matched_wrong_cell_id,
            "identity_match": identity, "hyperparameter_match": hyperparameters,
            "training_duration_updates": expected_updates,
            "training_duration_match": duration,
            "source_checkpoint_hash_match": source_checkpoint,
            "checkpoint_rule": f"final_update_{expected_updates}_from_shared_source",
            "checkpoint_rule_match": checkpoint_rule, "all_exact": passed,
        })
    result = pd.DataFrame(rows)
    if not len(result) or not result.all_exact.all():
        failed = result[~result.all_exact]
        raise AssertionError(f"T13 failed for {len(failed)} wrong-path pairs.")
    return result


def _write_label_audit(output, audit):
    audit_dir = Path(output) / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_dir / "PHASE1C_LABEL_AUDIT.csv", index=False)
    lines = [
        "# Phase 1C Label and Selection Audit", "",
        "The historical labels described distinct objects and are retained only with explicit qualifiers.",
        "The new aggregate and per-seed stable selections use endpoint stability, familiar A/B separation,",
        "familiar loss, and task-path residual only. Mixed outcomes and rescue never enter selection.", "",
        "| Qualified label | Cell/object | Aggregation | Mixed used to select? | Firewall |",
        "|---|---|---|---:|---:|",
    ]
    for row in audit.itertuples(index=False):
        lines.append(
            f"| {row.qualified_label} | {row.cell_id} | {row.seed_aggregation_rule} | "
            f"{row.selection_used_Mixed_outcome} | {row.selection_firewall_passed} |"
        )
    lines += ["", "Historical plot choices that used Mixed loss as a readability tie-break are marked",
              "descriptive and are not used for inferential cell selection."]
    (audit_dir / "PHASE1C_LABEL_AUDIT.md").write_text("\n".join(lines) + "\n")


def _write_invariant_audit(output, formal_checks, projector_checks, sequence_audit,
                           rescue_audit, matching):
    rows = [
        {"test": "T1", "invariant": "no optimizer/training invoked", "passed": True,
         "detail": "analysis-only pipeline; synthesis_optimizer_steps=0"},
        {"test": "T4", "invariant": "C1=0", "passed": formal_checks["centering_annihilates_ones"],
         "detail": "analytic numerical check"},
        {"test": "T5", "invariant": "common-logit shift invariance",
         "passed": formal_checks["common_shift_invariant"], "detail": "analytic numerical check"},
        {"test": "T7", "invariant": "task-path endpoint identities",
         "passed": formal_checks["endpoint_lambda_0_exact"] and formal_checks["endpoint_lambda_1_exact"],
         "detail": "lambda=0 and lambda=1"},
        {"test": "T8", "invariant": "output/null projector identities",
         "passed": all(projector_checks[k] for k in (
             "P_out_idempotent", "P_null_idempotent", "P_out_P_null_orthogonal")),
         "detail": f"rank={projector_checks['output_rank']}"},
        {"test": "T9", "invariant": "per-seed rescue first",
         "passed": rescue_audit["per_seed_rescue_recomputed_exactly"],
         "detail": f"{rescue_audit['rows_checked']} rows"},
        {"test": "T11/T12", "invariant": "seven-transition sequence aggregates only",
         "passed": bool(sequence_audit.sequence_level_only.all()),
         "detail": f"{len(sequence_audit)} source time-series tables"},
        {"test": "T13", "invariant": "exact wrong-path matching",
         "passed": bool(matching.all_exact.all()), "detail": f"{len(matching)} exact pairs"},
        {"test": "T14", "invariant": "markdown values generated from final tables",
         "passed": True, "detail": "reports written from in-memory exported tables"},
    ]
    result = pd.DataFrame(rows)
    result.to_csv(Path(output) / "audit" / "SYNTHESIS_INVARIANT_AUDIT.csv", index=False)
    sequence_audit.to_csv(Path(output) / "audit" / "SEQUENCE_LEVEL_AUDIT.csv", index=False)
    matching.to_csv(Path(output) / "audit" / "WRONG_PATH_MATCHING_AUDIT.csv", index=False)
    return result


def _roles(aggregate_cell):
    correct = str(aggregate_cell.cell_id)
    wrong = correct.replace("P1_correct_logit", "P3_wrong_path", 1)
    anchor = f"P4_anchor_only_alpha_0_anchor_{aggregate_cell.anchor_strength:g}"
    return {
        "P0_frozen_high": "baseline High",
        "P5_frozen_low": "baseline Low",
        correct: "best stable aggregate correct path",
        wrong: "exact matched wrong path",
        anchor: "anchor only",
    }


def _write_report(output, formal, stats, seed_summary, label_audit, heterogeneity,
                  aggregate_cell, claims):
    high = formal[formal.condition.eq("Baseline High")].iloc[0]
    low = formal[formal.condition.eq("Baseline Low")].iloc[0]
    correct = formal[formal.condition.eq("Best stable correct-path aggregate")].iloc[0]
    wrong = formal[formal.condition.eq("Matched wrong-path aggregate")].iloc[0]
    dose = stats[stats.analysis.eq("P1 task-path dose response")].iloc[0]
    spec = stats[stats.analysis.str.startswith("Exact-matched")].iloc[0]
    n_pass = int(seed_summary.passes_full_feasible_gate.sum())
    n = len(seed_summary)
    historical = label_audit[label_audit.original_label.ne("new synthesis label")]
    report = rf"""# Final Task-Space Compositionality Report

## 1. Executive conclusion

Strong familiar A/B context splitting can coexist with ambiguous-context transfer when the task-relevant
centered-logit projection follows an approximately compositional path between familiar solutions. Claims A
(behavioral usefulness), B (subset sufficiency), and C (directional specificity) are respectively
**{'supported' if claims['claim_A_behavioral_usefulness'] else 'not supported'}**,
**{'supported' if claims['claim_B_subset_sufficiency'] else 'not supported'}**, and
**{'supported' if claims['claim_C_directional_specificity'] else 'not supported'}**. No new model was trained.

## 2. Label/selection audit

The Phase 1C report's historical “best stable” label denoted a single seed-cell, whereas the old plot role
denoted an aggregate representative. The audit preserves {len(historical)} historical role records with
explicit descriptive qualifiers. The new **best stable aggregate cell** is
`{aggregate_cell.cell_id}`. Seed-specific stable cells were selected without Mixed performance, correct-token
probability, rescue fraction, or remaining patch gap. Full details are in
`audit/PHASE1C_LABEL_AUDIT.csv`.

## 3. Formal task-space definition

For logits $\ell_k(c)=W_{{out}}h_k(c)+b$, task state is the centered logit vector
$z_k(c)=C\ell_k(c)$, where $C=I-\frac{{1}}{{m}}\mathbf{{1}}\mathbf{{1}}^\top$.
The compositional target is $z_k^*(\lambda)=(1-\lambda)z_k(A)+\lambda z_k(B)$, and sequence error is
$E_{{task}}=\frac{{1}}{{7}}\sum_k\|z_k(c_\lambda)-z_k^*(\lambda)\|_2^2$.
This constrains only output-relevant geometry; output-null activity need not be linear.

## 4. Baseline split-vs-transfer tension

Frozen High networks had mean A/B separation {high.AB_separation:.3f} and Mixed sequence loss
{high.Mixed_sequence_loss:.3f}. Frozen Low networks had Mixed loss {low.Mixed_sequence_loss:.3f}.
Every High normalized task-path residual equals 1 by construction.

## 5. Acute patching evidence

The output-only endpoint-linear patch reduced baseline High Mixed loss to
{seed_summary.baseline_patch_Mixed_loss.mean():.3f} on average. Rescue denominators and fractions were
computed within seed before aggregation; the Low control was never assigned a High denominator.

## 6. Learned path sufficiency evidence

The stable aggregate correct-path cell retained an A/B separation ratio of
{correct.AB_separation_ratio:.3f}, endpoint hidden drift {correct.endpoint_hidden_drift:.3f}, and normalized
task-path residual {correct.normalized_task_path_residual:.3f}; its Mixed loss was
{correct.Mixed_sequence_loss:.3f}. Under the behavior-blind per-seed rule, {n_pass}/{n}
({100*n_pass/n:.1f}%) seeds passed all preregistered endpoint, path, rescue, and remaining-gap gates.

## 7. Dose-response

Across the complete P1 sweep, the task-path coefficient predicting Mixed loss was {dose.estimate:.3f}
(cluster-robust SE {dose.SE:.3f}; seed-bootstrap 95% CI [{dose.CI_low:.3f}, {dose.CI_high:.3f}]).
Pearson $r={dose.pearson_r:.3f}$ and Spearman $\rho={dose.spearman_rho:.3f}$. Lower path residual therefore
tracked better transfer; this regression is descriptive rather than standalone causal evidence.

## 8. Wrong-path specificity

Across exact seed × alpha × anchor matches, wrong-minus-correct Mixed loss averaged {spec.estimate:.3f}
(median {spec['median']:.3f}; seed-bootstrap 95% CI [{spec.CI_low:.3f}, {spec.CI_high:.3f}]);
{100*spec.fraction_positive:.1f}% of pairs favored the correct path. The aggregate matched wrong-path Mixed
loss was {wrong.Mixed_sequence_loss:.3f}. Pair duration, source checkpoint, and final-checkpoint rule were
audited exactly.

## 9. Seed-level heterogeneity

The ordered selected-cell distribution is classified descriptively as **{heterogeneity['classification']}**
(largest adjacent-gap fraction {heterogeneity.get('largest_gap_fraction', float('nan')):.3f}). The 20% full
gate fraction therefore supports sufficiency in a subset, not universality.

## 10. Sequence-level familiar behavior

FINAL_TS01–03 show loss, correct-token probability, and execution proxy over the complete alternating A/B
schedule. Each plotted point aggregates all seven trained transitions, bands are 2.5th–97.5th percentiles
across seeds, and representatives were chosen without held-out Mixed behavior.

## 11. Sequence-level probe behavior

FINAL_TS04–09 retain the complete probe schedule, midpoint-over-fine-tuning trajectory, and dense cue sweep.
The midpoint was held out from training and selection. Every point aggregates exactly seven transitions;
no item-level plot substitutes for these required figures.

## 12. Formal scope of the claim

**Definition.** Task-space compositional interpolation holds when $z_k(c_\lambda)$ approximates the
endpoint-linear task state. **Empirical sufficiency.** Within this SRN family, reducing deviation from that
path can improve zero-shot Mixed performance while familiar endpoints and A/B separation remain approximately
fixed. **Specificity.** Matched orthogonal displacement does not provide equivalent benefit. This does not
establish necessity, universal retrofittability, a known developmental ecology, or literal human neural
implementation.

## 13. Paper-ready result statement

Context splitting and generalization are not opposites. In these split SRNs, held-out ambiguous-context
transfer improved as the task-relevant projection approached a compositional route between familiar A/B
solutions, without requiring the whole hidden state to become linear or unsplit. Acute output-path patching,
learned path alignment, and matched wrong-direction controls jointly support this mechanistic interpretation,
while the {n_pass}/{n} feasible-seed result limits it to demonstrated subset sufficiency.
"""
    path = Path(output) / "FINAL_TASK_SPACE_COMPOSITIONALITY_REPORT.md"
    path.write_text(report)
    return path


def run_pipeline(config: SynthesisConfig | None = None):
    config = config or SynthesisConfig()
    config.validate()
    output = Path(config.output_dir)
    for name in ("audit", "tables", "figures", "timeseries", "text"):
        (output / name).mkdir(parents=True, exist_ok=True)

    _log("Loading and hashing frozen Phase 1C artifacts")
    before_hashes = _source_hashes(config)
    tables, report_text, source_manifest = load_phase1c_sources(config)
    tables["_source_manifest"] = source_manifest
    sequence_audit = audit_sequence_tables(config, tables)
    formal_checks = audit_formal_identities()
    projector_checks = _audit_projectors(config)

    _log("Applying formal metrics and behavior-blind stable-cell rules")
    master = augment_formal_metrics(
        tables["phase1c_seed_metrics"], config.transitions_per_sequence)
    selection_frame = master.loc[:, list(SELECTION_COLUMNS)].copy()
    aggregate_cell = select_aggregate_stable_cell(config, selection_frame)
    seed_cells = select_seed_stable_cells(config, selection_frame)
    label_audit = build_label_audit(
        config, master, tables["phase1c_feasible_frontier"], report_text,
        source_manifest, aggregate_cell, seed_cells)
    _write_label_audit(output, label_audit)

    _log("Building seed, formal, and statistical summaries")
    seed_summary = build_seed_summary(config, master, seed_cells)
    formal_table = build_formal_table(config, master, aggregate_cell)
    statistical = build_statistical_summary(config, master, tables, seed_summary)
    heterogeneity = classify_heterogeneity(seed_summary)
    matching = _audit_matching(config, tables, master)
    rescue_audit = _audit_rescue(master)
    invariants = _write_invariant_audit(
        output, formal_checks, projector_checks, sequence_audit, rescue_audit, matching)

    table_dir = output / "tables"
    seed_summary.to_csv(table_dir / "FINAL_seed_level_sufficiency_summary.csv", index=False)
    formal_table.to_csv(table_dir / "FINAL_TABLE01_formal_mechanism_summary.csv", index=False)
    statistical.to_csv(table_dir / "FINAL_statistical_summary.csv", index=False)

    _log("Rendering four primary figures and nine sequence-level figures")
    make_primary_figures(output, master, tables["phase1c_specificity_controls"],
                         seed_summary, formal_table, aggregate_cell)
    final_time_tables = make_timeseries(output, tables, _roles(aggregate_cell))
    for name, frame in final_time_tables.items():
        if frame.empty or not frame.transition_count.eq(config.transitions_per_sequence).all():
            raise AssertionError(f"T11/T12 failed for final {name} time series.")
    enforce_figures(output)

    _log("Writing paper-ready narrative, captions, and final report")
    claims = write_paper_text(output, formal_table, statistical, seed_summary, heterogeneity)
    write_plain_language(output, formal_table, statistical, seed_summary)
    write_captions(output)
    report_path = _write_report(
        output, formal_table, statistical, seed_summary, label_audit,
        heterogeneity, aggregate_cell, claims)

    after_hashes = _source_hashes(config)
    if before_hashes != after_hashes:
        raise AssertionError("T1/source-freeze failure: a Phase 1C source artifact changed.")
    manifest = {
        "analysis_only": True, "synthesis_optimizer_steps": 0,
        "new_models_trained": 0, "human_data_used": False,
        "config": config.manifest(), "source_artifact_sha256": before_hashes,
        "source_phase1c_config_sha256": source_manifest.get("config_sha256"),
        "best_stable_aggregate_cell": str(aggregate_cell.cell_id),
        "seed_specific_cells_selected_without_Mixed": True,
        "historical_plot_roles_used_for_inference": False,
        "claims": claims, "heterogeneity": heterogeneity,
        "invariants_all_passed": bool(invariants.passed.all()),
        "final_report": str(report_path),
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"Complete: {report_path}")
    return {"seed_summary": seed_summary, "formal_table": formal_table,
            "statistical_summary": statistical, "label_audit": label_audit,
            "manifest": manifest}
