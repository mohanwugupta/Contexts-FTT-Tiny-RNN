"""End-to-end nonhuman execution-readout diagnostic."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from execution_objective_search.gates import evaluate_screening_gates

from .config import READOUT_CONDITIONS, READOUT_LABELS, ReadoutDiagnosticConfig
from .plotting import make_figures
from .state_space import fit_passing_readouts
from .training import train_readout_condition


GATE_COLUMNS = [
    "gate_A_learning_direction", "gate_B_execution_grounding",
    "gate_C_objective_tracking", "gate_D_no_pathological_saturation",
    "gate_E_contextual_dynamics", "screening_passed",
]


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] READOUT-DIAGNOSTIC | {message}", flush=True)


def _source_hashes(repo: Path, relative_directory: str) -> dict[str, str]:
    result = {}
    for path in sorted((repo / relative_directory).glob("*.py")):
        result[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _write_report(
    output: Path,
    config: ReadoutDiagnosticConfig,
    gates: pd.DataFrame,
    state_tables: dict[str, pd.DataFrame],
):
    display = gates[[
        "readout_condition", *GATE_COLUMNS,
        "ce_change_late_minus_early", "confidence_change_late_minus_early",
        "rt_change_late_minus_early", "fraction_seeds_rt_down",
        "confidence_rt_slope", "fraction_seeds_confidence_rt_negative",
        "speed_optimum_correlation",
        "fraction_seeds_positive_speed_optimum_correlation",
        "speed_total_saturation_fraction", "trained_speed_95_minus_05",
        "late_switch_excess_over_within",
    ]].copy()
    state_fits = state_tables["state_space_model_fits"]
    if state_fits.empty:
        state_text = "No readout passed all A--E gates, so state-space fitting was skipped."
        coexistence = False
        winner_text = "none"
    else:
        winners = state_fits[state_fits.winner.eq(True)][[
            "readout_condition", "model_label", "BIC", "RMSE", "R2",
            "converged", "convergence_verified",
        ]]
        state_text = "```text\n" + winners.to_string(index=False) + "\n```"
        coexistence = bool(
            (state_fits.winner.eq(True) & state_fits.convergence_verified.eq(True))
            .loc[state_fits.model.eq("SS_CROSS")]
            .any()
        )
        winner_text = ", ".join(
            f"{row.readout_condition}: {row.model_label}"
            for row in winners.itertuples(index=False)
        )
    if coexistence:
        critical = (
            "Yes. At least one readout passed every behavioral gate and its RT trajectory "
            "was best described by Cross."
        )
    elif state_fits.empty:
        critical = (
            "Not established. No readout passed every behavioral gate, so Cross was not fit."
        )
    else:
        critical = (
            "No in this diagnostic. At least one readout produced sensible RT-like learning, "
            "but Cross was not the lowest-BIC state-space description."
        )

    report = f"""# Execution readout diagnostic report

## Critical result

**Can sensible RT-like learning and High-scale Cross structure coexist?**

{critical}

State-space winner(s): {winner_text}.

## Frozen design

- High initialization scale: `{config.high_init_scale:g}` for recurrent and sequence weights
- Seeds: `{list(config.seeds)}`
- Exposure: `{config.training_sequences}` A10/B10 sequences
- Normative loss: `{config.normative_candidate.equation}`
- Fixed execution weight: `{config.lambda_exec:g}`
- Human observations or target RTs used: `0`
- Objective-search source or output files modified: `no`
- Confirmatory 20-seed paper pipeline started: `no`

The same seeded recurrent and sequence parameters were used for all readouts.
The unrestricted execution head alone was reset to exactly zero as requested.

## Readouts

1. Deterministic: `v = v*(stopgrad(p))`.
2. Monotonic: `v = sigmoid(softplus(a)*logit(stopgrad(p)) + b)` with two trainable scalars.
3. Unrestricted: the existing linear hidden-state execution head followed by sigmoid,
   with its weight and bias initialized to zero.

## Unchanged A--E gates

```text
{display.to_string(index=False)}
```

## Conditional Lump/Cross/Split fits

The canonical no-fatigue implementation in `state-space_bridge.state_space`, ported
from `state-space_SRN.Rmd`, was applied to the five-seed mean normalized-RT trace
only for conditions passing all A--E gates.

If its deliberately strict optimizer reports a line-search warning, the same
fit is checked against the repository's independent established R-model port;
`convergence_verified` requires matching BIC and predictions within `1e-6`.

{state_text}

All state-space fits are descriptive and use pure A/B training observations only.
No Mixed, Novel, probe, or human row enters this diagnostic.
"""
    (output / "EXECUTION_READOUT_DIAGNOSTIC_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def run_readout_diagnostic(
    output_dir: str | Path,
    *,
    config: ReadoutDiagnosticConfig,
) -> dict:
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    frozen_hashes_before = _source_hashes(repo, "execution_objective_search")
    sequence_frames, token_frames, audits, gate_rows = [], [], [], []

    _progress(
        config.show_progress,
        "Starting matched High-scale=2 readout diagnostic; objective search is frozen",
    )
    for condition_index, condition in enumerate(READOUT_CONDITIONS, start=1):
        _progress(
            config.show_progress,
            f"Training {READOUT_LABELS[condition]} ({condition_index}/3)",
        )
        condition_sequences, condition_tokens = [], []
        for seed in config.seeds:
            result = train_readout_condition(
                condition, config, seed=seed,
                progress=(lambda message, name=condition: _progress(
                    config.show_progress, f"{name} | {message}"
                )),
            )
            if result.optimizer_steps != config.training_sequences:
                raise AssertionError("A readout did not receive exactly one update per sequence.")
            condition_sequences.append(result.sequence_results)
            condition_tokens.append(result.token_results)
            audits.append(result.initialization_audit)
        sequence = pd.concat(condition_sequences, ignore_index=True)
        tokens = pd.concat(condition_tokens, ignore_index=True)
        condition_sequences = sequence
        condition_tokens = tokens
        sequence_frames.append(sequence)
        token_frames.append(tokens)
        gates = evaluate_screening_gates(
            sequence, tokens, config.normative_candidate, config.gate_config()
        )
        gates.update({
            "readout_condition": condition,
            "readout_label": READOUT_LABELS[condition],
        })
        gate_rows.append(gates)

    sequence_results = pd.concat(sequence_frames, ignore_index=True)
    token_results = pd.concat(token_frames, ignore_index=True)
    initialization_audit = pd.DataFrame(audits)
    gate_summary = pd.DataFrame(gate_rows)

    shared_counts = initialization_audit.groupby(
        "seed"
    ).shared_initial_parameters_sha256.nunique()
    if not shared_counts.eq(1).all():
        raise AssertionError("Matched readouts did not receive identical shared initial weights.")
    unrestricted = initialization_audit[
        initialization_audit.readout_condition.eq("unrestricted_zero_initialized")
    ]
    if not unrestricted.execution_head_zero_initialized.all():
        raise AssertionError("The unrestricted execution head was not initialized to zero.")

    passing_count = int(gate_summary.screening_passed.sum())
    _progress(
        config.show_progress,
        f"Behavioral gates complete; {passing_count}/3 readouts passed all A--E gates",
    )
    state_tables = fit_passing_readouts(sequence_results, gate_summary)
    if passing_count:
        _progress(config.show_progress, "Fitted canonical Lump/Cross/Split to passing RT traces")
    else:
        _progress(config.show_progress, "State-space fitting skipped by the A--E firewall")

    sequence_results.to_csv(output / "readout_sequence_results.csv", index=False)
    token_results.to_csv(output / "readout_token_results.csv", index=False)
    initialization_audit.to_csv(output / "readout_initialization_audit.csv", index=False)
    gate_summary.to_csv(output / "readout_gate_summary.csv", index=False)
    for name, table in state_tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    frozen_hashes_after = _source_hashes(repo, "execution_objective_search")
    if frozen_hashes_before != frozen_hashes_after:
        raise AssertionError("Frozen objective-search sources changed during the diagnostic.")
    state_fits = state_tables["state_space_model_fits"]
    winners = (
        state_fits[state_fits.winner.eq(True)][
            [
                "readout_condition", "model", "BIC", "delta_BIC",
                "converged", "convergence_verified",
            ]
        ].to_dict(orient="records")
        if not state_fits.empty else []
    )
    record = {
        **asdict(config),
        "analysis_version": "execution-readout-diagnostic-1.0.0",
        "normative_candidate": asdict(config.normative_candidate),
        "architecture_source": "ftt_srn_execution_v2.models.DualHeadElmanSRN",
        "gate_source": "execution_objective_search.gates.evaluate_screening_gates",
        "state_space_source": "state_space_bridge.state_space (state-space_SRN.Rmd port)",
        "objective_search_frozen": True,
        "objective_search_source_hashes_before": frozen_hashes_before,
        "objective_search_source_hashes_after": frozen_hashes_after,
        "matched_shared_initializations": True,
        "human_rows_used": 0,
        "human_target_rt_rows": 0,
        "mixed_novel_rows_used": 0,
        "passing_readout_count": passing_count,
        "state_space_fitted_readouts": sorted(
            state_fits.readout_condition.unique().tolist()
        ) if not state_fits.empty else [],
        "state_space_winners": winners,
        "confirmatory_20_seed_rerun_started": False,
    }
    (output / "execution_readout_diagnostic_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output, config, gate_summary, state_tables)
    if config.make_plots:
        _progress(config.show_progress, "Rendering readout and conditional state-space figures")
        make_figures(
            output / "figures_execution_readout_diagnostic",
            sequence_results, token_results, gate_summary, state_tables,
        )
    _progress(
        config.show_progress,
        "Complete; objective search unchanged and 20-seed pipeline not started",
    )
    return {
        "readout_sequence_results": sequence_results,
        "readout_token_results": token_results,
        "readout_initialization_audit": initialization_audit,
        "readout_gate_summary": gate_summary,
        **state_tables,
    }
