"""End-to-end analytical calibration and isolated five-seed objective screen."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FAMILY_LABELS, ObjectiveCandidate, SearchConfig, build_candidate_grid
from .gates import evaluate_screening_gates
from .objectives import calibrate_candidate, summarize_calibration
from .plotting import make_search_figures
from .training import train_candidate


GATE_COLUMNS = [
    "gate_A_learning_direction", "gate_B_execution_grounding",
    "gate_C_objective_tracking", "gate_D_no_pathological_saturation",
    "gate_E_contextual_dynamics", "screening_passed",
]


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] EXEC-OBJECTIVE | {message}", flush=True)


def _select_candidate(summary: pd.DataFrame) -> str | None:
    eligible = summary[summary.screening_passed.fillna(False).astype(bool)].copy()
    if eligible.empty:
        return None
    family_priority = {
        family: index for index, family in enumerate(("A", "B", "C", "D", "E"))
    }
    eligible["family_priority"] = eligible.family.map(family_priority)
    # Complexity first. Within an equally simple family/exponent, choose the
    # strongest nonhuman screen using uncertainty tracking, dynamic range, and
    # consistency of the required RT direction.
    eligible = eligible.sort_values(
        ["family_priority", "gamma", "nonhuman_screen_quality_score", "candidate_id"],
        ascending=[True, True, False, True],
    )
    return str(eligible.candidate_id.iloc[0])


def _write_report(
    output: Path,
    config: SearchConfig,
    candidates: tuple[ObjectiveCandidate, ...],
    summary: pd.DataFrame,
    selected_id: str | None,
):
    contains_policy_bce = any(item.family == "E" for item in candidates)
    selected_text = "No candidate passed every screening gate."
    hyperparameters = "None; the expensive 20-seed rerun remains blocked."
    rationale = (
        "No objective was selected; the prespecified families require diagnosis before any "
        "new expansion."
    )
    if contains_policy_bce:
        rationale = (
            "The bounded post-failure policy candidates also failed the stable execution-"
            "grounding gate. The gates were not relaxed, and no further family was added."
        )
    if selected_id is not None:
        candidate = next(item for item in candidates if item.candidate_id == selected_id)
        row = summary[summary.candidate_id.eq(selected_id)].iloc[0]
        selected_text = f"**{selected_id}** ({FAMILY_LABELS[candidate.family]})"
        hyperparameters = (
            f"`c_time={candidate.c_time:g}`, `c_risk={candidate.c_risk:g}`, "
            f"`gamma={candidate.gamma:g}`, `lambda_exec={candidate.lambda_exec:g}`"
        )
        rationale = (
            "This was the simplest objective family that passed analytical calibration and all "
            "five behavioral gates. Within that family/exponent it had the strongest nonhuman "
            f"screen-quality score ({row.nonhuman_screen_quality_score:.3f})."
        )
    equations = "\n".join(
        f"- `{item.candidate_id}`: `{item.equation}`"
        for item in candidates
    )
    display_columns = [
        "candidate_id", "family", "calibration_passed", "screening_ran", *GATE_COLUMNS,
        "ce_change_late_minus_early", "confidence_change_late_minus_early",
        "rt_change_late_minus_early", "speed_optimum_correlation",
        "speed_total_saturation_fraction", "trained_speed_95_minus_05",
        "late_switch_excess_over_within",
    ]
    table = summary[display_columns].to_string(index=False)
    policy_note = ""
    if contains_policy_bce:
        policy_note = """
Candidate E is a bounded post-failure expansion permitted by the PRD. It retains the same
risk-derived `v*` but minimizes `BCE(v, stopgrad(v*))`; this gives a stable
speed-logit gradient without adding human targets or external competence.
"""
    report = f"""# Execution objective search report

## Status

This is an exploratory, nonhuman objective-discovery analysis. It used the
unchanged 8-unit Elman SRN, High initialization scale 2.0, learning rate .005,
and A10/B10 curriculum. It did not run Mixed/Novel probes, causal analyses, or
the expensive 20-seed confirmation.

## Candidate equations

Families A--D have the direct economic form

```text
L_exec(v | uncertainty) = c_time*(1-v) + c_risk*uncertainty*v^gamma
```

and the confidence/entropy/margin quantity is detached from sequence logits in
the execution loss. Their constrained normative optimum is

```text
v* = clip((c_time / (gamma*c_risk*uncertainty))^(1/(gamma-1)), 0, 1)
```

{equations}
{policy_note}

## Analytical calibration

`objective_calibration.csv` contains each certainty-to-optimal-speed and
certainty-to-optimal-RT curve, the saturation point, and `dL/dv` at speeds .25,
.50, and .75. Candidates that lacked positive slope, usable range, an
intermediate optimum, or a nonsaturated central range were rejected before SRN
training.

## Five-seed screen and gates

```text
{table}
```

Gate definitions:

- A: CE decreases, confidence increases, and normalized RT decreases, with the
  required direction in at least {config.minimum_direction_seed_fraction:.0%} of seeds.
- B: confidence predicts lower RT and/or entropy predicts higher RT across token states.
- C: actual speed positively tracks the objective-implied optimum.
- D: fewer than {config.maximum_trained_saturation_fraction:.0%} of states are near a
  boundary and the trained 5th-to-95th percentile speed range is at least
  {config.minimum_trained_speed_range:g}.
- E: late switch jumps exceed matched within-context changes, or a context term
  remains after controlling training progress and within-block position.

## Selected objective

{selected_text}

Rationale: {rationale}

Exact hyperparameters: {hyperparameters}

## Human-data independence and next step

- Human target RTs supplied: **0**
- Human Mixed/Novel observations used: **0**
- Human transition profiles used: **0**
- Trial number supplied to the network: **no**
- External A/B competence states supplied: **no**
- Confirmatory 20-seed rerun started: **no**

The 20-seed paper pipeline remains a separate action even if a candidate passed.
"""
    (output / "EXECUTION_OBJECTIVE_SEARCH_REPORT.md").write_text(report, encoding="utf-8")


def run_objective_search(
    output_dir: str | Path,
    *,
    config: SearchConfig,
    candidates: tuple[ObjectiveCandidate, ...] | None = None,
) -> dict:
    config.validate()
    candidates = candidates or build_candidate_grid(config)
    for candidate in candidates:
        candidate.validate()
        if candidate.lambda_exec != config.lambda_exec:
            raise ValueError("All candidates must use the search's fixed lambda_exec.")
    output = Path(output_dir).expanduser().resolve()
    figures = output / "figures_execution_objective_search"
    output.mkdir(parents=True, exist_ok=True)

    _progress(config.show_progress, f"Analytically calibrating {len(candidates)} objectives")
    calibration_frames, calibration_rows = [], []
    for candidate in candidates:
        table = calibrate_candidate(candidate, config)
        calibration_frames.append(table)
        calibration_rows.append(summarize_calibration(table, candidate, config))
    calibration = pd.concat(calibration_frames, ignore_index=True)
    calibration_summary = pd.DataFrame(calibration_rows)
    if config.screen_only_calibration_passers:
        viable_ids = set(calibration_summary.loc[
            calibration_summary.calibration_passed, "candidate_id"
        ])
        viable = [candidate for candidate in candidates if candidate.candidate_id in viable_ids]
    else:
        viable = list(candidates)
    _progress(
        config.show_progress,
        f"Calibration retained {len(viable)}/{len(candidates)} candidates for neural screening",
    )

    sequence_frames, token_frames, gate_rows = [], [], []
    for candidate_index, candidate in enumerate(viable, start=1):
        _progress(
            config.show_progress,
            f"Screening {candidate.candidate_id} ({candidate_index}/{len(viable)})",
        )
        candidate_sequence, candidate_tokens = [], []
        for seed in config.seeds:
            result = train_candidate(
                candidate, config, seed=seed,
                progress=(lambda message, cid=candidate.candidate_id: _progress(
                    config.show_progress, f"{cid} | {message}"
                )),
            )
            if result.optimizer_steps != config.training_sequences:
                raise AssertionError("A candidate did not receive exactly one update per sequence.")
            candidate_sequence.append(result.sequence_results)
            candidate_tokens.append(result.token_results)
        sequence = pd.concat(candidate_sequence, ignore_index=True)
        tokens = pd.concat(candidate_tokens, ignore_index=True)
        sequence_frames.append(sequence); token_frames.append(tokens)
        gate_rows.append(evaluate_screening_gates(sequence, tokens, candidate, config))

    screening = pd.concat(sequence_frames, ignore_index=True) if sequence_frames else pd.DataFrame()
    token_screening = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()
    gates = pd.DataFrame(gate_rows)
    summary = calibration_summary.merge(gates, on=["candidate_id", "family"], how="left")
    summary["screening_ran"] = summary.candidate_id.isin(
        set(screening.candidate_id) if len(screening) else set()
    )
    for column in GATE_COLUMNS:
        summary[column] = summary[column].eq(True)
    selected_id = _select_candidate(summary)
    summary["selected_objective"] = summary.candidate_id.eq(selected_id)
    summary["selection_used_human_probe_data"] = False
    summary["confirmatory_20_seed_rerun_started"] = False

    calibration.to_csv(output / "objective_calibration.csv", index=False)
    screening.to_csv(output / "objective_screening_results.csv", index=False)
    token_screening.to_csv(output / "objective_screening_token_results.csv", index=False)
    summary.to_csv(output / "objective_candidate_summary.csv", index=False)
    record = {
        **asdict(config),
        "analysis_version": "objective-search-1.0.0",
        "candidate_count": len(candidates),
        "calibration_passing_candidates": len(viable),
        "selected_candidate_id": selected_id,
        "selected_candidate": (
            asdict(next(item for item in candidates if item.candidate_id == selected_id))
            if selected_id is not None else None
        ),
        "architecture_source": "ftt_srn_execution_v2.models.DualHeadElmanSRN",
        "trainer_source": "execution_objective_search.training.train_candidate",
        "training_contexts": ["A", "B"],
        "mixed_novel_training_or_probe_rows": 0,
        "human_target_rt_rows": 0,
        "human_transition_profile_rows": 0,
        "uncertainty_detached_from_sequence_logits": True,
        "confirmatory_20_seed_rerun_started": False,
    }
    (output / "objective_search_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output, config, candidates, summary, selected_id)
    if config.make_plots:
        _progress(config.show_progress, "Rendering calibration, trajectory, tracking, and gate figures")
        make_search_figures(figures, calibration, screening, summary)
    _progress(
        config.show_progress,
        f"Complete; selected={selected_id or 'none'}; 20-seed rerun was not started",
    )
    return {
        "objective_calibration": calibration,
        "objective_screening_results": screening,
        "objective_screening_token_results": token_screening,
        "objective_candidate_summary": summary,
        "selected_candidate_id": selected_id,
    }
