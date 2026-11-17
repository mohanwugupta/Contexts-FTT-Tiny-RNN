"""Phase-1 five-seed screen for recurrent-preparation commitment RT."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .analytical import run_analytical_calibration
from .config import CommitmentConfig
from .controls import threshold_control_trajectories
from .gates import evaluate_candidate
from .plotting import make_screen_figures
from .state_space import fit_screen_state_space
from .training import train_commitment_model


GATE_COLUMNS = [
    "gate_A_learning", "gate_B_rt_acquisition", "gate_C_uncertainty_grounding",
    "gate_D_preparation_dynamics", "gate_E_nondegenerate_rt",
    "gate_F_context_structure", "screening_passed",
]


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] COMMITMENT-RT | {message}", flush=True)


def _source_hashes(repo: Path, directories: tuple[str, ...]) -> dict[str, str]:
    hashes = {}
    for directory in directories:
        for path in sorted((repo / directory).glob("*.py")):
            hashes[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _interpretation(gates: pd.DataFrame) -> tuple[str, str]:
    if gates.screening_passed.any():
        return (
            "A",
            "recurrent preparation provides a viable emergent RT mechanism in the Phase-1 screen",
        )
    if not gates.gate_D_preparation_dynamics.any():
        return ("D", "recurrent preparation itself does not improve action readiness")
    if (gates.gate_F_context_structure & ~gates.gate_B_rt_acquisition).any():
        return ("C", "context structure works but acquisition direction fails")
    if (gates.gate_B_rt_acquisition & ~gates.gate_F_context_structure).any():
        return ("B", "timing works but context-specific state-space structure fails")
    return ("D", "recurrent preparation does not satisfy the complete mechanism gates")


def _write_report(
    output: Path,
    config: CommitmentConfig,
    analytical_summary: pd.DataFrame,
    analytical_checks: dict,
    gates: pd.DataFrame,
    switch_costs: pd.DataFrame,
    state_tables: dict[str, pd.DataFrame],
    selected_c_time: float | None,
):
    code, interpretation = _interpretation(gates)
    passing = gates[gates.screening_passed]
    selected = (
        gates[gates.c_time.eq(selected_c_time)].iloc[0] if selected_c_time is not None
        else None
    )
    if selected is None:
        readiness = "No time-cost candidate passed all A--F gates."
        direction = "No configuration qualified for Phase 2."
        switch_text = "Switch effects remain diagnostic only because no candidate passed."
    else:
        readiness = (
            f"For the frozen candidate `c_time={selected_c_time:g}`, confidence increased "
            f"on {selected.preparation_confidence_improvement_fraction:.1%} of adjacent "
            "preparation-cycle comparisons."
        )
        direction = (
            f"Mean late-minus-early CE was `{selected.ce_change_late_minus_early:.6f}`; "
            f"mean late-minus-early normalized RT was "
            f"`{selected.rt_change_late_minus_early:.6f}`."
        )
        chosen_switch = switch_costs[switch_costs.c_time.eq(selected_c_time)]
        late_switch = chosen_switch[chosen_switch.switch_block.ge(5)]
        switch_text = (
            f"Late switches had mean normalized-RT cost "
            f"`{late_switch.switch_cost.mean():.6f}`; "
            f"`{late_switch.switch_cost_positive.mean():.1%}` were positive and "
            f"`{late_switch.recovery_declines.mean():.1%}` showed declining first-three "
            "recovery trajectories."
        )
    fits = state_tables["screen_state_space_fits"]
    learned_winners = fits[
        fits.policy_type.eq("learned_hazard") & fits.winner.eq(True)
    ] if not fits.empty else fits
    winner_text = (
        "```text\n" + learned_winners[[
            "timing_condition", "model_label", "BIC", "RMSE", "R2", "converged"
        ]].to_string(index=False) + "\n```"
        if len(learned_winners) else
        "Not fit: the A--F gate firewall admitted no learned commitment candidate."
    )
    report = f"""# Recurrent preparation and commitment RT report

## Scope and frozen design

This report is the `{config.profile}` High-scale=2 Phase-1 mechanism run
(`{len(config.seeds)}` seed(s)).
It uses `{config.training_sequences}` external A10/B10 training sequences per seed,
`{config.max_preparation_cycles}` internal preparation cycles per transition, and no
human data or RT/speed target. Mixed and Novel were not used in training or selection.

The 20-seed confirmatory pipeline, Low-scale comparison, sequence-only comparison,
post-training probes, and causal patching were **not started**. Their final output files
are therefore intentionally absent rather than populated with Phase-1 results.

The analytical hard trajectory follows the PRD's `.27, .32, .40, .52, ...` prefix and
ends at `.999`; the near-certain endpoint represents successful continued preparation
and prevents the high-time-cost toy agent from rationally giving up immediately.

## Analytical sanity check

All global checks passed: `{analytical_checks['analytical_sanity_passed']}`.

```text
{analytical_summary.to_string(index=False)}
```

## 1. Did recurrent preparation improve sequence readiness?

{readiness}

## 2. Did RT emerge in the correct learning direction?

{direction}

```text
{gates[["c_time", *GATE_COLUMNS, "ce_change_late_minus_early", "rt_change_late_minus_early", "mean_within_seed_confidence_rt_slope"]].to_string(index=False)}
```

## 3. Did context switching slow commitment?

{switch_text}

## 4. Did High-scale and Low-scale networks differ?

Low-scale is a Phase-2 confirmatory control and was not run. Canonical no-fatigue
Lump/Cross/Split fits were permitted only for learned candidates passing A--F.

{winner_text}

The nonlearned thresholds `.60/.70/.80/.90` were applied to the same frozen cycle
trajectories with zero retraining and fitted descriptively for comparison; they were
not used to select the learned time cost.

## 5. What happened on Mixed and Novel probes?

Deferred by design until a configuration is frozen for Phase 2. The implemented probe
path is read-only, starts each context from the same post-training hidden state, and
asserts identical parameter hashes before and after evaluation.

## 6. Did causal hidden-state interventions alter commitment time?

Not tested in Phase 1. Causal analyses remain behind the acquisition, nondegeneracy,
and preparation-dynamics stop gates.

## 7. Final interpretation

**{code}. {interpretation}.**

This interpretation follows the observed A--F gates, not the desired manuscript story.
Confirmatory 20-seed rerun started: `false`.
"""
    (output / "COMMITMENT_RT_REPORT.md").write_text(report, encoding="utf-8")


def run_commitment_screen(
    output_dir: str | Path,
    *,
    config: CommitmentConfig,
) -> dict[str, pd.DataFrame]:
    """Run analytical checks and the five-seed Phase-1 screen only."""
    config.validate()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    frozen_directories = ("execution_objective_search", "execution_readout_diagnostic")
    frozen_before = _source_hashes(repo, frozen_directories)
    _progress(config.show_progress, "Starting analytical time/error calibration")
    analytical, analytical_summary, analytical_checks = run_analytical_calibration(config)
    analytical.to_csv(output / "commitment_analytical_calibration.csv", index=False)
    analytical_summary.to_csv(output / "commitment_analytical_summary.csv", index=False)
    if not analytical_checks["analytical_sanity_passed"]:
        raise RuntimeError("Analytical commitment sanity check failed; neural training stopped.")

    training_frames, transition_frames, cycle_frames, audits, gate_rows, switch_frames = (
        [], [], [], [], [], []
    )
    checkpoint_dir = output / "screen_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total = len(config.c_time_values) * len(config.seeds)
    completed = 0
    for c_time in config.c_time_values:
        _progress(config.show_progress, f"Screening c_time={c_time:g}")
        candidate_training, candidate_transitions, candidate_cycles = [], [], []
        for seed in config.seeds:
            result = train_commitment_model(
                config, c_time=c_time, seed=seed,
                progress=(lambda message, cost=c_time: _progress(
                    config.show_progress, f"c_time={cost:g} | {message}"
                )),
            )
            completed += 1
            if result.optimizer_steps != config.training_sequences:
                raise AssertionError("External training exposure changed.")
            torch.save({
                "model_state_dict": result.model.state_dict(),
                "final_hidden": result.final_hidden,
                "seed": seed,
                "c_time": c_time,
                "config": asdict(config),
            }, checkpoint_dir / f"commitment_ctime_{c_time:g}_seed_{seed}.pt")
            candidate_training.append(result.training_trajectories)
            candidate_transitions.append(result.transition_dynamics)
            candidate_cycles.append(result.cycle_dynamics)
            audits.append(result.audit)
            _progress(config.show_progress, f"Completed model {completed}/{total}")
        one_training = pd.concat(candidate_training, ignore_index=True)
        one_transitions = pd.concat(candidate_transitions, ignore_index=True)
        one_cycles = pd.concat(candidate_cycles, ignore_index=True)
        gate, switches = evaluate_candidate(
            one_training, one_transitions, one_cycles, config
        )
        gate_rows.append(gate)
        switch_frames.append(switches)
        training_frames.append(one_training)
        transition_frames.append(one_transitions)
        cycle_frames.append(one_cycles)
        status = "PASS" if gate["screening_passed"] else "FAIL"
        _progress(config.show_progress, f"c_time={c_time:g} gates: {status}")

    training = pd.concat(training_frames, ignore_index=True)
    transitions = pd.concat(transition_frames, ignore_index=True)
    cycles = pd.concat(cycle_frames, ignore_index=True)
    gate_summary = pd.DataFrame(gate_rows)
    switch_costs = pd.concat(switch_frames, ignore_index=True)
    audit = pd.DataFrame(audits)
    initial_counts = audit.groupby("seed").initial_model_sha256.nunique()
    if not initial_counts.eq(1).all():
        raise AssertionError("A seed did not receive identical initial weights across c_time.")

    _progress(config.show_progress, "Applying frozen nonlearned threshold controls")
    threshold_transitions, threshold_sequences, threshold_summary = (
        threshold_control_trajectories(cycles, config)
    )
    _progress(config.show_progress, "Fitting conditional canonical state-space models")
    state_tables = fit_screen_state_space(
        training, gate_summary, threshold_sequences
    )
    passing = sorted(gate_summary.loc[
        gate_summary.screening_passed.eq(True), "c_time"
    ].astype(float).tolist())
    # Selection never consults state-space fits, thresholds, probes, or human data.
    selected_c_time = passing[0] if passing else None

    outputs = {
        "commitment_screening_summary": gate_summary,
        "commitment_training_trajectories": training,
        "commitment_transition_dynamics": transitions,
        "commitment_cycle_dynamics": cycles,
        "commitment_switch_costs": switch_costs,
        "commitment_training_audit": audit,
        "commitment_threshold_transition_controls": threshold_transitions,
        "commitment_threshold_trajectories": threshold_sequences,
        "commitment_threshold_controls": threshold_summary,
        **state_tables,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    frozen_after = _source_hashes(repo, frozen_directories)
    if frozen_before != frozen_after:
        raise AssertionError("A frozen earlier experiment was modified by this pipeline.")
    record = {
        **asdict(config),
        "analysis_version": "commitment-rt-phase1-1.0.0",
        "architecture": "8-unit tanh Elman SRN with repeated-input preparation cycles",
        "commitment_distribution": "q_m = a_m product_{j<m}(1-a_j), final hazard forced to 1",
        "rt_definition": "(E[commitment cycle]-1)/(M_max-1)",
        "commit_head_initialization": "zeros (neutral hazard=0.5)",
        "analytical_checks": analytical_checks,
        "passing_c_time_values": passing,
        "selected_c_time": selected_c_time,
        "selection_rule": "lowest prespecified c_time passing A-F; no state-space or human criterion",
        "human_rows_used": 0,
        "human_rt_target_rows": 0,
        "mixed_novel_training_rows": 0,
        "confirmatory_20_seed_rerun_started": False,
        "phase_2_started": False,
        "frozen_source_hashes_before": frozen_before,
        "frozen_source_hashes_after": frozen_after,
    }
    (output / "commitment_screen_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        output, config, analytical_summary, analytical_checks, gate_summary,
        switch_costs, state_tables, selected_c_time,
    )
    if config.make_plots:
        _progress(config.show_progress, "Rendering Phase-1 figures")
        plotted_c_time = selected_c_time if selected_c_time is not None else config.c_time_values[0]
        make_screen_figures(
            output / "figures_commitment_rt", training, cycles,
            plotted_c_time, state_tables,
        )
    _progress(
        config.show_progress,
        f"Phase 1 complete; {len(passing)}/{len(config.c_time_values)} candidates passed; "
        "20-seed pipeline not started",
    )
    return outputs
