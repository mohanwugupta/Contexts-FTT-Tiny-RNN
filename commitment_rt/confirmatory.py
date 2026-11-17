"""Guarded Phase-2 architecture comparison, probes, and causal analysis."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .causal import run_commitment_patching
from .config import CommitmentConfig
from .controls import threshold_control_trajectories
from .gates import compute_switch_costs, evaluate_candidate
from .plotting import make_confirmatory_figures
from .probes import evaluate_frozen_contexts, evaluate_sequence_only_contexts
from .state_space import fit_confirmatory_state_space
from .training import train_commitment_model, train_sequence_only_model


CONDITIONS = (
    ("High_commitment", 2.0, True),
    ("Low_commitment", .02, True),
    ("High_sequence_only", 2.0, False),
)


def _progress(enabled: bool, message: str):
    if enabled:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] COMMITMENT-CONFIRMATORY | {message}", flush=True)


def _read_screen(screen_dir: Path) -> tuple[float, pd.DataFrame, dict]:
    summary_path = screen_dir / "commitment_screening_summary.csv"
    config_path = screen_dir / "commitment_screen_config.json"
    if not summary_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Phase 2 requires a completed Phase-1 screening summary and config."
        )
    summary = pd.read_csv(summary_path)
    record = json.loads(config_path.read_text(encoding="utf-8"))
    passing = sorted(summary.loc[summary.screening_passed.eq(True), "c_time"].astype(float))
    if not passing:
        raise RuntimeError(
            "Phase 2 blocked: no Phase-1 time cost passed A--F. "
            "The 20-seed pipeline was not started."
        )
    selected = record.get("selected_c_time")
    selected = passing[0] if selected is None else float(selected)
    if selected not in passing:
        raise ValueError("Recorded selected c_time did not pass the saved Phase-1 gates.")
    if record.get("human_rows_used", 0) != 0:
        raise ValueError("Phase-1 selection was not independent of human data.")
    return selected, summary, record


def _ab_hidden_separation(cycles: pd.DataFrame, config: CommitmentConfig) -> float:
    late = cycles[
        cycles.training_sequence.gt(cycles.training_sequence.max() - 20)
        & cycles.prep_cycle.eq(config.max_preparation_cycles)
    ]
    columns = [f"hidden_{index}" for index in range(1, config.hidden_dim + 1)]
    means = late.groupby(["context", "sequence_position"])[columns].mean()
    distances = []
    for position in range(1, config.transitions_per_sequence + 1):
        if ("A", position) in means.index and ("B", position) in means.index:
            difference = (
                means.loc[("A", position)].to_numpy(dtype=float)
                - means.loc[("B", position)].to_numpy(dtype=float)
            )
            distances.append(float(np.linalg.norm(difference)))
    return float(np.mean(distances)) if distances else np.nan


def _seed_summaries(training: pd.DataFrame, cycles: pd.DataFrame,
                    config: CommitmentConfig) -> pd.DataFrame:
    rows = []
    for (condition, seed), group in training.groupby(["condition", "seed"], sort=True):
        group = group.sort_values("global_sequence")
        window = min(20, len(group) // 2)
        early, late = group.head(window), group.tail(window)
        active = bool(group.commitment_policy_active.iloc[0])
        seed_cycles = cycles[
            cycles.condition.eq(condition) & cycles.seed.eq(seed)
        ]
        rows.append({
            "condition": condition,
            "seed": int(seed),
            "init_scale": .02 if condition == "Low_commitment" else 2.0,
            "commitment_policy_active": active,
            "rt_output_available": active,
            "ce_early": float(early.sequence_ce.mean()),
            "ce_late": float(late.sequence_ce.mean()),
            "ce_change_late_minus_early": float(
                late.sequence_ce.mean() - early.sequence_ce.mean()
            ),
            "rt_early": float(early.normalized_rt.mean()) if active else np.nan,
            "rt_late": float(late.normalized_rt.mean()) if active else np.nan,
            "rt_change_late_minus_early": (
                float(late.normalized_rt.mean() - early.normalized_rt.mean())
                if active else np.nan
            ),
            "ab_hidden_separation": _ab_hidden_separation(seed_cycles, config),
            "external_training_sequences": len(group),
            "human_data_used": 0,
        })
    return pd.DataFrame(rows)


def _write_report(output: Path, selected: float, seed_summary: pd.DataFrame,
                  gates: pd.DataFrame, switches: pd.DataFrame,
                  state_tables: dict[str, pd.DataFrame], probes: pd.DataFrame,
                  patching: pd.DataFrame):
    high = gates[gates.condition.eq("High_commitment")].iloc[0]
    high_switch = switches[switches.condition.eq("High_commitment")]
    late_switch = high_switch[high_switch.switch_block.ge(5)]
    separation = seed_summary.groupby("condition").ab_hidden_separation.mean()
    fits = state_tables["final_state_space_fits"]
    winners = fits[fits.winner.eq(True)][
        ["timing_condition", "model_label", "BIC", "RMSE", "R2", "converged"]
    ]
    probe_summary = probes.groupby(["condition", "probe_context"])[
        ["expected_rt_cycles", "normalized_rt", "expected_p_correct"]
    ].mean().reset_index()
    if high.screening_passed:
        interpretation = "A. recurrent preparation provides a viable emergent RT mechanism"
    elif not high.gate_D_preparation_dynamics:
        interpretation = "D. recurrent preparation itself does not improve action readiness"
    elif high.gate_B_rt_acquisition and not high.gate_F_context_structure:
        interpretation = "B. timing works but context-specific state-space structure fails"
    else:
        interpretation = "C. context structure works but acquisition direction fails"
    patch_text = (
        f"Mean absolute normalized-RT effect was `{patching.delta_normalized_rt.abs().mean():.6f}`; "
        f"prediction identity was preserved in `{patching.prediction_identity_preserved.mean():.1%}` "
        "of token-matched interventions."
        if len(patching) else
        "Skipped because the confirmatory High model did not pass the full acquisition firewall."
    )
    report = f"""# Recurrent preparation and commitment RT report

## Frozen confirmatory design

The Phase-1-selected `c_time={selected:g}` was frozen before this 20-seed comparison.
Conditions were High scale=2 + commitment, Low scale=.02 + commitment, and High
scale=2 sequence-only. No human row or RT/speed target entered training or selection.

## 1. Did recurrent preparation improve sequence readiness?

The confirmatory High preparation-improvement fraction was
`{high.preparation_confidence_improvement_fraction:.1%}`. Gate D:
`{bool(high.gate_D_preparation_dynamics)}`.

## 2. Did RT emerge in the correct learning direction?

High late-minus-early CE: `{high.ce_change_late_minus_early:.6f}`. High
late-minus-early normalized RT: `{high.rt_change_late_minus_early:.6f}`.

## 3. Did context switching slow commitment?

High late mean switch cost: `{late_switch.switch_cost.mean():.6f}`; positive fraction:
`{late_switch.switch_cost_positive.mean():.1%}`; declining-recovery fraction:
`{late_switch.recovery_declines.mean():.1%}`.

## 4. Did High-scale and Low-scale networks differ?

Mean A/B hidden separation was `{separation.get('High_commitment', np.nan):.6f}` for
High and `{separation.get('Low_commitment', np.nan):.6f}` for Low.

```text
{winners.to_string(index=False)}
```

The High sequence-only control has no RT output by construction.

## 5. What happened on Mixed and Novel probes?

All probes were frozen and made zero optimizer steps or parameter updates.

```text
{probe_summary.to_string(index=False)}
```

## 6. Did causal hidden-state interventions alter commitment time?

{patch_text}

RT, correct-token probability, CE, hard commitment cycle, and prediction identity are
reported jointly in `final_commitment_patching.csv`.

## 7. Final interpretation

**{interpretation}.**

This label follows the prespecified gates rather than the desired manuscript result.
"""
    (output / "COMMITMENT_RT_REPORT.md").write_text(report, encoding="utf-8")


def run_commitment_confirmatory(
    screen_dir: str | Path,
    output_dir: str | Path,
    *,
    config: CommitmentConfig,
) -> dict[str, pd.DataFrame]:
    """Run Phase 2 only when a saved Phase-1 result passed every gate."""
    config.validate()
    screen = Path(screen_dir).expanduser().resolve()
    selected_c_time, _, screen_record = _read_screen(screen)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "final_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _progress(config.show_progress, f"Using frozen Phase-1 c_time={selected_c_time:g}")

    training_frames, transition_frames, cycle_frames, audits = [], [], [], []
    probe_frames, probe_hidden_frames, probe_audits = [], [], []
    trained_models = {}
    for condition, scale, commitment_active in CONDITIONS:
        _progress(config.show_progress, f"Training {condition}")
        for seed in config.seeds:
            callback = lambda message, name=condition: _progress(
                config.show_progress, f"{name} | {message}"
            )
            if commitment_active:
                result = train_commitment_model(
                    config, c_time=selected_c_time, seed=seed,
                    init_scale=scale, progress=callback,
                )
                trajectory = result.training_trajectories.copy()
                transitions = result.transition_dynamics.copy()
                cycles = result.cycle_dynamics.copy()
                trajectory["commitment_policy_active"] = True
                trajectory["rt_output_available"] = True
                transitions["condition"] = condition
                transition_frames.append(transitions)
            else:
                result = train_sequence_only_model(
                    config, seed=seed, init_scale=scale, progress=callback
                )
                trajectory = result.training_trajectories.copy()
                trajectory["c_time"] = np.nan
                cycles = result.cycle_dynamics.copy()
                cycles["c_time"] = np.nan
            trajectory["condition"] = condition
            cycles["condition"] = condition
            training_frames.append(trajectory)
            cycle_frames.append(cycles)
            audit = dict(result.audit)
            audit["condition"] = condition
            audits.append(audit)
            trained_models[(condition, seed)] = (
                result.model, result.final_hidden.detach().clone()
            )
            torch.save({
                "model_state_dict": result.model.state_dict(),
                "final_hidden": result.final_hidden,
                "condition": condition,
                "seed": seed,
                "c_time": selected_c_time if commitment_active else None,
                "config": asdict(config),
            }, checkpoint_dir / f"{condition}_seed_{seed}.pt")

            if commitment_active:
                probe, hidden, probe_audit = evaluate_frozen_contexts(
                    result.model, config, seed=seed, c_time=selected_c_time,
                    initial_hidden=result.final_hidden,
                )
                probe["condition"] = condition
                hidden["condition"] = condition
                hidden["commitment_policy_active"] = True
                hidden["rt_output_available"] = True
                probe_frames.append(probe)
                probe_hidden_frames.append(hidden)
            else:
                hidden, probe_audit = evaluate_sequence_only_contexts(
                    result.model, config, seed=seed,
                    initial_hidden=result.final_hidden,
                )
                hidden["condition"] = condition
                probe_hidden_frames.append(hidden)
            probe_audit["condition"] = condition
            probe_audits.append(probe_audit)

    training = pd.concat(training_frames, ignore_index=True, sort=False)
    transitions = pd.concat(transition_frames, ignore_index=True, sort=False)
    cycles = pd.concat(cycle_frames, ignore_index=True, sort=False)
    seed_summary = _seed_summaries(training, cycles, config)

    gate_rows, switch_frames = [], []
    for condition in ("High_commitment", "Low_commitment"):
        gate, switch = evaluate_candidate(
            training[training.condition.eq(condition)],
            transitions[transitions.condition.eq(condition)],
            cycles[cycles.condition.eq(condition)], config,
        )
        gate["condition"] = condition
        gate_rows.append(gate)
        switch["condition"] = condition
        switch_frames.append(switch)
    gates = pd.DataFrame(gate_rows)
    switches = pd.concat(switch_frames, ignore_index=True)
    state_tables = fit_confirmatory_state_space(training)

    control_frames = []
    for condition in ("High_commitment", "Low_commitment"):
        _, _, summary = threshold_control_trajectories(
            cycles[cycles.condition.eq(condition)], config
        )
        summary["condition"] = condition
        summary["control"] = "fixed_confidence_threshold"
        control_frames.append(summary)
    controls = pd.concat(control_frames, ignore_index=True, sort=False)
    controls = pd.concat([controls, pd.DataFrame([{
        "condition": "High_sequence_only", "control": "no_timing_output",
        "c_time": np.nan, "threshold": np.nan, "optimizer_steps": 0,
        "retraining": False, "human_data_used": 0,
    }])], ignore_index=True, sort=False)

    probes = pd.concat(probe_frames, ignore_index=True)
    probe_hidden = pd.concat(probe_hidden_frames, ignore_index=True, sort=False)
    high_gate = gates[gates.condition.eq("High_commitment")].iloc[0]
    patch_frames, patch_audits = [], []
    if bool(high_gate.screening_passed):
        _progress(config.show_progress, "Running frozen High token-matched causal patches")
        for seed in config.seeds:
            model, final_hidden = trained_models[("High_commitment", seed)]
            patched, patch_audit = run_commitment_patching(
                model, config, seed=seed, condition="High_commitment",
                c_time=selected_c_time, initial_hidden=final_hidden,
            )
            patch_frames.append(patched)
            patch_audits.append(patch_audit)
    patching = (
        pd.concat(patch_frames, ignore_index=True) if patch_frames
        else pd.DataFrame(columns=[
            "condition", "seed", "c_time", "probe_context", "donor_context",
            "sequence_position", "delta_expected_rt_cycles", "delta_normalized_rt",
            "delta_hard_rt_cycles", "delta_p_correct", "delta_sequence_ce",
            "prediction_identity_preserved", "optimizer_steps", "parameter_updates",
        ])
    )

    outputs = {
        "final_commitment_seed_summary": seed_summary,
        "final_commitment_training_trajectories": training,
        "final_commitment_cycle_dynamics": cycles,
        "final_commitment_gate_summary": gates,
        "final_commitment_switch_costs": switches,
        "final_state_space_fits": state_tables["final_state_space_fits"],
        "final_state_space_trace": state_tables["final_state_space_trace"],
        "final_state_space_parameters": state_tables["final_state_space_parameters"],
        "final_state_space_predictions": state_tables["final_state_space_predictions"],
        "final_probe_commitment": probes,
        "final_probe_hidden_states": probe_hidden,
        "final_commitment_patching": patching,
        "final_commitment_controls": controls,
        "final_training_audit": pd.DataFrame(audits),
        "final_probe_audit": pd.DataFrame(probe_audits),
        "final_patching_audit": pd.DataFrame(patch_audits),
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    record = {
        **asdict(config),
        "analysis_version": "commitment-rt-confirmatory-1.0.0",
        "selected_c_time": selected_c_time,
        "selected_by_saved_phase_1_only": True,
        "source_screen_config": screen_record,
        "conditions": [name for name, _, _ in CONDITIONS],
        "human_rows_used": 0,
        "human_rt_target_rows": 0,
        "mixed_novel_training_rows": 0,
        "confirmatory_20_seed_rerun_started": config.profile == "confirmatory",
    }
    (output / "final_commitment_config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(
        output, selected_c_time, seed_summary, gates, switches,
        state_tables, probes, patching,
    )
    if config.make_plots:
        make_confirmatory_figures(
            output / "figures_commitment_rt", state_tables, probes, patching
        )
    _progress(config.show_progress, "Confirmatory analysis complete")
    return outputs
