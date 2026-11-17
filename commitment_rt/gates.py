"""Phase-1 behavioral gates and context-switch diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CommitmentConfig


def _slope(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return np.nan
    design = np.column_stack([np.ones(len(x)), x])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def compute_switch_costs(
    training: pd.DataFrame,
    config: CommitmentConfig,
) -> pd.DataFrame:
    rows = []
    for (c_time, seed), group in training.groupby(["c_time", "seed"], sort=True):
        group = group.sort_values("global_sequence").reset_index(drop=True)
        for block in sorted(group.block_index.unique()):
            if block <= 1:
                continue
            current = group[group.block_index.eq(block)].sort_values("sequence_in_block")
            preceding = group[group.block_index.eq(block - 1)].sort_values("sequence_in_block")
            first = current.iloc[0]
            late_preceding = preceding.tail(config.switch_preceding_window).normalized_rt.mean()
            recovery = current.head(config.switch_recovery_window)
            recovery_slope = _slope(recovery.sequence_in_block, recovery.normalized_rt)
            rows.append({
                "c_time": c_time,
                "seed": int(seed),
                "switch_block": int(block),
                "from_context": str(preceding.context.iloc[-1]),
                "to_context": str(first.context),
                "switch_sequence": int(first.global_sequence),
                "first_new_context_rt": float(first.normalized_rt),
                "late_preceding_context_rt": float(late_preceding),
                "switch_cost": float(first.normalized_rt - late_preceding),
                "first_three_recovery_slope": recovery_slope,
                "switch_cost_positive": bool(first.normalized_rt > late_preceding),
                "recovery_declines": bool(np.isfinite(recovery_slope) and recovery_slope < 0),
            })
    return pd.DataFrame(rows)


def _context_coefficient(training: pd.DataFrame) -> float:
    frame = training.copy()
    progress = frame.global_sequence.to_numpy(dtype=float)
    progress = (progress - progress.mean()) / max(progress.std(), 1e-12)
    within = frame.sequence_in_block.to_numpy(dtype=float)
    within = (within - within.mean()) / max(within.std(), 1e-12)
    context_b = frame.context.eq("B").astype(float).to_numpy()
    seeds = sorted(frame.seed.unique())
    seed_dummies = np.column_stack([
        frame.seed.eq(seed).astype(float).to_numpy() for seed in seeds[1:]
    ]) if len(seeds) > 1 else np.empty((len(frame), 0))
    design = np.column_stack([np.ones(len(frame)), progress, within, context_b, seed_dummies])
    return float(np.linalg.lstsq(design, frame.normalized_rt, rcond=None)[0][3])


def evaluate_candidate(
    training: pd.DataFrame,
    transitions: pd.DataFrame,
    cycles: pd.DataFrame,
    config: CommitmentConfig,
) -> tuple[dict, pd.DataFrame]:
    if training.c_time.nunique() != 1:
        raise ValueError("Gate evaluation requires exactly one c_time candidate.")
    c_time = float(training.c_time.iloc[0])
    direction_rows, grounding_slopes, preparation_fractions = [], [], []
    for seed, group in training.groupby("seed", sort=True):
        group = group.sort_values("global_sequence")
        window = min(config.direction_window_sequences, max(len(group) // 2, 1))
        early, late = group.head(window), group.tail(window)
        direction_rows.append({
            "seed": int(seed),
            "ce_change": float(late.sequence_ce.mean() - early.sequence_ce.mean()),
            "rt_change": float(late.normalized_rt.mean() - early.normalized_rt.mean()),
        })
        state_rows = transitions[transitions.seed.eq(seed)]
        grounding_slopes.append(_slope(
            state_rows.expected_p_correct, state_rows.normalized_rt
        ))
        cycle_rows = cycles[cycles.seed.eq(seed)].sort_values([
            "training_sequence", "sequence_position", "prep_cycle"
        ])
        differences = cycle_rows.groupby(
            ["training_sequence", "sequence_position"]
        ).p_correct.diff()
        preparation_fractions.append(float((differences.dropna() > 0).mean()))
    directions = pd.DataFrame(direction_rows)
    minimum = config.minimum_seed_fraction
    ce_change = float(directions.ce_change.mean())
    rt_change = float(directions.rt_change.mean())
    fraction_ce_down = float((directions.ce_change < 0).mean())
    fraction_rt_down = float((directions.rt_change < 0).mean())
    gate_a = bool(ce_change < 0 and fraction_ce_down >= minimum)
    gate_b = bool(rt_change < 0 and fraction_rt_down >= minimum)

    finite_grounding = [value for value in grounding_slopes if np.isfinite(value)]
    grounding_mean = float(np.mean(finite_grounding)) if finite_grounding else np.nan
    grounding_fraction = float(np.mean([
        value < 0 for value in finite_grounding
    ])) if finite_grounding else np.nan
    gate_c = bool(grounding_mean < 0 and grounding_fraction >= minimum)

    preparation_mean = float(np.mean(preparation_fractions))
    preparation_seed_fraction = float(np.mean([
        value > config.minimum_preparation_improvement_fraction
        for value in preparation_fractions
    ]))
    gate_d = bool(
        preparation_mean > config.minimum_preparation_improvement_fraction
        and preparation_seed_fraction >= minimum
    )

    hard = transitions.hard_rt_cycles.to_numpy(dtype=int)
    cycle_one_fraction = float(np.mean(hard == 1))
    maximum_fraction = float(np.mean(hard == config.max_preparation_cycles))
    gate_e = bool(
        cycle_one_fraction <= config.maximum_boundary_commit_fraction
        and maximum_fraction <= config.maximum_boundary_commit_fraction
    )

    switch = compute_switch_costs(training, config)
    late_switch = switch[switch.switch_block.ge(5)]
    mean_switch = float(late_switch.switch_cost.mean()) if len(late_switch) else np.nan
    fraction_positive_switch = float(
        late_switch.switch_cost_positive.mean()
    ) if len(late_switch) else np.nan
    recovery_fraction = float(
        late_switch.recovery_declines.mean()
    ) if len(late_switch) else np.nan
    context_coefficient = _context_coefficient(training)
    gate_f = bool(
        abs(context_coefficient) >= config.minimum_context_effect
        or (np.isfinite(mean_switch) and abs(mean_switch) >= config.minimum_context_effect)
    )
    passed = bool(gate_a and gate_b and gate_c and gate_d and gate_e and gate_f)
    summary = {
        "c_time": c_time,
        "seeds": int(training.seed.nunique()),
        "external_sequences_per_seed": int(training.global_sequence.nunique()),
        "ce_change_late_minus_early": ce_change,
        "fraction_seeds_ce_down": fraction_ce_down,
        "rt_change_late_minus_early": rt_change,
        "fraction_seeds_rt_down": fraction_rt_down,
        "mean_within_seed_confidence_rt_slope": grounding_mean,
        "fraction_seeds_confidence_rt_negative": grounding_fraction,
        "preparation_confidence_improvement_fraction": preparation_mean,
        "fraction_seeds_preparation_improvement_above_half": preparation_seed_fraction,
        "hard_commit_cycle_1_fraction": cycle_one_fraction,
        "hard_commit_max_cycle_fraction": maximum_fraction,
        "late_mean_switch_cost": mean_switch,
        "late_fraction_positive_switch_cost": fraction_positive_switch,
        "late_fraction_declining_recovery": recovery_fraction,
        "context_B_rt_coefficient_controlling_progress_within_block": context_coefficient,
        "gate_A_learning": gate_a,
        "gate_B_rt_acquisition": gate_b,
        "gate_C_uncertainty_grounding": gate_c,
        "gate_D_preparation_dynamics": gate_d,
        "gate_E_nondegenerate_rt": gate_e,
        "gate_F_context_structure": gate_f,
        "screening_passed": passed,
        "human_data_used": 0,
    }
    return summary, switch
