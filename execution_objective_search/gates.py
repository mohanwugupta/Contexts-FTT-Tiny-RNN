"""Behavioral acceptance gates for execution-objective screening."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ObjectiveCandidate, SearchConfig


def _slope(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12:
        return float("nan")
    design = np.column_stack([np.ones(len(x)), x])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def _corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _finite_mean(values):
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _fraction(values, predicate):
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean([predicate(value) for value in finite])) if finite else np.nan


def _direction_changes(sequence: pd.DataFrame, config: SearchConfig) -> dict:
    rows = []
    for seed, group in sequence.groupby("seed", sort=True):
        group = group.sort_values("global_sequence")
        window = min(config.direction_window_sequences, max(len(group) // 2, 1))
        early = group.head(window)
        late = group.tail(window)
        rows.append({
            "seed": seed,
            "ce_change_late_minus_early": float(late.sequence_ce.mean() - early.sequence_ce.mean()),
            "confidence_change_late_minus_early": float(
                late.correct_token_confidence.mean() - early.correct_token_confidence.mean()
            ),
            "speed_change_late_minus_early": float(
                late.execution_speed.mean() - early.execution_speed.mean()
            ),
            "rt_change_late_minus_early": float(
                late.normalized_rt.mean() - early.normalized_rt.mean()
            ),
        })
    table = pd.DataFrame(rows)
    result = {}
    for column in table.columns.difference(["seed"]):
        result[column] = float(table[column].mean())
    result.update({
        "fraction_seeds_ce_down": float((table.ce_change_late_minus_early < 0).mean()),
        "fraction_seeds_confidence_up": float(
            (table.confidence_change_late_minus_early > 0).mean()
        ),
        "fraction_seeds_speed_up": float((table.speed_change_late_minus_early > 0).mean()),
        "fraction_seeds_rt_down": float((table.rt_change_late_minus_early < 0).mean()),
    })
    return result


def _context_controlled_coefficient(sequence: pd.DataFrame) -> float:
    frame = sequence.copy()
    progress = frame.global_sequence.to_numpy(dtype=float)
    progress = (progress - progress.mean()) / max(progress.std(), 1e-12)
    within = frame.sequence_in_block.to_numpy(dtype=float)
    within = (within - within.mean()) / max(within.std(), 1e-12)
    context_b = frame.context.eq("B").astype(float).to_numpy()
    seed_levels = sorted(frame.seed.unique())
    seed_dummies = np.column_stack([
        frame.seed.eq(seed).astype(float).to_numpy() for seed in seed_levels[1:]
    ]) if len(seed_levels) > 1 else np.empty((len(frame), 0))
    design = np.column_stack([np.ones(len(frame)), progress, within, context_b, seed_dummies])
    return float(np.linalg.lstsq(design, frame.normalized_rt, rcond=None)[0][3])


def _switch_structure(sequence: pd.DataFrame) -> dict:
    switch_jumps, within_jumps = [], []
    for _, group in sequence.groupby("seed", sort=True):
        group = group.sort_values("global_sequence").reset_index(drop=True)
        difference = group.normalized_rt.diff().abs()
        switched = group.context.ne(group.context.shift(1)) & group.block_index.ge(5)
        within = group.context.eq(group.context.shift(1)) & group.block_index.ge(5)
        switch_jumps.extend(difference[switched].dropna().tolist())
        within_jumps.extend(difference[within].dropna().tolist())
    switch_mean = float(np.mean(switch_jumps)) if switch_jumps else np.nan
    within_mean = float(np.mean(within_jumps)) if within_jumps else np.nan
    return {
        "late_switch_absolute_rt_jump": switch_mean,
        "late_within_context_absolute_rt_change": within_mean,
        "late_switch_excess_over_within": switch_mean - within_mean,
        "late_switches_evaluated": int(len(switch_jumps)),
    }


def evaluate_screening_gates(
    sequence: pd.DataFrame,
    tokens: pd.DataFrame,
    candidate: ObjectiveCandidate,
    config: SearchConfig,
) -> dict:
    """Return one fully auditable candidate-level gate record."""
    directions = _direction_changes(sequence, config)
    minimum_fraction = config.minimum_direction_seed_fraction
    gate_a = bool(
        directions["ce_change_late_minus_early"] < 0
        and directions["confidence_change_late_minus_early"] > 0
        and directions["rt_change_late_minus_early"] < 0
        and directions["fraction_seeds_ce_down"] >= minimum_fraction
        and directions["fraction_seeds_confidence_up"] >= minimum_fraction
        and directions["fraction_seeds_rt_down"] >= minimum_fraction
    )

    confidence_slopes, entropy_slopes, tracking = [], [], []
    for _, group in tokens.groupby("seed", sort=True):
        confidence_slopes.append(_slope(group.correct_token_confidence, group.normalized_rt))
        entropy_slopes.append(_slope(group.predictive_entropy, group.normalized_rt))
        tracking.append(_corr(group.objective_optimal_speed, group.execution_speed))
    confidence_slope = _finite_mean(confidence_slopes)
    entropy_slope = _finite_mean(entropy_slopes)
    fraction_confidence = _fraction(confidence_slopes, lambda value: value < 0)
    fraction_entropy = _fraction(entropy_slopes, lambda value: value > 0)
    gate_b = bool(
        (confidence_slope < 0 and fraction_confidence >= minimum_fraction)
        or (entropy_slope > 0 and fraction_entropy >= minimum_fraction)
    )
    tracking_correlation = _finite_mean(tracking)
    fraction_tracking = _fraction(tracking, lambda value: value > 0)
    gate_c = bool(tracking_correlation > 0 and fraction_tracking >= minimum_fraction)

    speed = tokens.execution_speed.to_numpy(dtype=float)
    low_saturation = float(np.mean(speed <= config.saturation_epsilon))
    high_saturation = float(np.mean(speed >= 1.0 - config.saturation_epsilon))
    total_saturation = low_saturation + high_saturation
    trained_range = float(np.quantile(speed, .95) - np.quantile(speed, .05))
    gate_d = bool(
        total_saturation < config.maximum_trained_saturation_fraction
        and trained_range >= config.minimum_trained_speed_range
    )

    switch = _switch_structure(sequence)
    context_coefficient = _context_controlled_coefficient(sequence)
    switch_signal = bool(
        np.isfinite(switch["late_switch_absolute_rt_jump"])
        and switch["late_switch_absolute_rt_jump"] >= config.minimum_switch_jump
        and switch["late_switch_excess_over_within"] > 0
    )
    context_signal = bool(abs(context_coefficient) >= config.minimum_context_coefficient)
    gate_e = bool(switch_signal or context_signal)

    passed = bool(gate_a and gate_b and gate_c and gate_d and gate_e)
    score = float(
        (tracking_correlation if np.isfinite(tracking_correlation) else -1)
        + trained_range - total_saturation
        + directions["fraction_seeds_rt_down"]
    )
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "seeds": int(sequence.seed.nunique()),
        "training_sequences_per_seed": int(sequence.global_sequence.nunique()),
        **directions,
        "confidence_rt_slope": confidence_slope,
        "entropy_rt_slope": entropy_slope,
        "fraction_seeds_confidence_rt_negative": fraction_confidence,
        "fraction_seeds_entropy_rt_positive": fraction_entropy,
        "speed_optimum_correlation": tracking_correlation,
        "fraction_seeds_positive_speed_optimum_correlation": fraction_tracking,
        "speed_low_saturation_fraction": low_saturation,
        "speed_high_saturation_fraction": high_saturation,
        "speed_total_saturation_fraction": total_saturation,
        "trained_speed_95_minus_05": trained_range,
        **switch,
        "context_B_rt_coefficient_controlling_progress_within_block": context_coefficient,
        "switch_signal_present": switch_signal,
        "context_signal_present": context_signal,
        "gate_A_learning_direction": gate_a,
        "gate_B_execution_grounding": gate_b,
        "gate_C_objective_tracking": gate_c,
        "gate_D_no_pathological_saturation": gate_d,
        "gate_E_contextual_dynamics": gate_e,
        "screening_passed": passed,
        "nonhuman_screen_quality_score": score,
        "human_probe_data_used": 0,
    }
