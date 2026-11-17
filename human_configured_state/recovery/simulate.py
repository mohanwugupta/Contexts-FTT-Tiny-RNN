"""Capacity-penalized recovery on observed-scale profile templates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fit_template(y, template):
    template = np.asarray(template, dtype=float)
    if float(np.std(template)) < 1e-12:
        x = np.ones((len(y), 1))
    else:
        x = np.column_stack([np.ones(len(y)), template])
    beta = np.linalg.pinv(x.T @ x) @ x.T @ y
    residual = y - x @ beta
    sse = max(float(residual @ residual), 1e-12)
    bic = len(y) * np.log(sse / len(y)) + x.shape[1] * np.log(len(y))
    return bic


def run_recovery(
    templates, *, participant_count: int, observations_per_position: int,
    noise_sd: float, replicates: int, random_seed: int,
    observation_counts=None, endpoint_scale=None, effect_scale: float = 1.0,
):
    names = tuple(templates)
    positions = len(next(iter(templates.values())))
    rng = np.random.default_rng(random_seed)
    scale = np.ones(positions) if endpoint_scale is None else np.asarray(endpoint_scale, dtype=float)
    if len(scale) != positions:
        raise ValueError("endpoint_scale must have one value per transition position.")
    if observation_counts is None:
        counts = np.full((int(participant_count), positions), int(observations_per_position))
    else:
        counts = np.asarray(observation_counts, dtype=int)
        if counts.ndim != 2 or counts.shape[1] != positions:
            raise ValueError("observation_counts must be participant by transition position.")
    wins = {(generator, candidate): 0 for generator in names for candidate in names}
    bic_totals = {(generator, candidate): [] for generator in names for candidate in names}
    for generator in names:
        generating = np.asarray(templates[generator], dtype=float)
        for _ in range(int(replicates)):
            values = []
            template_rows = {candidate: [] for candidate in names}
            for participant_index in range(len(counts)):
                amplitude = rng.normal(1.0, 0.15)
                intercept = rng.normal(0.0, noise_sd)
                for position in range(positions):
                    for _observation in range(int(counts[participant_index, position])):
                        values.append(
                            intercept + amplitude * effect_scale * scale[position] *
                            generating[position] + rng.normal(0, noise_sd)
                        )
                        for candidate in names:
                            template_rows[candidate].append(
                                effect_scale * scale[position] * np.asarray(templates[candidate])[position]
                            )
            bics = {
                candidate: _fit_template(np.asarray(values), template_rows[candidate])
                for candidate in names
            }
            winner = min(bics, key=bics.get)
            wins[(generator, winner)] += 1
            for candidate, value in bics.items():
                bic_totals[(generator, candidate)].append(value)
    rows = []
    for generator in names:
        for candidate in names:
            rows.append({
                "generating_model": generator, "candidate_model": candidate,
                "recovery_probability": wins[(generator, candidate)] / float(replicates),
                "mean_bic": float(np.mean(bic_totals[(generator, candidate)])),
                "replicates": int(replicates),
                "observed_count_pattern_used": observation_counts is not None,
                "endpoint_scale_used": endpoint_scale is not None,
                "effect_scale": float(effect_scale),
            })
    matrix = pd.DataFrame(rows)
    diagonal = matrix[matrix.generating_model.eq(matrix.candidate_model)].copy()
    summary = diagonal.rename(columns={
        "generating_model": "model", "recovery_probability": "diagonal_recovery_probability"
    })[["model", "diagonal_recovery_probability", "replicates"]]
    chance = 1.0 / len(names)
    summary["chance_probability"] = chance
    summary["recovery_above_chance"] = summary.diagonal_recovery_probability > chance
    summary["strong_interpretation_allowed"] = summary.recovery_above_chance
    return matrix, summary
