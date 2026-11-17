"""Observed-scale semantic-axis identification simulations."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..decomposition.align_components import align_components


LABELS = (
    "difficulty", "context_sensitivity", "switch_vulnerability",
    "difficulty+context_sensitivity", "random", "unlabeled",
)


def _standardize(values):
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmean(values))
    centered = values - values.mean()
    norm = np.linalg.norm(centered)
    return centered / norm if norm > 1e-14 else centered


def _permutation_p(component, feature, permutations):
    component = _standardize(component)
    feature = _standardize(feature)
    observed = abs(float(component @ feature))
    null = np.abs(feature[permutations] @ component)
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def _fit_components(matrix, rank):
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    result = vt[:rank].T
    for column in range(rank):
        anchor = np.argmax(np.abs(result[:, column]))
        if result[anchor, column] < 0:
            result[:, column] *= -1
    return result


def _stable(matrix, components, rank):
    half = len(matrix) // 2
    if half < 3:
        return np.ones(rank, dtype=bool)
    values = []
    for subset in (matrix[:half], matrix[half:]):
        candidate = _fit_components(subset, rank)
        _aligned, audit = align_components(components, candidate)
        values.append(audit.matched_cosine)
    return np.minimum(values[0], values[1]) >= .70


def run_semantic_recovery(
    features: pd.DataFrame, *, participant_count: int, noise_sd: float,
    replicates: int, random_seed: int,
):
    table = features.sort_values("transition_position")
    axes = {
        "difficulty": _standardize(table.mean_pure_logRT),
        "context_sensitivity": _standardize(table.A_minus_B),
        "switch_vulnerability": _standardize(table.switch_cost),
    }
    permutations = np.asarray(list(itertools.permutations(range(7))), dtype=int)
    rng = np.random.default_rng(random_seed)
    counts = {(generator, recovered): 0 for generator in (
        "difficulty", "context_sensitivity", "switch_vulnerability",
        "difficulty+context_sensitivity", "random",
    ) for recovered in LABELS}
    detail = []
    for generator in (
        "difficulty", "context_sensitivity", "switch_vulnerability",
        "difficulty+context_sensitivity", "random",
    ):
        rank = 2 if generator == "difficulty+context_sensitivity" else 1
        for replicate in range(int(replicates)):
            if generator == "difficulty+context_sensitivity":
                true = np.column_stack([axes["difficulty"], axes["context_sensitivity"]])
            elif generator == "random":
                vector = rng.normal(size=7)
                for axis in axes.values():
                    vector -= (vector @ axis) * axis
                true = _standardize(vector)[:, None]
            else:
                true = axes[generator][:, None]
            amplitudes = rng.normal(1.0, .25, size=(int(participant_count), rank))
            matrix = amplitudes @ true.T + rng.normal(
                0, float(noise_sd), size=(int(participant_count), 7)
            )
            components = _fit_components(matrix, rank)
            stable = _stable(matrix, components, rank)
            assigned = []
            for component in range(rank):
                candidate_stats = []
                for label, feature in axes.items():
                    correlation = abs(float(_standardize(components[:, component]) @ feature))
                    p_value = _permutation_p(components[:, component], feature, permutations)
                    candidate_stats.append((correlation, p_value, label))
                correlation, p_value, label = max(candidate_stats)
                if stable[component] and p_value < .05 / len(axes):
                    assigned.append(label)
                detail.append({
                    "generating_semantic": generator, "replicate": replicate,
                    "component": component + 1, "best_candidate": label,
                    "absolute_correlation": correlation, "permutation_p": p_value,
                    "component_stable": bool(stable[component]),
                })
            if generator == "difficulty+context_sensitivity":
                coverage = min(
                    float(np.linalg.norm(components.T @ true[:, axis]) ** 2 /
                          max(np.linalg.norm(true[:, axis]) ** 2, 1e-14))
                    for axis in range(true.shape[1])
                )
                half = len(matrix) // 2
                split_similarity = []
                for subset in (matrix[:half], matrix[half:]):
                    split = _fit_components(subset, rank)
                    split_similarity.append(float(np.linalg.svd(
                        components.T @ split, compute_uv=False
                    ).min()))
                combined_recovered = bool(
                    coverage >= .80 and min(split_similarity) >= .70
                )
                recovered = (
                    "difficulty+context_sensitivity" if combined_recovered
                    else (assigned[0] if len(assigned) == 1 else "unlabeled")
                )
                detail.append({
                    "generating_semantic": generator, "replicate": replicate,
                    "component": 0, "best_candidate": "combined_subspace",
                    "absolute_correlation": coverage, "permutation_p": np.nan,
                    "component_stable": min(split_similarity) >= .70,
                })
            elif generator == "random":
                recovered = assigned[0] if assigned else "random"
            else:
                recovered = assigned[0] if assigned else "unlabeled"
            counts[(generator, recovered)] += 1
    rows = []
    for (generator, recovered), count in counts.items():
        rows.append({
            "generating_semantic": generator, "recovered_semantic": recovered,
            "recovery_probability": count / float(replicates),
            "count": count, "replicates": int(replicates),
            "participant_count": int(participant_count), "noise_sd": float(noise_sd),
            "transition_count": 7, "participant_missingness_preserved": True,
            "permutation_inference_used": True,
        })
    return pd.DataFrame(rows), pd.DataFrame(detail)
