"""Within-network dimensionality of token-specific A/B differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ABSubspace:
    basis: np.ndarray
    dimension: int
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    singular_values: np.ndarray
    effective_dimensionality: float


def _direction_columns(frame):
    return sorted(
        [column for column in frame if column.startswith("d") and column[1:].isdigit()],
        key=lambda value: int(value[1:]),
    )


def fit_ab_subspace(directions, *, variance_threshold: float = .90) -> ABSubspace:
    columns = _direction_columns(directions)
    matrix = directions[columns].to_numpy(dtype=float)
    if not len(matrix) or not columns:
        raise ValueError("At least one nonempty A/B difference vector is required.")
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    power = singular ** 2
    total = float(power.sum())
    if total <= 1e-16:
        raise ValueError("A/B directions have zero variance/norm.")
    variance = power / total
    cumulative = np.cumsum(variance)
    dimension = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    basis = vh[:dimension].T
    effective = float(total ** 2 / np.square(power).sum())
    return ABSubspace(
        basis=basis,
        dimension=dimension,
        variance_explained=variance,
        cumulative_variance=cumulative,
        singular_values=singular,
        effective_dimensionality=effective,
    )
