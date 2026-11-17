"""Separate the shared PC1 projection from token-specific A/B residuals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .local_decomposition import numbered_columns


@dataclass(frozen=True)
class SharedContextAxis:
    unit_vector: np.ndarray
    singular_values: np.ndarray
    variance_explained: np.ndarray


def fit_shared_context_axis(decomposition: pd.DataFrame) -> SharedContextAxis:
    columns = numbered_columns(decomposition, "d")
    matrix = decomposition[columns].to_numpy(dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("Shared-axis estimation requires at least two token differences.")
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    axis = vt[0].copy()
    # SVD sign is arbitrary. Orient PC1 toward the mean natural B->A component.
    if float(matrix.mean(axis=0) @ axis) < 0:
        axis *= -1.0
    energy = singular ** 2
    variance = energy / energy.sum() if energy.sum() > 1e-20 else np.zeros_like(energy)
    return SharedContextAxis(axis, singular, variance)


def add_shared_specific_components(
    decomposition: pd.DataFrame, axis: SharedContextAxis
) -> pd.DataFrame:
    result = decomposition.copy()
    columns = numbered_columns(result, "d")
    matrix = result[columns].to_numpy(dtype=float)
    shared = np.outer(matrix @ axis.unit_vector, axis.unit_vector)
    specific = matrix - shared
    if not np.allclose(shared + specific, matrix, atol=1e-10):
        raise AssertionError("Shared and specific components do not reconstruct local d.")
    if not np.allclose(specific @ axis.unit_vector, 0.0, atol=1e-10):
        raise AssertionError("Token-specific residual is not orthogonal to shared PC1.")
    for index in range(matrix.shape[1]):
        result[f"shared_d{index}"] = shared[:, index]
        result[f"specific_d{index}"] = specific[:, index]
        result[f"shared_axis_h{index}"] = axis.unit_vector[index]
    result["shared_context_norm"] = np.linalg.norm(shared, axis=1)
    result["specific_context_norm"] = np.linalg.norm(specific, axis=1)
    result["pc1_variance_explained"] = float(axis.variance_explained[0])
    return result

