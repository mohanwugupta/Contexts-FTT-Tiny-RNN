"""Leak-free PCA representation of participant A/B contrast profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


POSITIONS = tuple(range(1, 8))


def assert_pure_profile_contract(profiles: pd.DataFrame):
    required = {
        "participant_uid", "dataset", "transition_position", "delta",
        "pure_source_contexts", "probe_rows_used_for_profile",
    }
    missing = required - set(profiles)
    if missing:
        raise ValueError(f"Pure profile table is missing {sorted(missing)}.")
    if not profiles.pure_source_contexts.eq("A|B").all():
        raise ValueError("Pure basis source contains a non-A/B context.")
    if pd.to_numeric(profiles.probe_rows_used_for_profile).fillna(0).ne(0).any():
        raise ValueError("Pure basis source contains probe outcomes.")
    if set(profiles.transition_position.astype(int)) != set(POSITIONS):
        raise ValueError("Pure profiles must contain all seven registered transitions.")


def profile_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    assert_pure_profile_contract(profiles)
    matrix = profiles.pivot_table(
        index="participant_uid", columns="transition_position", values="delta", aggfunc="first"
    ).reindex(columns=POSITIONS)
    if matrix.isna().any().any():
        raise ValueError("Every participant must have a complete seven-transition A/B profile.")
    return matrix.sort_index()


def _orient(components: np.ndarray) -> np.ndarray:
    result = np.asarray(components, dtype=float).copy()
    for component in range(result.shape[1]):
        vector = result[:, component]
        anchor = int(np.argmax(np.abs(vector)))
        if vector[anchor] < 0:
            result[:, component] *= -1
    return result


@dataclass(frozen=True)
class LowRankBasis:
    mean: np.ndarray
    components: np.ndarray
    singular_values: np.ndarray
    variance_explained: np.ndarray
    cumulative_variance: np.ndarray
    rank: int
    training_participants: tuple[str, ...]
    training_datasets: tuple[str, ...]
    source_contexts: str = "A|B"
    probe_rows_used: int = 0

    def transform(self, matrix):
        values = np.asarray(matrix, dtype=float)
        return (values - self.mean) @ self.components

    def reconstruct(self, matrix):
        return self.mean + self.transform(matrix) @ self.components.T


def fit_pca_basis(profiles: pd.DataFrame, *, rank: int) -> LowRankBasis:
    matrix = profile_matrix(profiles)
    rank = int(rank)
    if not 1 <= rank <= matrix.shape[1]:
        raise ValueError("PCA rank exceeds the seven-transition feature space.")
    values = matrix.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    centered = values - mean
    _, singular_raw, vt = np.linalg.svd(centered, full_matrices=True)
    singular = np.zeros(matrix.shape[1], dtype=float)
    singular[:len(singular_raw)] = singular_raw
    components = _orient(vt[:rank].T)
    energy = singular ** 2
    variance = energy / energy.sum() if energy.sum() > 1e-20 else np.zeros_like(energy)
    datasets = tuple(sorted(profiles.dataset.astype(str).unique()))
    return LowRankBasis(
        mean=mean,
        components=components,
        singular_values=singular,
        variance_explained=variance,
        cumulative_variance=np.cumsum(variance),
        rank=rank,
        training_participants=tuple(matrix.index.astype(str)),
        training_datasets=datasets,
    )
