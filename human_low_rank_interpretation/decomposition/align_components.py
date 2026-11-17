"""Align fold/bootstrap components despite arbitrary signs and ordering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AlignmentAudit:
    reference_component: np.ndarray
    candidate_component: np.ndarray
    matched_cosine: np.ndarray
    sign_flipped: np.ndarray


def _cosine_matrix(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    denominator = np.outer(
        np.linalg.norm(reference, axis=0), np.linalg.norm(candidate, axis=0)
    )
    return np.divide(
        reference.T @ candidate, denominator,
        out=np.zeros_like(reference.T @ candidate), where=denominator > 1e-14,
    )


def align_components(reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate component matrices must have equal shape.")
    cosine = _cosine_matrix(reference, candidate)
    reference_indices, candidate_indices = linear_sum_assignment(-np.abs(cosine))
    order = candidate_indices[np.argsort(reference_indices)]
    aligned = candidate[:, order].copy()
    matched = np.array([cosine[index, order[index]] for index in range(reference.shape[1])])
    signs = matched < 0
    aligned[:, signs] *= -1
    return aligned, AlignmentAudit(
        reference_component=np.arange(1, reference.shape[1] + 1),
        candidate_component=order + 1,
        matched_cosine=np.abs(matched),
        sign_flipped=signs,
    )
