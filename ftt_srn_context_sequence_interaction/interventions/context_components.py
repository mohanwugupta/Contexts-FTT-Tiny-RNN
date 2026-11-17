"""Patching and ablation constrained to locally observed context components."""

from __future__ import annotations

import numpy as np


def _projection(vector, component):
    vector = np.asarray(vector, dtype=float)
    component = np.asarray(component, dtype=float)
    denominator = float(component @ component)
    if denominator <= 1e-20:
        return np.zeros_like(vector)
    return component * float(vector @ component) / denominator


def patch_natural_component(recipient, midpoint, component, *, sign: int):
    """Replace one local component while preserving its orthogonal residual."""
    if sign not in {-1, 1}:
        raise ValueError("Patch sign must be +1 (A) or -1 (B).")
    recipient = np.asarray(recipient, dtype=float)
    midpoint = np.asarray(midpoint, dtype=float)
    component = np.asarray(component, dtype=float)
    centered = recipient - midpoint
    residual = centered - _projection(centered, component)
    patched = midpoint + float(sign) * component + residual
    patched_residual = patched - midpoint - _projection(patched - midpoint, component)
    if not np.allclose(patched_residual, residual, atol=1e-10):
        raise AssertionError("Natural patch did not preserve the orthogonal recipient residual.")
    return patched


def patch_wrong_token_component(
    recipient, midpoint, target_component, wrong_component, *, sign: int
):
    """Insert a norm-matched wrong-token component while fixing target residual."""
    if sign not in {-1, 1}:
        raise ValueError("Patch sign must be +1 (A) or -1 (B).")
    recipient = np.asarray(recipient, dtype=float)
    midpoint = np.asarray(midpoint, dtype=float)
    target_component = np.asarray(target_component, dtype=float)
    wrong_component = np.asarray(wrong_component, dtype=float)
    centered = recipient - midpoint
    target_residual = centered - _projection(centered, target_component)
    return midpoint + float(sign) * wrong_component + target_residual


def ablate_natural_component(hidden, midpoint, component):
    hidden = np.asarray(hidden, dtype=float)
    midpoint = np.asarray(midpoint, dtype=float)
    return hidden - _projection(hidden - midpoint, component)


def wrong_token_positions(positions, *, target_position: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in positions if int(value) != int(target_position))
    if int(target_position) in result:
        raise AssertionError("Wrong-token control included its target token.")
    return result


def norm_match_wrong_token(reference, wrong_token):
    reference = np.asarray(reference, dtype=float)
    wrong_token = np.asarray(wrong_token, dtype=float)
    reference_norm = float(np.linalg.norm(reference))
    wrong_norm = float(np.linalg.norm(wrong_token))
    if reference_norm <= 1e-20:
        return np.zeros_like(reference)
    if wrong_norm <= 1e-20:
        raise ValueError("Cannot norm-match a zero wrong-token component.")
    matched = wrong_token * reference_norm / wrong_norm
    if not np.isclose(np.linalg.norm(matched), reference_norm, atol=1e-10):
        raise AssertionError("Wrong-token component norm matching failed.")
    return matched
