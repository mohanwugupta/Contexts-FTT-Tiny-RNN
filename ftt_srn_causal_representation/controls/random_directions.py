"""Deterministic norm-matched orthogonal negative-control directions."""

import numpy as np


def orthogonal_random_direction(reference, subspace_basis, *, rng):
    reference = np.asarray(reference, dtype=float)
    basis = np.asarray(subspace_basis, dtype=float)
    target_norm = float(np.linalg.norm(reference))
    if target_norm <= 1e-12:
        return np.zeros_like(reference)
    random = rng.normal(size=reference.shape)
    if basis.size:
        random = random - basis @ (basis.T @ random)
    norm = float(np.linalg.norm(random))
    if norm <= 1e-10:
        unit = reference / target_norm
        random = rng.normal(size=reference.shape)
        random -= unit * float(random @ unit)
        norm = float(np.linalg.norm(random))
    if norm <= 1e-10:
        raise ValueError("No orthogonal random-control direction is available.")
    return random / norm * target_norm


def orthogonal_random_subspace(reference_basis, *, rng):
    """Return a dimension-matched subspace orthogonal to the reference if possible."""
    reference_basis = np.asarray(reference_basis, dtype=float)
    hidden_dim, requested = reference_basis.shape
    projector = np.eye(hidden_dim) - reference_basis @ reference_basis.T
    available = hidden_dim - requested
    dimension = min(requested, max(available, 1))
    candidate = projector @ rng.normal(size=(hidden_dim, max(dimension, requested)))
    q, singular, _ = np.linalg.svd(candidate, full_matrices=False)
    rank = int(np.sum(singular > 1e-9))
    if rank:
        return q[:, :min(dimension, rank)]
    # A full-rank reference has no orthogonal complement; use a random
    # dimension-matched basis and mark non-orthogonality in downstream output.
    q, _ = np.linalg.qr(rng.normal(size=(hidden_dim, requested)))
    return q[:, :requested]
