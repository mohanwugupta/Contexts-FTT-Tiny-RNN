"""Exact projection, ablation, and component-patching operations."""

import numpy as np


def projection(vector, basis):
    vector = np.asarray(vector, dtype=float)
    basis = np.asarray(basis, dtype=float)
    return basis @ (basis.T @ vector)


def ablate_component(hidden, midpoint, basis):
    hidden = np.asarray(hidden, dtype=float)
    midpoint = np.asarray(midpoint, dtype=float)
    return hidden - projection(hidden - midpoint, basis)


def patch_component(recipient, donor, midpoint, basis):
    recipient = np.asarray(recipient, dtype=float)
    donor = np.asarray(donor, dtype=float)
    midpoint = np.asarray(midpoint, dtype=float)
    return recipient - projection(recipient - midpoint, basis) + projection(donor - midpoint, basis)
