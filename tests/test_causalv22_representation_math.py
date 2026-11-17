import numpy as np
import pandas as pd

from ftt_srn_causal_representation.interventions.components import (
    ablate_component,
    patch_component,
)
from ftt_srn_causal_representation.representation.ab_subspace import fit_ab_subspace
from ftt_srn_causal_representation.representation.matched_states import estimate_ab_directions


def _matched_states(role="estimation"):
    rows = []
    for context, sign in (("A", 1.0), ("B", -1.0)):
        for position in range(1, 4):
            rows.append({
                "pass_role": role,
                "pass_id": f"{role}-pass",
                "context": context,
                "token_position": position,
                "target_token": position % 4,
                "h0": sign * position,
                "h1": sign * (position % 2),
                "h2": 0.25 * position,
            })
    return pd.DataFrame(rows)


def test_ab_directions_are_token_matched_and_oriented_toward_a():
    directions = estimate_ab_directions(_matched_states())
    assert list(directions.token_position) == [1, 2, 3]
    assert (directions.estimation_pass_id == "estimation-pass").all()
    assert (directions.target_a == directions.target_b).all()
    assert (directions.d0 > 0).all()
    np.testing.assert_allclose(
        directions.ab_distance,
        np.sqrt(directions.d0 ** 2 + directions.d1 ** 2 + directions.d2 ** 2),
    )


def test_ablation_removes_only_parallel_component_and_patching_preserves_residual():
    basis = np.array([[1.0], [0.0], [0.0]])
    midpoint = np.array([2.0, 3.0, 4.0])
    hidden = midpoint + np.array([5.0, 7.0, -2.0])
    ablated = ablate_component(hidden, midpoint, basis)
    np.testing.assert_allclose(ablated, midpoint + [0.0, 7.0, -2.0])

    mixed = midpoint + np.array([1.5, 8.0, -3.0])
    donor = midpoint + np.array([-4.0, 2.0, 9.0])
    patched = patch_component(mixed, donor, midpoint, basis)
    np.testing.assert_allclose(patched, midpoint + [-4.0, 8.0, -3.0])
    mixed_residual = (np.eye(3) - basis @ basis.T) @ (mixed - midpoint)
    patched_residual = (np.eye(3) - basis @ basis.T) @ (patched - midpoint)
    np.testing.assert_allclose(patched_residual, mixed_residual)


def test_subspace_uses_smallest_dimension_reaching_ninety_percent():
    # Two orthogonal dimensions are required: PC1 explains only 80%.
    directions = pd.DataFrame({
        "token_position": [1, 2],
        "d0": [2.0, 0.0],
        "d1": [0.0, 1.0],
        "d2": [0.0, 0.0],
    })
    subspace = fit_ab_subspace(directions, variance_threshold=.90)
    assert subspace.dimension == 2
    assert subspace.cumulative_variance[0] == .8
    assert subspace.cumulative_variance[1] == 1.0
    np.testing.assert_allclose(subspace.basis.T @ subspace.basis, np.eye(2), atol=1e-7)
