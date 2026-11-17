import numpy as np
import pandas as pd

from ftt_srn_context_sequence_interaction.interventions.context_components import (
    ablate_natural_component,
    norm_match_wrong_token,
    patch_natural_component,
    wrong_token_positions,
)
from ftt_srn_context_sequence_interaction.representation.local_decomposition import (
    build_local_decomposition,
)
from ftt_srn_context_sequence_interaction.representation.shared_context_axis import (
    add_shared_specific_components,
    fit_shared_context_axis,
)


def _states():
    rows = []
    for context, sign in (("A", 1.0), ("B", -1.0)):
        for position, d in ((1, [2.0, 0.0, 0.0]),
                            (2, [1.0, 1.0, 0.0]),
                            (3, [1.0, -1.0, 0.0])):
            mu = np.array([0.25 * position, 2.0, -0.5])
            hidden = mu + sign * np.asarray(d)
            rows.append({
                "pass_role": "estimation", "pass_id": "estimation-1",
                "context": context, "token_position": position,
                "target_token": position % 4,
                **{f"h{i}": value for i, value in enumerate(hidden)},
            })
    return pd.DataFrame(rows)


def test_local_decomposition_exactly_reconstructs_token_matched_a_and_b():
    states = _states()
    decomposition = build_local_decomposition(states)
    for row in decomposition.itertuples(index=False):
        mu = np.array([getattr(row, f"mu{i}") for i in range(3)])
        d = np.array([getattr(row, f"d{i}") for i in range(3)])
        h_a = np.array([getattr(row, f"h_a{i}") for i in range(3)])
        h_b = np.array([getattr(row, f"h_b{i}") for i in range(3)])
        np.testing.assert_allclose(mu + d, h_a)
        np.testing.assert_allclose(mu - d, h_b)
        assert row.target_a == row.target_b


def test_shared_and_specific_components_reconstruct_d_and_are_orthogonal():
    decomposition = build_local_decomposition(_states())
    axis = fit_shared_context_axis(decomposition)
    components = add_shared_specific_components(decomposition, axis)
    for row in components.itertuples(index=False):
        d = np.array([getattr(row, f"d{i}") for i in range(3)])
        shared = np.array([getattr(row, f"shared_d{i}") for i in range(3)])
        specific = np.array([getattr(row, f"specific_d{i}") for i in range(3)])
        np.testing.assert_allclose(shared + specific, d, atol=1e-10)
        assert abs(float(specific @ axis.unit_vector)) < 1e-10


def test_natural_patch_preserves_residual_and_identity_patch_is_exact():
    midpoint = np.array([1.0, 2.0, 3.0])
    component = np.array([2.0, 0.0, 0.0])
    recipient = midpoint + np.array([0.5, 4.0, -3.0])
    patched = patch_natural_component(recipient, midpoint, component, sign=1)
    np.testing.assert_allclose(patched, midpoint + [2.0, 4.0, -3.0])
    np.testing.assert_allclose(patched[1:] - midpoint[1:], recipient[1:] - midpoint[1:])

    natural_a = midpoint + component
    natural_b = midpoint - component
    np.testing.assert_allclose(
        patch_natural_component(natural_a, midpoint, component, sign=1), natural_a
    )
    np.testing.assert_allclose(
        patch_natural_component(natural_b, midpoint, component, sign=-1), natural_b
    )
    np.testing.assert_allclose(ablate_natural_component(natural_a, midpoint, component), midpoint)


def test_wrong_token_control_excludes_target_and_norm_matches():
    assert wrong_token_positions([1, 2, 3], target_position=2) == (1, 3)
    matched = norm_match_wrong_token(np.array([3.0, 4.0, 0.0]), np.array([0.0, 2.0, 0.0]))
    assert np.isclose(np.linalg.norm(matched), 5.0)

