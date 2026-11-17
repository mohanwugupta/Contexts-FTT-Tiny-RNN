import numpy as np
import pandas as pd
import pytest


from state_space_bridge.state_space import (
    MODEL_SPECS,
    fit_model_family,
    simulate_lump,
    simulate_two_context,
)


def _trace(include_probe=False):
    contexts = ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10
    frame = pd.DataFrame({
        "context": contexts,
        "observed": np.linspace(1.0, 0.45, len(contexts)),
        "global_step": np.arange(1, len(contexts) + 1),
        "block_id": np.repeat(np.arange(1, 5), 10),
        "sequence_in_block": np.tile(np.arange(1, 11), 4),
    })
    if include_probe:
        probe = frame.iloc[[0]].copy()
        probe["context"] = "Mixed"
        probe["observed"] = 999.0
        probe["global_step"] = 21
        frame = pd.concat([frame, probe], ignore_index=True)
    return frame


def test_r_port_has_exact_model_bounds_and_bic_counts():
    assert MODEL_SPECS["SS_LUMP"].bounds == ((0.60, 0.999),)
    assert MODEL_SPECS["SS_CROSS"].bounds == (
        (0.60, 0.999), (0.00, 0.50), (0.60, 0.999)
    )
    assert MODEL_SPECS["SS_SPLIT"].bounds == ((0.60, 0.999), (0.60, 0.999))
    assert {key: value.bic_parameters for key, value in MODEL_SPECS.items()} == {
        "SS_LUMP": 3, "SS_CROSS": 5, "SS_SPLIT": 4,
    }


def test_lump_and_two_context_match_hand_calculation():
    y = np.array([2.0, 1.0, 0.5])
    lump = simulate_lump(np.array([0.5]), y)
    np.testing.assert_allclose(lump["x_readout"], [1.0, 0.5, 0.25])

    two = simulate_two_context(
        np.array([0.8, 0.1, 0.7]), y, np.array(["A", "B", "A"])
    )
    # The R model reads the state before the gated update.
    np.testing.assert_allclose(two["x_readout"], [2.0, 2.0, 1.8])
    np.testing.assert_allclose(two["x_a"], [1.8, 1.8, 1.598])
    np.testing.assert_allclose(two["x_b"], [2.0, 1.58, 1.58])


def test_cross_boundary_is_split():
    y = np.linspace(1, 0.2, 20)
    contexts = np.array(["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5)
    cross = simulate_two_context(np.array([0.91, 0.0, 0.83]), y, contexts)
    split = simulate_two_context(
        np.array([0.91, 0.83]), y, contexts, cross_talk_fixed_zero=True
    )
    for key in ("x_readout", "x_a", "x_b"):
        np.testing.assert_allclose(cross[key], split[key], atol=1e-12, rtol=0)


def test_probe_rows_are_ignored_and_cannot_change_fit():
    base = fit_model_family(_trace(False))
    contaminated = fit_model_family(_trace(True))
    for model in MODEL_SPECS:
        np.testing.assert_allclose(base[model].parameters, contaminated[model].parameters)
        np.testing.assert_allclose(base[model].predicted, contaminated[model].predicted)
        assert base[model].bic == contaminated[model].bic


def test_unknown_non_ab_context_is_never_simulated():
    with pytest.raises(ValueError, match="pure A/B"):
        simulate_two_context(
            np.array([0.9, 0.1, 0.9]), np.array([1.0, 0.9]), np.array(["A", "Mixed"])
        )
