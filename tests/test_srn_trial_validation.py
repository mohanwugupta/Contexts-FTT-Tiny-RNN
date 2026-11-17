import numpy as np
import pandas as pd

from srn_validation.pipeline import prepare_average_trace, validate_training_dynamics
from srn_validation.state_space import (
    fit_three_models,
    simulate_lump,
    simulate_two_context,
)


def test_lump_and_two_context_simulators_match_r_reference_updates():
    y = np.array([2.0, 1.0, 0.5, 0.25])
    np.testing.assert_allclose(simulate_lump(0.5, y), [1.0, 0.5, 0.25, 0.125])

    states = simulate_two_context(
        a_a=0.5,
        a_b=0.25,
        a_cross=0.0,
        y=y,
        contexts=np.array(["A", "A", "B", "B"]),
    )
    np.testing.assert_allclose(states.readout, [2.0, 1.0, 2.0, 0.5])
    np.testing.assert_allclose(states.a_state, [1.0, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(states.b_state, [2.0, 2.0, 0.5, 0.125])


def _training_fixture():
    rows = []
    for condition, active in (("High", True), ("High_seq_only", False)):
        for seed in range(3):
            for step in range(1, 21):
                block = 1 if step <= 10 else 2
                context = "A" if block == 1 else "B"
                speed = 0.2 + 0.01 * step + 0.005 * seed
                rows.append({
                    "seed": seed,
                    "model_condition": condition,
                    "global_sequence": step,
                    "block_index": block,
                    "sequence_in_block": (step - 1) % 10 + 1,
                    "context": context,
                    "sequence_ce": 1.5 * np.exp(-0.05 * step) + 0.01 * seed,
                    "total_loss": 1.6 * np.exp(-0.05 * step) + 0.01 * seed,
                    "execution_speed": speed,
                    "normalized_rt": 1.0 - speed,
                    "execution_objective_active": active,
                })
    return pd.DataFrame(rows)


def test_training_contract_and_average_trace_preserve_trials_and_seed_uncertainty():
    frame = _training_fixture()
    audit = validate_training_dynamics(frame, expected_sequences=20)
    assert audit["n_seeds_per_condition"] == {"High": 3, "High_seq_only": 3}
    assert audit["max_rt_identity_error"] < 1e-12

    trace = prepare_average_trace(frame, "High", "sequence_ce")
    assert trace.global_sequence.tolist() == list(range(1, 21))
    assert trace.context.tolist() == ["A"] * 10 + ["B"] * 10
    assert (trace.n_seeds == 3).all()
    assert (trace.se > 0).all()


def test_three_state_space_models_return_trial_aligned_predictions_and_metrics():
    frame = _training_fixture()
    trace = prepare_average_trace(frame, "High", "sequence_ce")
    fits = fit_three_models(trace.observed.to_numpy(), trace.context.to_numpy())

    assert set(fits) == {"Lump", "Cross", "Split"}
    for name, fit in fits.items():
        assert fit.name == name
        assert fit.predicted.shape == (20,)
        assert np.isfinite(fit.predicted).all()
        assert np.isfinite(fit.rmse)
        assert np.isfinite(fit.bic)
        assert fit.parameter_count == {"Lump": 3, "Cross": 5, "Split": 4}[name]


def test_training_contract_rejects_an_incorrect_rt_identity():
    frame = _training_fixture()
    frame.loc[0, "normalized_rt"] += 0.1
    try:
        validate_training_dynamics(frame, expected_sequences=20)
    except ValueError as error:
        assert "normalized_rt" in str(error)
    else:
        raise AssertionError("An incorrect normalized-RT identity must be rejected")
