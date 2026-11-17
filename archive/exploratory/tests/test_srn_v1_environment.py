import numpy as np

from srn_partial_observability_v1.environment.bayes_filter import (
    BayesFilter,
    context_posterior,
)
from srn_partial_observability_v1.environment.contexts import bayes_accuracy
from srn_partial_observability_v1.environment.episodes import (
    EpisodeConfig,
    build_inputs,
    generate_batch,
)


def test_srn_environment_has_two_independent_states_and_exact_input_contract():
    batch = generate_batch(12, EpisodeConfig(length=30, context_noise_sd=1.1), seed=8)
    assert batch.inputs.shape == (12, 30, 4)
    np.testing.assert_allclose(batch.inputs[:, 0, 1].numpy(), 0.0)
    np.testing.assert_allclose(batch.inputs[:, 0, 2].numpy(), 0.0)
    np.testing.assert_allclose(batch.inputs[:, 0, 3].numpy(), 1.0)
    # Only the active state changes after a trial; the inactive state is held.
    a = batch.skill_a.numpy()
    b = batch.skill_b.numpy()
    z = batch.contexts.numpy()
    np.testing.assert_allclose(np.diff(a, axis=1)[z[:, :-1] == 1], 0.0)
    np.testing.assert_allclose(np.diff(b, axis=1)[z[:, :-1] == 0], 0.0)


def test_current_target_never_enters_current_input():
    x = np.array([-1.0, 0.2, 1.0])
    y = np.array([0.3, 0.6, 0.8])
    original = build_inputs(x, y)
    changed = build_inputs(x, np.array([0.3, 99.0, 0.8]))
    np.testing.assert_allclose(original[1], changed[1])
    assert original[2, 1] != changed[2, 1]


def test_context_noise_and_exact_bayes_filter_are_calibrated():
    assert bayes_accuracy(0.0) == 1.0
    assert 0.70 <= bayes_accuracy(1.1) <= 0.85
    assert context_posterior(0.0, 0.5, 1.1) == 0.5
    benchmark = BayesFilter(context_noise_sd=1.1, persistence=0.85, outcome_noise_sd=0.02)
    q = benchmark.predict(0.0, transition=False)
    assert q == 0.5

