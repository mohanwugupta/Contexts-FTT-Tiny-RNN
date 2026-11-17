import numpy as np

from gru_partial_observability_v1.analysis.bayes_risk import mean_risk, sampling_risk
from gru_partial_observability_v1.environment.bayes_filter import context_posterior
from gru_partial_observability_v1.environment.contexts import ContextConfig, bayes_accuracy, sample_context
from gru_partial_observability_v1.environment.episodes import V1EpisodeConfig, build_inputs, generate_batch
from gru_partial_observability_v1.environment.latent_states import LatentStateParameters, LatentStates


def test_v1_latent_states_update_only_the_active_state_reproducibly():
    parameters = LatentStateParameters(0.3, 0.7, 0.1, 0.2, 0.1, 0.9)
    first, second = LatentStates(parameters), LatentStates(parameters)
    b_before = first.s_b
    first.practice(0)
    second.practice(0)
    assert first.s_b == b_before
    assert first.values == second.values
    a_before = first.s_a
    first.practice(1)
    assert first.s_a == a_before
    assert first.s_b != b_before


def test_v1_context_means_and_accuracy_contract():
    rng = np.random.default_rng(3)
    config = ContextConfig(noise_sd=0.4)
    a = np.array([sample_context(0, config, rng) for _ in range(5000)])
    b = np.array([sample_context(1, config, rng) for _ in range(5000)])
    assert a.mean() < b.mean()
    assert abs(a.mean() + 1) < 0.03
    assert abs(b.mean() - 1) < 0.03
    assert bayes_accuracy(0.2) > bayes_accuracy(0.8) > bayes_accuracy(1.4)


def test_v1_no_current_target_leakage_and_exact_posterior():
    observations = np.array([-1.0, 1.0, 0.0])
    outcomes = np.array([0.3, 0.7, 0.5])
    inputs = build_inputs(observations, outcomes)
    outcomes[-1] = 999
    assert inputs[-1, 1] == 0.7
    assert context_posterior(0.0, 0.5, 1.0) == 0.5
    assert context_posterior(-1.0, 0.5, 1.0) > 0.5
    assert context_posterior(1.0, 0.5, 1.0) < 0.5
    batch = generate_batch(2, V1EpisodeConfig(length=8), seed=5)
    assert batch.inputs.shape == (2, 8, 4)


def test_v1_posterior_mean_has_half_sampling_risk():
    assert np.isclose(mean_risk(0.3, 0.7, 0.5), 0.04)
    assert np.isclose(sampling_risk(0.3, 0.7, 0.5), 0.08)
    for q in (0.05, 0.25, 0.75, 0.95):
        assert np.isclose(sampling_risk(0.3, 0.7, q), 2 * mean_risk(0.3, 0.7, q))
