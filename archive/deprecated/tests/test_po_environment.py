import numpy as np

from partial_observability_gru.analysis.bayes_risk import mean_squared_risk, sampling_squared_risk
from partial_observability_gru.environment.bayes_filter import BayesFilter, context_posterior
from partial_observability_gru.environment.context_generator import ContextGenerator
from partial_observability_gru.environment.episode import EpisodeConfig, build_model_inputs, generate_batch
from partial_observability_gru.environment.latent_skills import LatentSkills, SkillParameters


def test_latent_skill_updates_are_independent_and_seeded():
    params = SkillParameters(0.0, 1.0, 0.2, 0.3, -1.0, 2.0, 0.01)
    first = LatentSkills(params, seed=7)
    second = LatentSkills(params, seed=7)
    b_before = first.s_b
    first.practice(0)
    second.practice(0)
    assert first.s_b == b_before
    assert first.s_a == second.s_a
    a_before = first.s_a
    first.practice(1)
    assert first.s_a == a_before
    assert first.s_b != b_before


def test_context_noise_reduces_bayes_accuracy_and_midpoint_is_ambiguous():
    generator = ContextGenerator.random(context_dim=2, separation=2.0, noise_sd=0.2, seed=3)
    midpoint = 0.5 * (generator.mu_a + generator.mu_b)
    assert context_posterior(midpoint, 0.5, generator.mu_a, generator.mu_b, 0.2) == 0.5
    assert context_posterior(generator.mu_a, 0.5, generator.mu_a, generator.mu_b, 0.2) > 0.99
    assert generator.theoretical_accuracy(0.1) > generator.theoretical_accuracy(0.8)


def test_exact_filter_uses_feedback_only_after_current_prediction():
    benchmark = BayesFilter(
        np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
        context_noise_sd=0.5, switch_probability=0.1, performance_noise_sd=0.05,
    )
    q_before_feedback = benchmark.predict_context(np.zeros(2), transition=False)
    assert q_before_feedback == 0.5
    benchmark.observe_performance(observed_y=0.02, skill_a=0.0, skill_b=1.0)
    assert benchmark.belief_a > 0.99


def test_mean_prediction_has_half_the_sampling_risk_at_ambiguity():
    assert mean_squared_risk(400, 600, 0.5) == 10_000
    assert sampling_squared_risk(400, 600, 0.5) == 20_000
    for q in (0.1, 0.25, 0.7, 0.9):
        assert np.isclose(sampling_squared_risk(1, 5, q), 2 * mean_squared_risk(1, 5, q))


def test_inputs_use_only_lagged_feedback_and_training_has_no_ood_probes():
    observations = np.arange(12, dtype=float).reshape(4, 3)
    target = np.array([10.0, 20.0, 30.0, 40.0])
    inputs = build_model_inputs(observations, target)
    target[2] = 999.0
    assert inputs[2, -3] == 20.0
    assert inputs[2, -3] != target[2]
    batch = generate_batch(3, EpisodeConfig(length=8), seed=4)
    assert set(batch.probe_type.ravel()) == {"training"}
    assert batch.inputs.shape[-1] == EpisodeConfig().input_dim
