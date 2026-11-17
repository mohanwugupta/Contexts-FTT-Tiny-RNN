import numpy as np

from gru_partial_observability_v1.analysis.ambiguity_tests import evaluate_ambiguity, generate_ambiguity_batch
from gru_partial_observability_v1.analysis.hidden_probes import fit_hidden_probes
from gru_partial_observability_v1.analysis.interventions import intervene_direction
from gru_partial_observability_v1.environment.episodes import V1EpisodeConfig, generate_batch
from gru_partial_observability_v1.training.trainer import TrainingConfig, evaluate_predictor, train_predictor


def test_v1_gru_beats_simple_baselines_on_clear_contexts():
    episode = V1EpisodeConfig(length=25, context_noise_sd=0.15)
    training = TrainingConfig(epochs=65, batches_per_epoch=6, batch_size=64, learning_rate=0.006)
    model, _ = train_predictor(episode, training, seed=7)
    evaluation = generate_batch(128, episode, seed=901)
    _, metrics, _ = evaluate_predictor(model, evaluation, training_mean=0.5)
    assert metrics["model_mse"] < metrics["global_mean_mse"]
    assert metrics["model_mse"] < metrics["previous_outcome_mse"]


def test_v1_ambiguity_batch_hits_requested_posteriors_and_reports_unclipped_q():
    levels = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
    batch = generate_ambiguity_batch(levels, episodes_per_level=3, history_length=10, seed=2)
    np.testing.assert_allclose(np.sort(np.unique(batch.q_star[:, -1])), levels, atol=1e-5)
    model, _ = train_predictor(
        V1EpisodeConfig(length=12, context_noise_sd=1.1),
        TrainingConfig(epochs=2, batches_per_epoch=2, batch_size=6), seed=2,
    )
    rows, summary, _ = evaluate_ambiguity(model, batch)
    np.testing.assert_allclose(np.sort(rows.q_star.unique()), levels, atol=1e-5)
    assert "q_implied_unclipped" in rows
    assert {"posterior_mean_r2", "responsibility_slope", "responsibility_intercept"}.issubset(summary)


def test_v1_hidden_probe_splits_by_episode_and_intervention_is_exact():
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(300, 3))
    hidden = latent @ rng.normal(size=(3, 5))
    episodes = np.repeat(np.arange(30), 10)
    probes = fit_hidden_probes(
        hidden, {"S_A": latent[:, 0], "S_B": latent[:, 1], "q_star": latent[:, 2]}, episodes, seed=5
    )
    assert probes.heldout_r2.min() > 0.98
    assert (probes.train_episode_count + probes.test_episode_count).eq(30).all()
    moved = intervene_direction(np.zeros((2, 3)), np.array([2.0, 0.0, 0.0]), 0.0, 0.75)
    np.testing.assert_allclose(moved @ np.array([2.0, 0.0, 0.0]), 0.75)
