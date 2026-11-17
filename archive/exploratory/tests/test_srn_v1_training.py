from srn_partial_observability_v1.environment.episodes import EpisodeConfig, generate_batch
from srn_partial_observability_v1.training.trainer import (
    TrainingConfig,
    evaluate_predictor,
    train_predictor,
)


def test_elman_srn_learns_clear_context_prediction_and_beats_all_simple_baselines():
    episode = EpisodeConfig(length=25, context_noise_sd=0.15, context_persistence=0.65)
    model, result = train_predictor(
        episode,
        TrainingConfig(epochs=70, batches_per_epoch=6, batch_size=64, learning_rate=0.006),
        seed=7,
    )
    evaluation = generate_batch(128, episode, seed=901)
    _, metrics, _ = evaluate_predictor(model, evaluation, result.baselines)
    assert metrics["model_mse"] < metrics["global_mean_mse"]
    assert metrics["model_mse"] < metrics["previous_outcome_mse"]
    assert metrics["model_mse"] < metrics["context_only_mse"]

