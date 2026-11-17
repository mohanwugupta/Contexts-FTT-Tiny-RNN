import torch

from partial_observability_gru.environment.episode import EpisodeConfig, generate_batch
from partial_observability_gru.models import build_model
from partial_observability_gru.training.losses import objective_loss
from partial_observability_gru.training.trainer import TrainingConfig, evaluate_model, train_model


def test_all_objective_heads_have_finite_losses_and_expected_contracts():
    batch = generate_batch(4, EpisodeConfig(length=10), seed=11)
    for objective in ("deterministic", "gaussian", "classifier", "stochastic", "distributional"):
        model = build_model(objective, batch.inputs.shape[-1], hidden_dim=3)
        output = model(batch.inputs)
        loss = objective_loss(objective, output, batch)
        assert torch.isfinite(loss)
        assert output["hidden"].shape == (4, 10, 3)
        if objective == "classifier":
            assert output["q"].min() >= 0 and output["q"].max() <= 1
        else:
            assert output["mean"].shape == batch.targets.shape


def test_target_changes_do_not_change_same_time_prediction():
    batch = generate_batch(2, EpisodeConfig(length=8), seed=12)
    model = build_model("deterministic", batch.inputs.shape[-1], hidden_dim=2)
    before = model(batch.inputs)["mean"].detach().clone()
    batch.targets[:, 4] += 100
    after = model(batch.inputs)["mean"].detach()
    torch.testing.assert_close(before, after)


def test_deterministic_and_classifier_grus_learn_tractable_clear_context_task():
    config = EpisodeConfig(length=14, uncertainty="low", skill_separation="large")
    training = TrainingConfig(
        epochs=12, batches_per_epoch=3, batch_size=24, learning_rate=0.006
    )
    evaluation = generate_batch(32, config, seed=445)
    deterministic, _ = train_model("deterministic", 8, config, training, seed=8)
    _, continuous_metrics, _ = evaluate_model(deterministic, "deterministic", evaluation)
    classifier, _ = train_model("classifier", 8, config, training, seed=8)
    _, classifier_metrics, _ = evaluate_model(classifier, "classifier", evaluation)
    assert continuous_metrics["rmse"] < 0.75
    assert continuous_metrics["posterior_mean_r2"] > 0.2
    assert classifier_metrics["belief_r2"] > 0.1
