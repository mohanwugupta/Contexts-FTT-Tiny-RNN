"""Deterministic MSE training and held-out baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..environment.episodes import V1Batch, V1EpisodeConfig, generate_batch
from ..models import GRUPredictor
from .baselines import global_mean_prediction, mean_squared_error, previous_outcome_prediction


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 4
    epochs: int = 50
    batches_per_epoch: int = 6
    batch_size: int = 32
    learning_rate: float = 0.004
    burn_in: int = 1
    gradient_clip: float = 1.0


def train_predictor(
    episode_config: V1EpisodeConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    progress=None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GRUPredictor(input_dim=4, hidden_dim=training_config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    best_loss, best_state, history = np.inf, None, []
    running_sum, running_n = 0.0, 0
    for epoch in range(max(1, training_config.epochs)):
        losses = []
        model.train()
        for batch_index in range(max(1, training_config.batches_per_epoch)):
            batch = generate_batch(
                training_config.batch_size, episode_config,
                seed=seed * 1_000_003 + epoch * 10_007 + batch_index,
            )
            running_sum += float(batch.targets.sum())
            running_n += int(batch.targets.numel())
            optimizer.zero_grad()
            prediction, _, _ = model(batch.inputs)
            keep = slice(max(0, training_config.burn_in), None)
            loss = torch.mean((prediction[:, keep] - batch.targets[:, keep]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        value = float(np.mean(losses))
        history.append(
            {"epoch": epoch + 1, "train_mse": value, "training_target_mean": running_sum / running_n}
        )
        if value < best_loss:
            best_loss = value
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        if progress and (epoch == 0 or (epoch + 1) % max(1, training_config.epochs // 5) == 0):
            progress(f"epoch {epoch + 1}/{training_config.epochs}, mse={value:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def evaluate_predictor(model: GRUPredictor, batch: V1Batch, training_mean: float):
    model.eval()
    with torch.no_grad():
        prediction, hidden, _ = model(batch.inputs)
    predicted = prediction.numpy()
    target = batch.targets.numpy()
    inputs = batch.inputs.numpy()
    mean_prediction = global_mean_prediction(target.shape, training_mean)
    previous_prediction = previous_outcome_prediction(inputs, training_mean)
    oracle_prediction = batch.y_star.numpy()
    metrics = {
        "model_mse": mean_squared_error(target, predicted),
        "model_rmse": float(np.sqrt(mean_squared_error(target, predicted))),
        "global_mean_mse": mean_squared_error(target, mean_prediction),
        "previous_outcome_mse": mean_squared_error(target, previous_prediction),
        "oracle_bayes_mse": mean_squared_error(target, oracle_prediction),
    }
    n_episode, n_time = target.shape
    rows = pd.DataFrame(
        {
            "episode_id": batch.episode_ids.numpy().ravel(),
            "time": np.tile(np.arange(n_time), n_episode),
            "x": batch.observations.numpy().ravel(),
            "true_context": batch.contexts.numpy().ravel(),
            "skill_a": batch.skill_a.numpy().ravel(), "skill_b": batch.skill_b.numpy().ravel(),
            "q_star": batch.q_star.numpy().ravel(), "posterior_mean": batch.y_star.numpy().ravel(),
            "observed_y": target.ravel(), "predicted_y": predicted.ravel(),
            "global_mean_prediction": mean_prediction.ravel(),
            "previous_outcome_prediction": previous_prediction.ravel(),
            "oracle_prediction": oracle_prediction.ravel(),
            "time_since_switch": batch.time_since_switch.numpy().ravel(),
        }
    )
    return rows, metrics, hidden.numpy()
