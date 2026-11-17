"""Deterministic MSE training shared by the SRN and optional GRU control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..environment.episodes import EpisodeBatch, EpisodeConfig, generate_batch
from ..models import ElmanSRN, GRUControl
from .baselines import (
    FittedBaselines,
    context_only_prediction,
    fit_context_only,
    global_mean_prediction,
    previous_outcome_prediction,
    regression_metrics,
)


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 4
    epochs: int = 65
    batches_per_epoch: int = 6
    batch_size: int = 64
    learning_rate: float = 0.006
    burn_in: int = 1
    gradient_clip: float = 1.0


@dataclass
class TrainingResult:
    history: pd.DataFrame
    baselines: FittedBaselines
    best_train_mse: float
    architecture: str

    def __getattr__(self, name):
        # Compatibility convenience: result.training_target_mean behaves like a history column.
        if name in self.history.columns:
            return self.history[name]
        raise AttributeError(name)


def _build_model(architecture: str, hidden_dim: int):
    architecture = architecture.lower()
    if architecture == "srn":
        return ElmanSRN(input_dim=4, hidden_dim=hidden_dim)
    if architecture == "gru":
        return GRUControl(input_dim=4, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown architecture {architecture!r}.")


def train_predictor(
    episode_config: EpisodeConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    architecture: str = "srn",
    progress=None,
):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    model = _build_model(architecture, training_config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    best_loss = np.inf
    best_state = None
    history = []
    all_x, all_y = [], []
    running_sum = 0.0
    running_n = 0
    for epoch in range(max(1, training_config.epochs)):
        losses = []
        model.train()
        for batch_index in range(max(1, training_config.batches_per_epoch)):
            batch = generate_batch(
                training_config.batch_size,
                episode_config,
                seed=int(seed) * 1_000_003 + epoch * 10_007 + batch_index,
            )
            keep = slice(max(0, training_config.burn_in), None)
            x = batch.observations[:, keep].numpy().ravel()
            y = batch.targets[:, keep].numpy().ravel()
            all_x.append(x)
            all_y.append(y)
            running_sum += float(y.sum())
            running_n += int(y.size)
            optimizer.zero_grad()
            prediction, _, _ = model(batch.inputs)
            loss = torch.mean((prediction[:, keep] - batch.targets[:, keep]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        value = float(np.mean(losses))
        history.append({
            "epoch": epoch + 1,
            "train_mse": value,
            "training_target_mean": running_sum / running_n,
        })
        if value < best_loss:
            best_loss = value
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }
        if progress and (
            epoch == 0 or (epoch + 1) % max(1, training_config.epochs // 5) == 0
        ):
            progress(f"epoch {epoch + 1}/{training_config.epochs}, mse={value:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    context_intercept, context_slope = fit_context_only(
        np.concatenate(all_x), np.concatenate(all_y)
    )
    baselines = FittedBaselines(
        training_mean=float(running_sum / running_n),
        context_intercept=context_intercept,
        context_slope=context_slope,
    )
    result = TrainingResult(
        history=pd.DataFrame(history),
        baselines=baselines,
        best_train_mse=float(best_loss),
        architecture=architecture.upper(),
    )
    return model, result


def evaluate_predictor(model, batch: EpisodeBatch, baselines: FittedBaselines):
    model.eval()
    with torch.no_grad():
        prediction, hidden, _ = model(batch.inputs)
    predicted = prediction.numpy()
    target = batch.targets.numpy()
    inputs = batch.inputs.numpy()
    global_prediction = global_mean_prediction(target.shape, baselines.training_mean)
    previous_prediction = previous_outcome_prediction(inputs, baselines.training_mean)
    context_prediction = context_only_prediction(
        batch.observations.numpy(), baselines.context_intercept, baselines.context_slope
    )
    oracle_prediction = batch.y_star.numpy()
    predictions = {
        "model": predicted,
        "global_mean": global_prediction,
        "previous_outcome": previous_prediction,
        "context_only": context_prediction,
        "oracle_bayes": oracle_prediction,
    }
    metrics = {
        f"{name}_{metric}": value
        for name, values in predictions.items()
        for metric, value in regression_metrics(target, values).items()
    }
    n_episode, n_time = target.shape
    rows = pd.DataFrame({
        "episode_id": batch.episode_ids.numpy().ravel(),
        "time": np.tile(np.arange(n_time), n_episode),
        "x": batch.observations.numpy().ravel(),
        "true_context": batch.contexts.numpy().ravel(),
        "skill_a": batch.skill_a.numpy().ravel(),
        "skill_b": batch.skill_b.numpy().ravel(),
        "q_star": batch.q_star.numpy().ravel(),
        "posterior_mean": batch.y_star.numpy().ravel(),
        "observed_y": target.ravel(),
        "predicted_y": predicted.ravel(),
        "global_mean_prediction": global_prediction.ravel(),
        "previous_outcome_prediction": previous_prediction.ravel(),
        "context_only_prediction": context_prediction.ravel(),
        "oracle_bayes_prediction": oracle_prediction.ravel(),
        "time_since_switch": batch.time_since_switch.numpy().ravel(),
    })
    return rows, metrics, hidden.numpy()

