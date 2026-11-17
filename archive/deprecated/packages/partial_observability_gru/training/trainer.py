"""Reproducible on-policy meta-training and held-out evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..analysis.posterior_alignment import alignment_metrics, implied_responsibility
from ..environment.episode import EpisodeBatch, EpisodeConfig, generate_batch
from ..models import build_model
from ..models.stochastic_gru import StochasticGRU
from .losses import objective_loss


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 60
    batches_per_epoch: int = 8
    batch_size: int = 32
    learning_rate: float = 0.003
    gradient_clip: float = 1.0
    burn_in: int = 1
    device: str = "cpu"


def train_model(
    objective: str,
    hidden_dim: int,
    episode_config: EpisodeConfig,
    training_config: TrainingConfig,
    *,
    seed: int,
    progress=None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(objective, episode_config.input_dim, hidden_dim).to(training_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    history, best_loss, best_state = [], np.inf, None
    for epoch in range(max(1, training_config.epochs)):
        losses = []
        model.train()
        for batch_index in range(max(1, training_config.batches_per_epoch)):
            batch_seed = seed * 1_000_003 + epoch * 10_007 + batch_index
            batch = generate_batch(training_config.batch_size, episode_config, seed=batch_seed).to(
                training_config.device
            )
            optimizer.zero_grad()
            output = model(batch.inputs)
            loss = objective_loss(objective, output, batch, burn_in=training_config.burn_in)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if progress and (epoch == 0 or (epoch + 1) % max(1, training_config.epochs // 5) == 0):
            progress(f"epoch {epoch + 1}/{training_config.epochs}, loss={epoch_loss:.5f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def evaluate_model(model, objective: str, batch: EpisodeBatch, *, seed: int = 0):
    model.eval()
    with torch.no_grad():
        output = model(batch.inputs)
    hidden = output["hidden"].cpu().numpy()
    skill_a = batch.skill_a.cpu().numpy()
    skill_b = batch.skill_b.cpu().numpy()
    if objective == "classifier":
        q_direct = output["q"].cpu().numpy()
        prediction = q_direct * skill_a + (1.0 - q_direct) * skill_b
        prediction_source = "classifier_q_with_evaluation_only_true_skills"
    else:
        prediction = output["mean"].cpu().numpy()
        q_direct = output.get("q")
        q_direct = q_direct.cpu().numpy() if q_direct is not None else np.full_like(prediction, np.nan)
        prediction_source = "network_continuous_readout"
    sampled = np.full_like(prediction, np.nan)
    if objective == "stochastic":
        generator = torch.Generator(device=output["q"].device).manual_seed(int(seed))
        sampled = StochasticGRU.sample(output, generator).cpu().numpy()
    target = batch.targets.cpu().numpy()
    q_star = batch.q_star.cpu().numpy()
    y_star = batch.y_star.cpu().numpy()
    q_implied = implied_responsibility(prediction, skill_a, skill_b)
    metrics = alignment_metrics(prediction.ravel(), y_star.ravel(), skill_a.ravel(), skill_b.ravel(), q_star.ravel())
    predicted_sigma = (
        output["sigma"].cpu().numpy() if "sigma" in output else np.full_like(prediction, np.nan)
    )
    state_a_hat = (
        output["state_a"].cpu().numpy() if "state_a" in output else np.full_like(prediction, np.nan)
    )
    state_b_hat = (
        output["state_b"].cpu().numpy() if "state_b" in output else np.full_like(prediction, np.nan)
    )
    sigma_a = output["sigma_a"].cpu().numpy() if "sigma_a" in output else np.full_like(prediction, np.nan)
    sigma_b = output["sigma_b"].cpu().numpy() if "sigma_b" in output else np.full_like(prediction, np.nan)
    ambiguous = np.abs(q_star - 0.5) <= 0.1
    q_available = np.isfinite(q_direct).any()
    metrics.update(
        rmse=float(np.sqrt(np.mean((prediction - target) ** 2))),
        mae=float(np.mean(np.abs(prediction - target))),
        sampled_rmse=(
            float(np.sqrt(np.nanmean((sampled - target) ** 2)))
            if np.isfinite(sampled).any() else np.nan
        ),
        objective_loss=float(objective_loss(objective, output, batch, burn_in=0).cpu()),
        context_accuracy=(
            float(np.mean((q_direct >= 0.5) == (batch.contexts.cpu().numpy() == 0)))
            if q_available else np.nan
        ),
        mean_predicted_sigma=(float(np.nanmean(predicted_sigma)) if np.isfinite(predicted_sigma).any() else np.nan),
        ambiguous_predicted_sigma=(
            float(np.nanmean(predicted_sigma[ambiguous]))
            if ambiguous.any() and np.isfinite(predicted_sigma[ambiguous]).any() else np.nan
        ),
        prediction_source=prediction_source,
    )
    shape = prediction.shape
    frame = pd.DataFrame(
        {
            "episode_id": batch.episode_ids.cpu().numpy().ravel(),
            "time": np.tile(np.arange(shape[1]), shape[0]),
            "true_context": batch.contexts.cpu().numpy().ravel(),
            "observed_y": target.ravel(), "predicted_y": prediction.ravel(),
            "sampled_y": sampled.ravel(), "posterior_mean_y": y_star.ravel(),
            "q_star": q_star.ravel(), "q_direct": q_direct.ravel(), "q_implied": q_implied.ravel(),
            "skill_a": skill_a.ravel(), "skill_b": skill_b.ravel(),
            "predicted_sigma": predicted_sigma.ravel(),
            "component_a_hat": state_a_hat.ravel(), "component_b_hat": state_b_hat.ravel(),
            "component_sigma_a": sigma_a.ravel(), "component_sigma_b": sigma_b.ravel(),
        }
    )
    return frame, metrics, hidden
