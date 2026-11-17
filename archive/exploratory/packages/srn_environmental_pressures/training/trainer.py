"""Fixed-cadence online training through one exact FTT schedule."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..config import EnvironmentalCondition, ExperimentConfig
from ..environment.simulation import generate_batch
from ..models import ElmanSRN


@dataclass
class TrainingResult:
    model: ElmanSRN
    training_history: pd.DataFrame
    checkpoints: dict[int, dict[str, torch.Tensor]]
    weight_updates: int


def train_condition(condition: EnvironmentalCondition, config: ExperimentConfig, *, seed: int,
                    evaluate_checkpoints: bool = True, progress=None) -> TrainingResult:
    del evaluate_checkpoints  # Checkpoint evaluation is deliberately analysis-side.
    condition.validate()
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    model = ElmanSRN(input_dim=4, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    batch = generate_batch(
        config.training_batch_size,
        condition,
        n_blocks=config.n_blocks,
        block_length=config.block_length,
        seed=int(seed) * 1_000_003 + 17,
        cue_mean=config.cue_mean,
    )
    hidden = None
    checkpoints = {}
    history = []
    model.train()
    for time in range(config.training_trials):
        optimizer.zero_grad()
        prediction, _, next_hidden = model(batch.inputs[:, time:time + 1], hidden)
        loss = torch.mean((prediction[:, 0] - batch.targets[:, time]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        hidden = next_hidden.detach()
        history.append({
            "update": time + 1,
            "trial": time + 1,
            "block_index": time // config.block_length + 1,
            "block_position": time % config.block_length + 1,
            "nominal_context": "A" if (time // config.block_length) % 2 == 0 else "B",
            "train_mse": float(loss.detach()),
        })
        if (time + 1) % (2 * config.block_length) == 0:
            pair = (time + 1) // (2 * config.block_length)
            checkpoints[pair] = copy.deepcopy(model.state_dict())
            if progress:
                progress(
                    f"completed A/B pair {pair}/{config.n_block_pairs}, "
                    f"update {time + 1}/{config.weight_updates}, mse={float(loss.detach()):.6f}"
                )
    model.eval()
    return TrainingResult(model, pd.DataFrame(history), checkpoints, config.weight_updates)
