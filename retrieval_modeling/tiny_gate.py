"""Shared low-dimensional mixture-of-experts gate for Both/Novel/Default."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from probe_modeling.metrics import regression_metrics

from .context_features import context_matrix
from .context_geometry import geometry_arrays


class LogisticRetrievalGate(nn.Module):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = 0
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(features).squeeze(-1))


class RecurrentRetrievalGate(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 1):
        super().__init__()
        if hidden_dim not in {1, 2}:
            raise ValueError("PRD3 limits the recurrent retrieval gate to 1D or 2D.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.gru(features)
        return torch.sigmoid(self.readout(hidden).squeeze(-1)), hidden


def _prepare_batch(data: pd.DataFrame):
    groups = [
        group.sort_values("global_sequence", kind="stable")
        for _, group in data.groupby(["dataset", "participant_id"], sort=False)
    ]
    maximum = max(len(group) for group in groups)
    n = len(groups)
    features = np.zeros((n, maximum, 5), dtype=np.float32)
    a = np.zeros((n, maximum), dtype=np.float32)
    b = np.zeros((n, maximum), dtype=np.float32)
    y = np.zeros((n, maximum), dtype=np.float32)
    indices = np.full((n, maximum), -1, dtype=int)
    keys = []
    for batch, group in enumerate(groups):
        keys.append((str(group.dataset.iloc[0]), str(group.participant_id.iloc[0])))
        previous = None
        for time, (index, row) in enumerate(group.iterrows()):
            cue = context_matrix([row.retrieval_context])[0]
            features[batch, time, :3] = cue
            features[batch, time, 3] = float(previous == "A")
            features[batch, time, 4] = float(previous == "B")
            a[batch, time] = row.pred_A_log_rt
            b[batch, time] = row.pred_B_log_rt
            y[batch, time] = row.log_rt
            indices[batch, time] = index
            if row.retrieval_context in {"A", "B"}:
                previous = row.retrieval_context
    return features, a, b, y, indices, keys


def _fit_network(model, features, a, b, y, mask, *, seed, epochs, learning_rate):
    torch.manual_seed(seed)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    x = torch.as_tensor(features, dtype=torch.float32)
    expert_a = torch.as_tensor(a, dtype=torch.float32)
    expert_b = torch.as_tensor(b, dtype=torch.float32)
    observed = torch.as_tensor(y, dtype=torch.float32)
    keep = torch.as_tensor(mask, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_state, stale = np.inf, None, 0
    patience = max(20, epochs // 5)
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        output = model(x)
        q = output[0] if isinstance(output, tuple) else output
        prediction = q * expert_a + (1 - q) * expert_b
        loss = torch.mean((prediction[keep] - observed[keep]) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss - 1e-9:
            best_loss = value
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def fit_tiny_gate_ladder(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    seeds: Iterable[int] = (42,),
    epochs: int = 300,
    learning_rate: float = 0.02,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, a, b, y, indices, keys = _prepare_batch(data)
    split_lookup = {
        (str(row.dataset), str(row.participant_id)): row.split for row in splits.itertuples()
    }
    train_mask = np.zeros(indices.shape, dtype=bool)
    test_mask = np.zeros(indices.shape, dtype=bool)
    for batch, key in enumerate(keys):
        valid = indices[batch] >= 0
        if split_lookup.get(key, "test") == "train":
            train_mask[batch] = valid
        else:
            test_mask[batch] = valid
    if not train_mask.any():
        train_mask = indices >= 0
    if not test_mask.any():
        test_mask = indices >= 0
    fit_rows, state_rows = [], []
    for seed in seeds:
        specifications = (
            ("logistic_gate", LogisticRetrievalGate(5), 0),
            ("recurrent_gate_1d", RecurrentRetrievalGate(5, 1), 1),
            ("recurrent_gate_2d", RecurrentRetrievalGate(5, 2), 2),
        )
        for name, model, hidden_dim in specifications:
            if progress:
                progress(f"Tiny retrieval gate: seed {seed}, {name}")
            train_loss = _fit_network(
                model, features, a, b, y, train_mask,
                seed=int(seed), epochs=int(epochs), learning_rate=float(learning_rate),
            )
            model.eval()
            with torch.no_grad():
                output = model(torch.as_tensor(features, dtype=torch.float32))
                if isinstance(output, tuple):
                    q_tensor, hidden_tensor = output
                else:
                    q_tensor = output
                    hidden_tensor = torch.zeros((*q_tensor.shape, 0))
            q = q_tensor.numpy()
            hidden = hidden_tensor.numpy()
            prediction = q * a + (1 - q) * b
            metrics = regression_metrics(y[test_mask], prediction[test_mask])
            fit_rows.append(
                {
                    "model": name, "seed": int(seed), "hidden_dim": hidden_dim,
                    "shared_across_datasets": True, "train_loss": train_loss,
                    **{f"heldout_{key}": value for key, value in metrics.items()},
                }
            )
            for position in zip(*np.where(indices >= 0)):
                index = int(indices[position])
                row = data.loc[index]
                geometry = geometry_arrays([row.retrieval_context], beta=1.0)
                split = split_lookup.get((str(row.dataset), str(row.participant_id)), "test")
                state_rows.append(
                    {
                        "source_index": index, "dataset": row.dataset,
                        "participant_id": row.participant_id, "global_sequence": row.global_sequence,
                        "retrieval_context": row.retrieval_context, "model": name,
                        "seed": int(seed), "hidden_dim": hidden_dim, "split": split,
                        "q_A": float(q[position]),
                        "predicted_log_rt": float(prediction[position]),
                        "observed_log_rt": float(y[position]),
                        "hidden_state": json.dumps(hidden[position].tolist()),
                        "history_state": float(hidden[position][0]) if hidden_dim else 0.0,
                        "relative_evidence_R": float(geometry["relative_evidence_R"][0]),
                        "absolute_evidence_E": float(geometry["absolute_evidence_E"][0]),
                    }
                )
    return pd.DataFrame(fit_rows), pd.DataFrame(state_rows)
