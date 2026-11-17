"""Training, ablations, interventions, and distillation for tiny gates."""

from __future__ import annotations

import json
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from .metrics import gate_entropy, regression_metrics
from .prediction import cue_features
from .tiny_gate_rnn import NonRecurrentGate, TinyGateRNN


def _tensor(value, *, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype)


def fit_gate(
    model: torch.nn.Module,
    features,
    a_states,
    b_states,
    observed,
    mask,
    *,
    epochs: int = 300,
    learning_rate: float = 0.02,
    seed: int = 42,
) -> tuple[torch.nn.Module, float]:
    """Fit fixed network weights; only recurrent state evolves at prediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    x = _tensor(features)
    a = _tensor(a_states)
    b = _tensor(b_states)
    y = _tensor(observed)
    keep = torch.as_tensor(mask, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = np.inf
    best_state = None
    patience = max(30, epochs // 5)
    stale = 0
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad()
        output = model(x)
        q = output[0] if isinstance(output, tuple) else output
        prediction = q * a + (1.0 - q) * b
        loss = torch.mean((prediction[keep] - y[keep]) ** 2)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss - 1e-9:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_loss)


def predict_gate(
    model: torch.nn.Module,
    features,
    initial_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    x = _tensor(features)
    h0 = None if initial_state is None else _tensor(initial_state)
    model.eval()
    with torch.no_grad():
        if isinstance(model, TinyGateRNN):
            q, hidden = model(x, h0)
        else:
            q = model(x)
            hidden = torch.zeros(*q.shape, 0, dtype=q.dtype)
    return q.cpu().numpy(), hidden.cpu().numpy()


FEATURE_SETS = {
    "cue": ("cue_A", "cue_B"),
    "cue_novel": ("cue_A", "cue_B", "novel"),
    "full": ("cue_A", "cue_B", "block_start", "previous_A", "previous_B", "novel"),
}


def _feature_row(row: pd.Series, names: Sequence[str], block_start: bool) -> list[float]:
    cue_a, cue_b, novel = cue_features(row)
    values = {
        "cue_A": cue_a,
        "cue_B": cue_b,
        "block_start": float(block_start),
        "previous_A": float(row.previous_context == "A"),
        "previous_B": float(row.previous_context == "B"),
        "novel": float(novel),
    }
    return [values[name] for name in names]


def prepare_gate_batch(
    data: pd.DataFrame,
    *,
    feature_set: str = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Pad participant timelines into a batch and retain source-row indices."""
    names = list(FEATURE_SETS[feature_set])
    participants = [
        group.sort_values("global_sequence", kind="stable")
        for _, group in data.groupby("participant_id", sort=False)
    ]
    max_length = max(len(group) for group in participants)
    shape = (len(participants), max_length)
    features = np.zeros((*shape, len(names)), dtype=np.float32)
    a = np.zeros(shape, dtype=np.float32)
    b = np.zeros(shape, dtype=np.float32)
    y = np.zeros(shape, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    indices = np.full(shape, -1, dtype=int)
    participant_ids = []
    for batch_i, group in enumerate(participants):
        participant_ids.append(str(group.participant_id.iloc[0]))
        previous_block = None
        for time_i, (index, row) in enumerate(group.iterrows()):
            start = previous_block is None or row.block_id != previous_block
            features[batch_i, time_i] = _feature_row(row, names, start)
            a[batch_i, time_i] = row.pred_A_log_rt
            b[batch_i, time_i] = row.pred_B_log_rt
            y[batch_i, time_i] = row.log_rt
            valid[batch_i, time_i] = np.isfinite(row.log_rt)
            indices[batch_i, time_i] = index
            previous_block = row.block_id
    return features, a, b, y, valid, indices, participant_ids


def _training_mask(
    data: pd.DataFrame,
    indices: np.ndarray,
    train_participants: set[str],
) -> np.ndarray:
    calibration = data.context.isin(["A", "B"]) | (
        data.is_probe & data.is_primary_probe & data.probe_block_number.eq(1)
    )
    allowed = calibration & data.participant_id.isin(train_participants)
    mask = np.zeros(indices.shape, dtype=bool)
    for position in zip(*np.where(indices >= 0)):
        source_index = indices[position]
        mask[position] = bool(allowed.loc[source_index])
    return mask


def run_cue_interpolation(
    model: torch.nn.Module,
    *,
    a_state: float,
    b_state: float,
    hidden_state: Optional[Iterable[float]] = None,
    cue_values: Iterable[float] = (1.0, 0.75, 0.5, 0.25, 0.0),
    feature_template: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    rows = []
    input_dim = model.input_dim
    template = np.zeros(input_dim, dtype=np.float32)
    if feature_template is not None:
        supplied = np.asarray(list(feature_template), dtype=np.float32)
        template[: min(len(supplied), input_dim)] = supplied[:input_dim]
    h0 = None if hidden_state is None else _tensor(np.asarray(hidden_state, dtype=np.float32))
    for cue_a in cue_values:
        cue_b = 1.0 - float(cue_a)
        vector = template.copy()
        vector[:2] = [cue_a, cue_b]
        if isinstance(model, TinyGateRNN):
            with torch.no_grad():
                q_tensor, next_hidden = model.step(_tensor(vector), h0)
            q = float(q_tensor.item())
            next_h = next_hidden.squeeze(0).cpu().numpy()
            start_h = np.zeros_like(next_h) if h0 is None else h0.cpu().numpy().reshape(-1)
            delta = next_h - start_h
        else:
            with torch.no_grad():
                q = float(model(_tensor(vector)).item())
            next_h = np.array([], dtype=float)
            delta = np.array([], dtype=float)
        rows.append(
            {
                "cue_A": float(cue_a),
                "cue_B": cue_b,
                "hidden_state": json.dumps([] if hidden_state is None else list(hidden_state)),
                "q": q,
                "predicted_rt": q * float(a_state) + (1 - q) * float(b_state),
                "delta_hidden": json.dumps(delta.tolist()),
                "next_hidden": json.dumps(next_h.tolist()),
            }
        )
    return pd.DataFrame(rows)


def run_state_intervention(
    model: TinyGateRNN,
    *,
    a_state: float,
    b_state: float,
    states: Iterable[float] = (-2.0, 0.0, 2.0),
    probe_cue: tuple[float, float] = (0.5, 0.5),
) -> pd.DataFrame:
    if model.hidden_dim != 1:
        raise ValueError("State intervention is defined for the interpretable 1D gate.")
    rows = []
    for value in states:
        vector = np.zeros(model.input_dim, dtype=np.float32)
        vector[:2] = probe_cue
        with torch.no_grad():
            q, next_hidden = model.step(_tensor(vector), _tensor([value]))
        q_value = float(q.item())
        next_value = float(next_hidden.item())
        rows.append(
            {
                "intervened_hidden": float(value),
                "cue_A": probe_cue[0],
                "cue_B": probe_cue[1],
                "q": q_value,
                "predicted_rt": q_value * a_state + (1 - q_value) * b_state,
                "next_hidden": next_value,
                "delta_hidden": next_value - float(value),
            }
        )
    return pd.DataFrame(rows)


def distill_one_dimensional_dynamics(
    features: np.ndarray,
    hidden: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Regress h(t) on h(t-1) and current gate features."""
    if hidden.shape[-1] != 1:
        raise ValueError("Dynamical distillation requires hidden_dim=1.")
    predictors, outcomes = [], []
    for batch_i in range(hidden.shape[0]):
        previous = 0.0
        for time_i in range(hidden.shape[1]):
            if not mask[batch_i, time_i]:
                continue
            current = float(hidden[batch_i, time_i, 0])
            predictors.append([previous, *features[batch_i, time_i].tolist()])
            outcomes.append(current)
            previous = current
    regression = LinearRegression().fit(np.asarray(predictors), np.asarray(outcomes))
    return {
        "distillation_r2": float(regression.score(np.asarray(predictors), np.asarray(outcomes))),
        "distillation_intercept": float(regression.intercept_),
        "distillation_coefficients": json.dumps(regression.coef_.tolist()),
    }


def fit_gate_ladder(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    seeds: Iterable[int] = (42,),
    epochs: int = 300,
    learning_rate: float = 0.02,
    model_version: str = "0.1.0",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit G1-G4 ablations and 1/2/4-unit predictive ceilings."""
    fit_rows, state_frames, cue_frames, intervention_frames = [], [], [], []
    specifications = [
        ("nonrecurrent_cue", "cue", 0),
        ("gru_cue", "cue", 1),
        ("gru_cue_novel", "cue_novel", 1),
        ("gru_full", "full", 1),
        ("gru_full", "full", 2),
        ("gru_full", "full", 4),
    ]
    for dataset, dataset_data in data.groupby("dataset", sort=False):
        dataset_splits = splits[splits.dataset.eq(dataset)]
        train_ids = set(dataset_splits.loc[dataset_splits.split.eq("train"), "participant_id"].astype(str))
        test_ids = set(dataset_splits.loc[dataset_splits.split.eq("test"), "participant_id"].astype(str))
        if not train_ids:
            train_ids = set(dataset_data.participant_id.astype(str))
        for model_name, feature_set, hidden_dim in specifications:
            features, a, b, y, valid, indices, participant_ids = prepare_gate_batch(
                dataset_data, feature_set=feature_set
            )
            train_mask = valid & _training_mask(dataset_data, indices, train_ids)
            for seed in seeds:
                model = (
                    NonRecurrentGate(features.shape[-1])
                    if hidden_dim == 0
                    else TinyGateRNN(features.shape[-1], hidden_dim)
                )
                model, train_loss = fit_gate(
                    model, features, a, b, y, train_mask,
                    epochs=epochs, learning_rate=learning_rate, seed=int(seed),
                )
                q, hidden = predict_gate(model, features)
                prediction = q * a + (1 - q) * b
                heldout_mask = valid.copy()
                for batch_i, participant in enumerate(participant_ids):
                    if participant not in test_ids:
                        heldout_mask[batch_i] = False
                metrics = regression_metrics(y[heldout_mask], prediction[heldout_mask])
                metadata = {
                    "seed": int(seed),
                    "hidden_dim": hidden_dim,
                    "train_participants": json.dumps(sorted(train_ids)),
                    "test_participants": json.dumps(sorted(test_ids)),
                    "dataset": dataset,
                    "model_version": model_version,
                    "model": model_name,
                    "feature_set": feature_set,
                }
                fit_row = {**metadata, "train_loss": train_loss, **{f"test_{k}": v for k, v in metrics.items()}}
                if hidden_dim == 1:
                    fit_row.update(distill_one_dimensional_dynamics(features, hidden, valid))
                fit_rows.append(fit_row)

                for batch_i, participant in enumerate(participant_ids):
                    # Export trajectories on held-out participants.  This keeps
                    # the required 20-seed final export tractable and ensures
                    # interpretability plots are not based on fitted subjects.
                    if test_ids and participant not in test_ids:
                        continue
                    previous_hidden = np.zeros(hidden_dim, dtype=float)
                    for time_i in range(indices.shape[1]):
                        source_index = indices[batch_i, time_i]
                        if source_index < 0:
                            continue
                        row = dataset_data.loc[source_index]
                        current_hidden = hidden[batch_i, time_i] if hidden_dim else np.array([])
                        state_frames.append({
                            **metadata,
                            "participant_id": participant,
                            "global_sequence": row.global_sequence,
                            "block_id": row.block_id,
                            "sequence_in_block": row.sequence_in_block,
                            "context": row.context,
                            "probe_type": row.probe_type,
                            "q": float(q[batch_i, time_i]),
                            "gate_entropy": float(gate_entropy([q[batch_i, time_i]])[0]),
                            "predicted_log_rt": float(prediction[batch_i, time_i]),
                            "hidden_state": json.dumps(current_hidden.tolist()),
                            "delta_hidden": json.dumps((current_hidden - previous_hidden).tolist()),
                        })
                        previous_hidden = current_hidden

                if model_name == "nonrecurrent_cue" or (model_name == "gru_full" and hidden_dim == 1):
                    for history_name, state in (("midpoint", 0.0), ("A_history", 2.0), ("B_history", -2.0)):
                        cue = run_cue_interpolation(
                            model,
                            a_state=float(dataset_data.pred_A_rt.median()),
                            b_state=float(dataset_data.pred_B_rt.median()),
                            hidden_state=None if hidden_dim == 0 else [state],
                        )
                        for key, value in metadata.items():
                            cue[key] = value
                        cue["history"] = history_name
                        cue_frames.append(cue)
                    if hidden_dim == 1:
                        intervention = run_state_intervention(
                            model,
                            a_state=float(dataset_data.pred_A_rt.median()),
                            b_state=float(dataset_data.pred_B_rt.median()),
                        )
                        for key, value in metadata.items():
                            intervention[key] = value
                        intervention_frames.append(intervention)
    states = pd.DataFrame(state_frames)
    cues = pd.concat(cue_frames, ignore_index=True) if cue_frames else pd.DataFrame()
    interventions = pd.concat(intervention_frames, ignore_index=True) if intervention_frames else pd.DataFrame()
    return pd.DataFrame(fit_rows), states, cues, interventions
