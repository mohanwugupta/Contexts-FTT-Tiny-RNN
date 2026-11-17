"""Epoch-aware tiny-gate training, recurrence resets, and distillation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from .gate_models import fit_gate, prepare_gate_batch
from .metrics import regression_metrics
from .tiny_gate_rnn import NonRecurrentGate, TinyGateRNN


def forward_gate_with_resets(
    model: TinyGateRNN,
    features: np.ndarray,
    reset_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a GRU while zeroing hidden state before selected observations."""
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 3:
        raise ValueError("features must have shape participant x time x feature")
    resets = np.zeros(values.shape[:2], dtype=bool) if reset_mask is None else np.asarray(reset_mask, bool)
    if resets.shape != tuple(values.shape[:2]):
        raise ValueError("reset_mask must match participant x time")
    hidden = torch.zeros(1, values.shape[0], model.hidden_dim, dtype=values.dtype)
    q_steps, h_steps = [], []
    model.eval()
    with torch.no_grad():
        for time in range(values.shape[1]):
            reset = torch.as_tensor(resets[:, time], dtype=torch.bool)
            if bool(reset.any()):
                hidden[:, reset, :] = 0.0
            output, hidden_sequence = model(values[:, time : time + 1], hidden)
            hidden = hidden_sequence[:, -1, :].unsqueeze(0)
            q_steps.append(output[:, 0])
            h_steps.append(hidden_sequence[:, 0, :])
    return (
        torch.stack(q_steps, dim=1).cpu().numpy(),
        torch.stack(h_steps, dim=1).cpu().numpy(),
    )


def _source_masks(data: pd.DataFrame, indices: np.ndarray) -> dict[str, np.ndarray]:
    masks = {name: np.zeros(indices.shape, dtype=bool) for name in ("block_reset", "before_mixed1", "between_mixed1_mixed2")}
    for participant_i in range(indices.shape[0]):
        previous_block = None
        for time_i in range(indices.shape[1]):
            index = int(indices[participant_i, time_i])
            if index < 0:
                continue
            row = data.loc[index]
            block_start = previous_block is None or row.block_id != previous_block
            masks["block_reset"][participant_i, time_i] = block_start
            masks["before_mixed1"][participant_i, time_i] = block_start and str(row.epoch) == "Mixed_1"
            masks["between_mixed1_mixed2"][participant_i, time_i] = block_start and str(row.epoch) == "Mixed_2"
            previous_block = row.block_id
    return masks


def _assign_states(
    data: pd.DataFrame,
    indices: np.ndarray,
    q: np.ndarray,
    hidden: np.ndarray,
    *,
    model: str,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for position in zip(*np.where(indices >= 0)):
        source_index = int(indices[position])
        row = data.loc[source_index]
        value = float(q[position])
        rows.append(
            {
                "source_index": source_index,
                "participant_id": row.participant_id,
                "dataset": row.dataset,
                "block_id": row.block_id,
                "global_sequence": row.global_sequence,
                "analysis_sequence_in_block": row.analysis_sequence_in_block,
                "epoch": str(row.epoch),
                "counterbalance_history": row.counterbalance_history,
                "model": model,
                "seed": int(seed),
                "q": value,
                "hidden_state": json.dumps(hidden[position].tolist()),
                "predicted_log_rt": value * row.pred_A_log_rt + (1.0 - value) * row.pred_B_log_rt,
                "observed_log_rt": row.log_rt,
            }
        )
    return pd.DataFrame(rows)


def fit_epoch_gate_models(
    data: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    seeds: Iterable[int] = (42,),
    epochs: int = 300,
    learning_rate: float = 0.02,
    progress: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fit N0/N1 and evaluate N2--N4 as state interventions.

    Parameters are fit only on training participants' Mixed 1/2 observations.
    Reset variants reuse the exact N1 weights, isolating recurrence state.
    """
    features, a, b, y, valid, indices, _ = prepare_gate_batch(data, feature_set="full")
    train_ids = set(splits.loc[splits.split.eq("train"), "participant_id"].astype(str))
    train_allowed = data.participant_id.astype(str).isin(train_ids) & data.epoch.astype("object").isin(["Mixed_1", "Mixed_2"])
    fit_mask = np.zeros(indices.shape, dtype=bool)
    test_mask = np.zeros(indices.shape, dtype=bool)
    valid_timeline = indices >= 0
    for position in zip(*np.where(valid_timeline)):
        index = int(indices[position])
        fit_mask[position] = bool(train_allowed.loc[index])
        test_mask[position] = bool(not data.loc[index, "participant_id"] in train_ids)
    if not test_mask.any():
        test_mask = valid_timeline.copy()

    fit_rows, state_frames, reset_rows, trained = [], [], [], []
    reset_masks = _source_masks(data, indices)
    for seed in seeds:
        specifications = (
            ("N0_nonrecurrent", NonRecurrentGate(features.shape[-1]), 0),
            ("N1_full_h1", TinyGateRNN(features.shape[-1], 1), 1),
            ("N1_full_h2", TinyGateRNN(features.shape[-1], 2), 2),
            ("N1_full_h4", TinyGateRNN(features.shape[-1], 4), 4),
        )
        for name, model, hidden_dim in specifications:
            if progress is not None:
                progress(f"Neural gate fit: seed {seed}, {name}, {epochs} epoch ceiling")
            model, train_loss = fit_gate(
                model, features, a, b, y, fit_mask,
                epochs=epochs, learning_rate=learning_rate, seed=int(seed),
            )
            if isinstance(model, TinyGateRNN):
                q, hidden = forward_gate_with_resets(model, features)
            else:
                model.eval()
                with torch.no_grad():
                    q = model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy()
                hidden = np.zeros((*q.shape, 0), dtype=float)
            predicted = q * a + (1.0 - q) * b
            metrics = regression_metrics(y[test_mask], predicted[test_mask])
            fit_rows.append(
                {
                    "model": name, "seed": int(seed), "hidden_dim": hidden_dim,
                    "train_loss": train_loss, **{f"test_{key}": value for key, value in metrics.items()},
                }
            )
            if progress is not None:
                progress(f"Neural gate complete: seed {seed}, {name}, train loss={train_loss:.6g}")
            state_frames.append(_assign_states(data, indices, q, hidden, model=name, seed=int(seed)))
            trained.append({"model": name, "seed": int(seed), "network": model, "features": features, "indices": indices})
            if name != "N1_full_h1":
                continue
            variants = {
                "N1_full": np.zeros(indices.shape, dtype=bool),
                "N2_block_reset": reset_masks["block_reset"],
                "N3_reset_before_Mixed_1": reset_masks["before_mixed1"],
                "N4_reset_between_Mixed_1_Mixed_2": reset_masks["between_mixed1_mixed2"],
            }
            baseline_q = None
            for variant, reset_mask in variants.items():
                intervention_q, _ = forward_gate_with_resets(model, features, reset_mask)
                if baseline_q is None:
                    baseline_q = intervention_q
                intervention_prediction = intervention_q * a + (1.0 - intervention_q) * b
                for epoch_name in ("Mixed_1", "Mixed_2", "Mixed_Test"):
                    epoch_mask = np.zeros(indices.shape, dtype=bool)
                    for position in zip(*np.where(valid_timeline)):
                        index = int(indices[position])
                        epoch_mask[position] = str(data.loc[index, "epoch"]) == epoch_name
                    evaluate = epoch_mask & test_mask
                    reset_rows.append(
                        {
                            "model": variant,
                            "seed": int(seed),
                            "epoch": epoch_name,
                            "mean_q": float(np.mean(intervention_q[evaluate])) if evaluate.any() else np.nan,
                            "mean_delta_q_from_full": float(np.mean(intervention_q[evaluate] - baseline_q[evaluate])) if evaluate.any() else np.nan,
                            **regression_metrics(y[evaluate], intervention_prediction[evaluate]),
                        }
                    )
    return (
        pd.DataFrame(fit_rows),
        pd.concat(state_frames, ignore_index=True) if state_frames else pd.DataFrame(),
        pd.DataFrame(reset_rows),
        trained,
    )


def distill_epoch_dynamics(
    data: pd.DataFrame,
    gate_states: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Progressively distill 1D GRU state and report held-out R-squared."""
    one_dim = gate_states[gate_states.model.eq("N1_full_h1")].copy()
    if one_dim.empty:
        return pd.DataFrame()
    one_dim["h"] = one_dim.hidden_state.map(lambda value: float(json.loads(value)[0]))
    source = data.reset_index().rename(columns={"index": "source_index"})
    merged = one_dim.merge(
        source[["source_index", "context", "pred_A_log_rt", "pred_B_log_rt"]],
        on="source_index", how="left",
    ).sort_values(["seed", "participant_id", "global_sequence"], kind="stable")
    merged["h_previous"] = merged.groupby(["seed", "participant_id"]).h.shift().fillna(0.0)
    merged["cue_A"] = merged.context.eq("A").astype(float) + 0.5 * merged.context.eq("Probe")
    merged["cue_B"] = merged.context.eq("B").astype(float) + 0.5 * merged.context.eq("Probe")
    merged["is_mixed"] = merged.context.eq("Probe").astype(float)
    merged["counterbalance"] = merged.counterbalance_history.eq("A_before_Mixed").astype(float)
    merged["mixed_experience_count"] = merged.groupby(["seed", "participant_id"]).is_mixed.cumsum()
    merged["time_since_mixed"] = merged.groupby(["seed", "participant_id"]).is_mixed.transform(
        lambda values: np.arange(len(values)) - pd.Series(np.where(values.to_numpy() > 0, np.arange(len(values)), np.nan)).ffill().fillna(0).to_numpy()
    )
    merged["ab_state_difference"] = merged.pred_A_log_rt - merged.pred_B_log_rt
    split_map = splits.drop_duplicates("participant_id").set_index("participant_id").split
    merged["split"] = merged.participant_id.map(split_map).fillna("test")
    base = ["h_previous", "cue_A", "cue_B", "is_mixed", "counterbalance"]
    specifications = (
        ("base", base),
        ("plus_mixed_experience_count", base + ["mixed_experience_count"]),
        ("plus_time_since_mixed", base + ["mixed_experience_count", "time_since_mixed"]),
        ("plus_AB_state_difference", base + ["mixed_experience_count", "time_since_mixed", "ab_state_difference"]),
    )
    rows = []
    for seed, seed_data in merged.groupby("seed"):
        train = seed_data[seed_data.split.eq("train")]
        test = seed_data[seed_data.split.eq("test")]
        if test.empty:
            test = seed_data
        for name, columns in specifications:
            regression = LinearRegression().fit(train[columns], train.h)
            rows.append(
                {
                    "seed": int(seed), "distilled_model": name,
                    "features": json.dumps(columns),
                    "intercept": float(regression.intercept_),
                    "coefficients": json.dumps(regression.coef_.tolist()),
                    "heldout_r2": float(regression.score(test[columns], test.h)),
                    "train_r2": float(regression.score(train[columns], train.h)),
                }
            )
    return pd.DataFrame(rows)
