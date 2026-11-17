"""Leakage-free split A/B skill states and fatigue/restart nuisance process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SplitSkillState:
    """Two context-specific exponentially updated skill states.

    States and targets are expressed on the log-RT scale.  ``cross_talk=0`` is
    the frozen split architecture used by the probe models.
    """

    a: float
    b: float
    learning_rate: float = 0.15
    retention: float = 1.0
    cross_talk: float = 0.0
    fatigue_slope: float = 0.0
    restart_cost: float = 0.0

    def nuisance(self, sequence_position: float = 0.0, block_start: bool = False) -> float:
        return self.fatigue_slope * float(sequence_position) + self.restart_cost * float(block_start)

    def predict(self, context: str, sequence_position: float = 0.0, block_start: bool = False) -> float:
        if context not in {"A", "B"}:
            raise ValueError(f"SplitSkillState predicts A or B, not {context!r}")
        value = self.a if context == "A" else self.b
        return float(value + self.nuisance(sequence_position, block_start))

    def update(self, context: str, target: float) -> None:
        if context not in {"A", "B"}:
            return
        if context == "A":
            error = float(target) - self.a
            self.a += self.learning_rate * error
            self.b += self.learning_rate * self.cross_talk * error
        else:
            error = float(target) - self.b
            self.b += self.learning_rate * error
            self.a += self.learning_rate * self.cross_talk * error

    def retain_toward(self, anchor_a: float, anchor_b: float) -> None:
        self.a = anchor_a + self.retention * (self.a - anchor_a)
        self.b = anchor_b + self.retention * (self.b - anchor_b)


def _position_and_starts(participant: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    position = np.zeros(len(participant), dtype=float)
    starts = np.zeros(len(participant), dtype=bool)
    for _, indices in participant.groupby("block_id", sort=False).groups.items():
        loc = participant.index.get_indexer(indices)
        seq = pd.to_numeric(participant.loc[indices, "sequence_in_block"], errors="coerce")
        if seq.notna().all():
            position[loc] = seq.to_numpy(float) - float(seq.min())
        else:
            position[loc] = np.arange(len(indices), dtype=float)
        starts[loc[0]] = True
    return position, starts


def _initial_values(participant: pd.DataFrame) -> tuple[float, float]:
    established = participant[participant.context.isin(["A", "B"])]
    grand = float(established.log_rt.median())
    values = {}
    for context in ("A", "B"):
        series = established.loc[established.context.eq(context), "log_rt"]
        values[context] = float(series.iloc[: min(5, len(series))].median()) if len(series) else grand
    return values["A"], values["B"]


def _simulate_participant(
    participant: pd.DataFrame,
    theta: np.ndarray,
    *,
    cross_talk: float,
) -> pd.DataFrame:
    initial_a, initial_b, learning_rate, retention, fatigue_slope, restart_cost = map(float, theta)
    state = SplitSkillState(
        a=initial_a,
        b=initial_b,
        learning_rate=learning_rate,
        retention=retention,
        cross_talk=cross_talk,
        fatigue_slope=fatigue_slope,
        restart_cost=restart_cost,
    )
    position, starts = _position_and_starts(participant)
    rows = []
    previous_block: Optional[object] = None
    for local_i, (_, row) in enumerate(participant.iterrows()):
        if previous_block is not None and row.block_id != previous_block:
            state.retain_toward(initial_a, initial_b)
        pred_a = state.predict("A", position[local_i], starts[local_i])
        pred_b = state.predict("B", position[local_i], starts[local_i])
        rows.append(
            {
                "state_A": state.a,
                "state_B": state.b,
                "fatigue_component": state.nuisance(position[local_i], starts[local_i]),
                "pred_A_log_rt": pred_a,
                "pred_B_log_rt": pred_b,
                "pred_A_rt": float(np.exp(np.clip(pred_a, -20, 20))),
                "pred_B_rt": float(np.exp(np.clip(pred_b, -20, 20))),
            }
        )
        # This is the no-leakage boundary: only established A/B observations
        # may change the foundational states.
        if row.context in {"A", "B"} and np.isfinite(row.log_rt):
            latent_target = float(row.log_rt) - state.nuisance(position[local_i], starts[local_i])
            state.update(row.context, latent_target)
        previous_block = row.block_id
    return pd.DataFrame(rows, index=participant.index)


def _fit_one(participant: pd.DataFrame, cross_talk: float) -> tuple[np.ndarray, object]:
    a0, b0 = _initial_values(participant)
    x0 = np.array([a0, b0, 0.18, 0.995, 0.0, 0.08], dtype=float)
    y = participant.log_rt.to_numpy(float)
    established_mask = participant.context.isin(["A", "B"]).to_numpy() & np.isfinite(y)

    def objective(theta: np.ndarray) -> float:
        simulated = _simulate_participant(participant, theta, cross_talk=cross_talk)
        context = participant.context.to_numpy()
        pred = np.where(
            context == "A",
            simulated.pred_A_log_rt.to_numpy(),
            simulated.pred_B_log_rt.to_numpy(),
        )
        residual = y[established_mask] - pred[established_mask]
        # A tiny regularizer stabilizes nuisance estimates in short schedules.
        penalty = 1e-5 * ((theta[4] / 0.05) ** 2 + (theta[5] / 0.2) ** 2)
        return float(np.mean(residual**2) + penalty)

    bounds = [
        (a0 - 1.0, a0 + 1.0),
        (b0 - 1.0, b0 + 1.0),
        (0.005, 0.8),
        (0.7, 1.0),
        (-0.1, 0.1),
        (-0.5, 1.0),
    ]
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    return np.asarray(result.x, dtype=float), result


def fit_counterfactual_states(
    data: pd.DataFrame,
    *,
    cross_talk: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit split states per participant and return A/B counterfactuals.

    The optimizer and state updates see only rows whose canonical context is A
    or B.  Probe rows are prediction-only and therefore cannot determine the
    split or alter its trajectory.
    """
    required = {"participant_id", "block_id", "sequence_in_block", "context", "log_rt"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Counterfactual fitting requires columns: {sorted(missing)}")
    predictions = []
    fits = []
    names = ["initial_A", "initial_B", "learning_rate", "retention", "fatigue_slope", "restart_cost"]
    for participant_id, participant in data.groupby("participant_id", sort=False):
        participant = participant.sort_values("global_sequence", kind="stable").copy()
        if not participant.context.isin(["A", "B"]).any():
            raise ValueError(f"Participant {participant_id!r} has no established A/B observations.")
        theta, result = _fit_one(participant, cross_talk)
        simulated = _simulate_participant(participant, theta, cross_talk=cross_talk)
        predictions.append(simulated)
        fit_row = {"participant_id": participant_id, **dict(zip(names, theta))}
        fit_row.update(
            {
                "cross_talk": float(cross_talk),
                "fit_loss": float(result.fun),
                "fit_success": bool(result.success),
                "n_ab_observations": int(participant.context.isin(["A", "B"]).sum()),
            }
        )
        fits.append(fit_row)
    predicted = pd.concat(predictions).sort_index()
    result = data.copy()
    for column in predicted.columns:
        result[column] = predicted[column]
    return result, pd.DataFrame(fits)

