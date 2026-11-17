"""Exact Python port of the no-fatigue models in ``state-space_SRN.Rmd``.

Only pure A/B observations are accepted by the simulator.  In particular,
this module deliberately has no ambiguous-context readout or update rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class ModelSpec:
    name: str
    parameter_names: tuple[str, ...]
    initial: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    bic_parameters: int


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "SS_LUMP": ModelSpec("SS_LUMP", ("A",), (0.95,), ((0.60, 0.999),), 3),
    "SS_CROSS": ModelSpec(
        "SS_CROSS", ("A_A", "A_cross", "A_B"), (0.95, 0.02, 0.95),
        ((0.60, 0.999), (0.00, 0.50), (0.60, 0.999)), 5,
    ),
    "SS_SPLIT": ModelSpec(
        "SS_SPLIT", ("A_A", "A_B"), (0.95, 0.95),
        ((0.60, 0.999), (0.60, 0.999)), 4,
    ),
}


@dataclass
class StateSpaceFit:
    model: str
    parameter_names: tuple[str, ...]
    parameters: np.ndarray
    decoder_names: tuple[str, ...]
    decoder: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    x_readout: np.ndarray
    x_a: np.ndarray
    x_b: np.ndarray
    contexts: np.ndarray
    rss: float
    rmse: float
    r2: float
    log_likelihood: float
    bic: float
    converged: bool
    optimizer_message: str


def _values(y) -> np.ndarray:
    result = np.asarray(y, dtype=float)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("State-space observations must be a finite one-dimensional trace.")
    return result


def _contexts(contexts, n: int) -> np.ndarray:
    result = np.asarray(contexts, dtype=str)
    if result.shape != (n,) or not set(result).issubset({"A", "B"}):
        raise ValueError("The canonical state-space simulator accepts pure A/B contexts only.")
    return result


def simulate_lump(parameters, observed) -> dict[str, np.ndarray]:
    """Reproduce ``simulate_state_Lump_noF`` including its post-update readout."""
    y = _values(observed)
    a = float(np.asarray(parameters, dtype=float)[0])
    state = np.zeros(len(y) + 1, dtype=float)
    state[0] = y[0]
    for index in range(len(y)):
        state[index + 1] = a * state[index]
    readout = state[1:]
    return {"x_readout": readout, "x_a": readout.copy(), "x_b": readout.copy()}


def simulate_two_context(
    parameters, observed, contexts, *, cross_talk_fixed_zero: bool = False
) -> dict[str, np.ndarray]:
    """Port ``simulate_state_twoctx_gated_noF`` for its pure-context branch."""
    y = _values(observed)
    cx = _contexts(contexts, len(y))
    p = np.asarray(parameters, dtype=float)
    if cross_talk_fixed_zero:
        if len(p) != 2:
            raise ValueError("Split expects A_A and A_B.")
        a_a, a_b = map(float, p)
        a_cross = 0.0
    else:
        if len(p) != 3:
            raise ValueError("Cross expects A_A, A_cross, and A_B.")
        a_a, a_cross, a_b = map(float, p)

    x_a = np.zeros(len(y) + 1, dtype=float)
    x_b = np.zeros(len(y) + 1, dtype=float)
    x_a[0] = y[0]
    x_b[0] = y[0]
    readout = np.zeros(len(y), dtype=float)
    for index, context in enumerate(cx):
        readout[index] = x_a[index] if context == "A" else x_b[index]
        if context == "A":
            x_a[index + 1] = a_a * x_a[index] + a_cross * x_b[index]
            x_b[index + 1] = x_b[index]
        else:
            x_b[index + 1] = a_b * x_b[index] + a_cross * x_a[index]
            x_a[index + 1] = x_a[index]
    return {"x_readout": readout, "x_a": x_a[1:], "x_b": x_b[1:]}


def fit_decoder_ols(observed, state) -> tuple[np.ndarray, np.ndarray]:
    """R's QR OLS decoder ``y = Intercept + c_x * state``."""
    y = _values(observed)
    x = np.asarray(state, dtype=float)
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta = np.nan_to_num(beta, nan=0.0)
    return beta, design @ beta


def _simulate(model: str, parameters, y, contexts):
    if model == "SS_LUMP":
        return simulate_lump(parameters, y)
    return simulate_two_context(
        parameters, y, contexts, cross_talk_fixed_zero=model == "SS_SPLIT"
    )


def fit_state_space_model(observed, contexts, model: str) -> StateSpaceFit:
    if model not in MODEL_SPECS:
        raise ValueError(f"Unknown state-space model {model!r}.")
    y = _values(observed)
    cx = _contexts(contexts, len(y))
    spec = MODEL_SPECS[model]

    def objective(parameters):
        simulation = _simulate(model, parameters, y, cx)
        _, prediction = fit_decoder_ols(y, simulation["x_readout"])
        value = float(np.sum((prediction - y) ** 2))
        return value if np.isfinite(value) else 1e12

    fitted = minimize(
        objective, np.asarray(spec.initial), method="L-BFGS-B", bounds=spec.bounds,
        options={"maxiter": 100000, "ftol": 1e-15, "gtol": 1e-10},
    )
    parameters = np.asarray(fitted.x, dtype=float)
    simulation = _simulate(model, parameters, y, cx)
    decoder, prediction = fit_decoder_ols(y, simulation["x_readout"])
    residual = y - prediction
    rss = max(float(residual @ residual), np.finfo(float).eps)
    n = len(y)
    sigma2 = rss / n
    log_likelihood = float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))
    bic = float(n * np.log(sigma2) + spec.bic_parameters * np.log(n))
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - rss / denominator) if denominator > 0 else np.nan
    return StateSpaceFit(
        model=model, parameter_names=spec.parameter_names, parameters=parameters,
        decoder_names=("Intercept", "c_x"), decoder=decoder, observed=y,
        predicted=prediction, x_readout=simulation["x_readout"],
        x_a=simulation["x_a"], x_b=simulation["x_b"], contexts=cx,
        rss=rss, rmse=float(np.sqrt(rss / n)), r2=r2,
        log_likelihood=log_likelihood, bic=bic, converged=bool(fitted.success),
        optimizer_message=str(fitted.message),
    )


def fit_model_family(frame: pd.DataFrame, *, observed_column="observed") -> dict[str, StateSpaceFit]:
    """Fit all R models after enforcing the pure-A/B scientific firewall."""
    missing = {"context", observed_column} - set(frame)
    if missing:
        raise ValueError(f"State-space frame is missing {sorted(missing)}.")
    pure = frame[frame.context.isin(["A", "B"])].copy()
    if len(pure) < 2:
        raise ValueError("At least two pure A/B observations are required.")
    y = pure[observed_column].to_numpy(dtype=float)
    contexts = pure.context.to_numpy(dtype=str)
    return {
        name: fit_state_space_model(y, contexts, name)
        for name in MODEL_SPECS
    }

