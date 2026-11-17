"""Python translation of the no-fatigue state-space fits in state-space_SRN.Rmd.

These models are descriptive diagnostics over an observed cohort-average trace.
They do not train, select, or modify an SRN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class TwoContextStates:
    readout: np.ndarray
    a_state: np.ndarray
    b_state: np.ndarray


@dataclass(frozen=True)
class StateSpaceFit:
    name: str
    dynamic_parameters: Mapping[str, float]
    decoder_intercept: float
    decoder_slope: float
    observed: np.ndarray
    predicted: np.ndarray
    rss: float
    rmse: float
    r_squared: float
    bic: float
    parameter_count: int
    optimizer_success: bool
    optimizer_message: str


def _finite_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values.")
    return array


def simulate_lump(a: float, y) -> np.ndarray:
    """Mirror R's Lump update: x[n+1] = A*x[n], initialized at y[0]."""
    observed = _finite_vector(y, "y")
    state = np.empty(len(observed) + 1, dtype=float)
    state[0] = observed[0]
    for index in range(len(observed)):
        state[index + 1] = float(a) * state[index]
    return state[1:]


def _context_weights(label: str) -> tuple[float, float]:
    normalized = str(label).strip().lower()
    if normalized in {"a", "maze"}:
        return 1.0, 0.0
    if normalized in {"b", "melody"}:
        return 0.0, 1.0
    if normalized in {"mixed", "nocontext", "no context", "novel"}:
        return 0.5, 0.5
    raise ValueError(f"Unknown context label {label!r}.")


def simulate_two_context(
    *,
    a_a: float,
    a_b: float,
    a_cross: float,
    y,
    contexts,
) -> TwoContextStates:
    """Mirror the gated Cross/Split state update from state-space_SRN.Rmd."""
    observed = _finite_vector(y, "y")
    context = np.asarray(contexts, dtype=object)
    if context.ndim != 1 or len(context) != len(observed):
        raise ValueError("contexts must be one-dimensional and trial-aligned with y.")

    state_a = np.empty(len(observed) + 1, dtype=float)
    state_b = np.empty(len(observed) + 1, dtype=float)
    state_a[0] = observed[0]
    state_b[0] = observed[0]
    readout = np.empty(len(observed), dtype=float)

    for index, label in enumerate(context):
        weight_a, weight_b = _context_weights(label)
        readout[index] = weight_a * state_a[index] + weight_b * state_b[index]

        normalized = str(label).strip().lower()
        if normalized in {"a", "maze"}:
            state_a[index + 1] = a_a * state_a[index] + a_cross * state_b[index]
            state_b[index + 1] = state_b[index]
        elif normalized in {"b", "melody"}:
            state_b[index + 1] = a_b * state_b[index] + a_cross * state_a[index]
            state_a[index + 1] = state_a[index]
        else:
            state_a[index + 1] = a_a * state_a[index] + a_cross * state_b[index]
            state_b[index + 1] = a_b * state_b[index] + a_cross * state_a[index]

    return TwoContextStates(
        readout=readout,
        a_state=state_a[1:],
        b_state=state_b[1:],
    )


def _decode(observed: np.ndarray, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(latent)), latent])
    coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
    return coefficients, design @ coefficients


def _summarize_fit(
    *,
    name: str,
    dynamic_parameters: Mapping[str, float],
    observed: np.ndarray,
    predicted: np.ndarray,
    decoder: np.ndarray,
    parameter_count: int,
    success: bool,
    message: str,
) -> StateSpaceFit:
    residual = observed - predicted
    rss = float(np.dot(residual, residual))
    rss_for_log = max(rss, np.finfo(float).eps)
    rmse = float(np.sqrt(np.mean(residual**2)))
    tss = float(np.sum((observed - observed.mean()) ** 2))
    r_squared = float(1.0 - rss / tss) if tss > 0 else float("nan")
    bic = float(len(observed) * np.log(rss_for_log / len(observed))
                + parameter_count * np.log(len(observed)))
    return StateSpaceFit(
        name=name,
        dynamic_parameters={key: float(value) for key, value in dynamic_parameters.items()},
        decoder_intercept=float(decoder[0]),
        decoder_slope=float(decoder[1]),
        observed=observed.copy(),
        predicted=np.asarray(predicted, dtype=float),
        rss=rss,
        rmse=rmse,
        r_squared=r_squared,
        bic=bic,
        parameter_count=parameter_count,
        optimizer_success=bool(success),
        optimizer_message=str(message),
    )


def _fit_lump(observed: np.ndarray) -> StateSpaceFit:
    def objective(parameters):
        _, predicted = _decode(observed, simulate_lump(parameters[0], observed))
        return float(np.sum((predicted - observed) ** 2))

    result = minimize(
        objective,
        x0=np.array([0.95]),
        method="L-BFGS-B",
        bounds=[(0.60, 0.999)],
        options={"maxiter": 100_000},
    )
    latent = simulate_lump(result.x[0], observed)
    decoder, predicted = _decode(observed, latent)
    return _summarize_fit(
        name="Lump",
        dynamic_parameters={"A": result.x[0]},
        observed=observed,
        predicted=predicted,
        decoder=decoder,
        parameter_count=3,
        success=result.success,
        message=result.message,
    )


def _fit_two_context(observed: np.ndarray, contexts: np.ndarray, *, split: bool) -> StateSpaceFit:
    if split:
        initial = np.array([0.95, 0.95])
        bounds = [(0.60, 0.999), (0.60, 0.999)]

        def unpack(parameters):
            return parameters[0], parameters[1], 0.0
    else:
        initial = np.array([0.95, 0.02, 0.95])
        bounds = [(0.60, 0.999), (0.00, 0.50), (0.60, 0.999)]

        def unpack(parameters):
            return parameters[0], parameters[2], parameters[1]

    def objective(parameters):
        a_a, a_b, a_cross = unpack(parameters)
        latent = simulate_two_context(
            a_a=a_a,
            a_b=a_b,
            a_cross=a_cross,
            y=observed,
            contexts=contexts,
        ).readout
        _, predicted = _decode(observed, latent)
        return float(np.sum((predicted - observed) ** 2))

    result = minimize(
        objective,
        x0=initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 100_000},
    )
    a_a, a_b, a_cross = unpack(result.x)
    latent = simulate_two_context(
        a_a=a_a,
        a_b=a_b,
        a_cross=a_cross,
        y=observed,
        contexts=contexts,
    ).readout
    decoder, predicted = _decode(observed, latent)
    dynamic_parameters = {"A_A": a_a, "A_B": a_b}
    if not split:
        dynamic_parameters["A_cross"] = a_cross
    return _summarize_fit(
        name="Split" if split else "Cross",
        dynamic_parameters=dynamic_parameters,
        observed=observed,
        predicted=predicted,
        decoder=decoder,
        parameter_count=4 if split else 5,
        success=result.success,
        message=result.message,
    )


def fit_three_models(observed, contexts) -> dict[str, StateSpaceFit]:
    """Fit the R notebook's Lump, Cross, and Split models to one mean trace."""
    y = _finite_vector(observed, "observed")
    context = np.asarray(contexts, dtype=object)
    if context.ndim != 1 or len(context) != len(y):
        raise ValueError("contexts must be one-dimensional and trial-aligned.")
    return {
        "Lump": _fit_lump(y),
        "Cross": _fit_two_context(y, context, split=False),
        "Split": _fit_two_context(y, context, split=True),
    }

