"""Schedule-matched state-space model recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .state_space import MODEL_SPECS, fit_model_family, simulate_lump, simulate_two_context


GENERATING_PARAMETERS = {
    "SS_LUMP": np.array([0.90]),
    "SS_CROSS": np.array([0.90, 0.12, 0.86]),
    "SS_SPLIT": np.array([0.90, 0.86]),
}


def _generate(model, contexts, rng, noise_sd):
    template = np.ones(len(contexts), dtype=float)
    parameters = GENERATING_PARAMETERS[model]
    if model == "SS_LUMP":
        state = simulate_lump(parameters, template)["x_readout"]
    else:
        state = simulate_two_context(
            parameters, template, contexts,
            cross_talk_fixed_zero=model == "SS_SPLIT",
        )["x_readout"]
    observed = 0.25 + 0.75 * state + rng.normal(0, noise_sd, len(state))
    observed[0] = 1.0 + rng.normal(0, noise_sd / 4)
    return observed


def run_state_space_recovery(
    contexts, *, replicates=100, noise_sd=0.03, random_seed=34000
) -> pd.DataFrame:
    cx = np.asarray(contexts, dtype=str)
    if not set(cx).issubset({"A", "B"}):
        raise ValueError("Recovery schedule must contain only A/B.")
    rng = np.random.default_rng(random_seed)
    rows = []
    schedule = pd.DataFrame({
        "context": cx, "global_step": np.arange(1, len(cx) + 1),
    })
    for generated in MODEL_SPECS:
        for replicate in range(int(replicates)):
            observed = _generate(generated, cx, rng, noise_sd)
            schedule["observed"] = observed
            fits = fit_model_family(schedule)
            winner = min(fits, key=lambda name: fits[name].bic)
            for fitted_name, fit in fits.items():
                cross = (
                    float(fit.parameters[1]) if fitted_name == "SS_CROSS" else
                    (0.0 if fitted_name == "SS_SPLIT" else np.nan)
                )
                rows.append({
                    "generating_model": generated, "replicate": replicate,
                    "fitted_model": fitted_name, "selected_model": winner,
                    "selected_correctly": winner == generated,
                    "BIC": fit.bic, "delta_BIC": fit.bic - min(x.bic for x in fits.values()),
                    "RMSE": fit.rmse, "true_A_cross": (
                        GENERATING_PARAMETERS[generated][1] if generated == "SS_CROSS" else 0.0
                    ),
                    "estimated_A_cross": cross,
                    "recovery_replicates_per_generating_model": int(replicates),
                })
    result = pd.DataFrame(rows)
    accuracy = result.drop_duplicates(["generating_model", "replicate"])
    accuracy = accuracy.groupby("generating_model").selected_correctly.mean().to_dict()
    result["generating_model_selection_accuracy"] = result.generating_model.map(accuracy)
    result["recovery_threshold"] = 0.80
    result["recovery_threshold_passed"] = result.generating_model_selection_accuracy >= 0.80
    return result

