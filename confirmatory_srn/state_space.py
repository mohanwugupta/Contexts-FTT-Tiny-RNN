"""Canonical R-equivalent Lump/Cross/Split fits at mean and seed levels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from state_space_bridge.state_space import MODEL_SPECS, fit_model_family


LABELS = {"SS_LUMP": "Lump", "SS_CROSS": "Cross", "SS_SPLIT": "Split"}


def _fit_trace(frame: pd.DataFrame, *, condition: str, fit_scope: str,
               seed) -> tuple[list[dict], list[dict]]:
    ordered = frame.sort_values("training_sequence").reset_index(drop=True)
    fit_input = ordered.rename(columns={"mean_normalized_RT": "observed"})
    fits = fit_model_family(fit_input, observed_column="observed")
    minimum = min(value.bic for value in fits.values())
    winner = min(fits, key=lambda key: fits[key].bic)
    raw_weights = {
        model: np.exp(-.5 * (fit.bic - minimum)) for model, fit in fits.items()
    }
    weight_total = sum(raw_weights.values())
    fit_rows, prediction_rows = [], []
    for model_name, fit in fits.items():
        spec = MODEL_SPECS[model_name]
        parameters = dict(zip(fit.parameter_names, map(float, fit.parameters)))
        decoder = dict(zip(fit.decoder_names, map(float, fit.decoder)))
        row = {
            "condition": condition,
            "fit_scope": fit_scope,
            "seed": seed,
            "model": model_name,
            "model_label": LABELS[model_name],
            "n_observations": len(fit.observed),
            "log_likelihood": fit.log_likelihood,
            "AIC": 2 * spec.bic_parameters - 2 * fit.log_likelihood,
            "BIC": fit.bic,
            "delta_BIC": fit.bic - minimum,
            "BIC_weight": raw_weights[model_name] / weight_total,
            "RSS": fit.rss,
            "RMSE": fit.rmse,
            "R2": fit.r2,
            "best_model": model_name == winner,
            "converged": fit.converged,
            "optimizer_message": fit.optimizer_message,
            "model_parameters": json.dumps(parameters, sort_keys=True),
            "decoder_parameters": json.dumps(decoder, sort_keys=True),
            "A": parameters.get("A", np.nan),
            "A_A": parameters.get("A_A", np.nan),
            "A_cross": parameters.get("A_cross", 0.0 if model_name == "SS_SPLIT" else np.nan),
            "A_B": parameters.get("A_B", np.nan),
            "decoder_intercept": decoder.get("Intercept", np.nan),
            "decoder_c_x": decoder.get("c_x", np.nan),
            "canonical_source": "state-space_SRN.Rmd no-fatigue ladder",
            "human_rows_used": 0,
        }
        fit_rows.append(row)
        for index, source in ordered.iterrows():
            prediction_rows.append({
                "condition": condition,
                "fit_scope": fit_scope,
                "seed": seed,
                "model": model_name,
                "model_label": LABELS[model_name],
                "training_sequence": int(source.training_sequence),
                "block_index": int(source.block_index),
                "context": source.context,
                "within_block_sequence": int(source.within_block_sequence),
                "observed": float(fit.observed[index]),
                "predicted": float(fit.predicted[index]),
                "residual": float(fit.observed[index] - fit.predicted[index]),
                "best_model": model_name == winner,
                "BIC": fit.bic,
                "RMSE": fit.rmse,
                "R2": fit.r2,
            })
    return fit_rows, prediction_rows


def fit_confirmatory_state_space(sequence: pd.DataFrame):
    execution = sequence[sequence.execution_active.eq(True)].copy()
    fit_rows, prediction_rows = [], []
    for (condition, seed), frame in execution.groupby(["condition", "seed"], sort=True):
        rows, predictions = _fit_trace(
            frame, condition=condition, fit_scope="seed", seed=int(seed)
        )
        fit_rows.extend(rows)
        prediction_rows.extend(predictions)
    for condition, frame in execution.groupby("condition", sort=True):
        mean_trace = frame.groupby([
            "training_sequence", "block_index", "context", "within_block_sequence"
        ], as_index=False, sort=True).mean_normalized_RT.mean()
        rows, predictions = _fit_trace(
            mean_trace, condition=condition, fit_scope="seed_average", seed=np.nan
        )
        fit_rows.extend(rows)
        prediction_rows.extend(predictions)
    fits = pd.DataFrame(fit_rows)
    predictions = pd.DataFrame(prediction_rows)
    seed_winners = fits[fits.fit_scope.eq("seed") & fits.best_model].copy()
    classifications = seed_winners[[
        "condition", "seed", "model", "model_label", "BIC", "AIC", "RMSE", "R2"
    ]].copy()
    classifications["row_type"] = "seed"
    counts = classifications.groupby(
        ["condition", "model", "model_label"], as_index=False
    ).seed.count().rename(columns={"seed": "classified_seed_count"})
    # Emit all three prespecified model categories, including zero-count winners,
    # so the classification table and figure always report complete fractions.
    combinations = pd.MultiIndex.from_product(
        [sorted(classifications.condition.unique()), list(LABELS)],
        names=["condition", "model"],
    ).to_frame(index=False)
    combinations["model_label"] = combinations.model.map(LABELS)
    fractions = combinations.merge(
        counts, on=["condition", "model", "model_label"], how="left"
    )
    fractions["classified_seed_count"] = (
        fractions.classified_seed_count.fillna(0).astype(int)
    )
    totals = classifications.groupby("condition").seed.nunique()
    fractions["classification_fraction"] = fractions.apply(
        lambda row: row.classified_seed_count / totals.loc[row.condition], axis=1
    )
    fraction_rows = fractions.assign(
        seed=np.nan, BIC=np.nan, AIC=np.nan, RMSE=np.nan, R2=np.nan,
        row_type="classification_fraction",
    )
    classifications["classified_seed_count"] = np.nan
    classifications["classification_fraction"] = np.nan
    classifications = pd.concat([classifications, fraction_rows], ignore_index=True, sort=False)
    return fits, classifications, predictions
