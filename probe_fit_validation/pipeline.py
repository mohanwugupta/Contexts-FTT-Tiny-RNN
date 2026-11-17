"""Assemble, audit, summarize, and plot frozen probe-model predictions.

This module never refits or changes the registered v3 retrieval models.  The
rank-2 v3.1 predictions are reconstructed with the frozen v3.1 implementation
only because the registered output retained errors rather than the signed
observed and predicted values needed for overlays.  Reconstruction is accepted
only when its row-wise errors match the frozen output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from human_low_rank_interpretation.decomposition.estimate_basis import fit_pca_basis
from human_low_rank_interpretation.probes.shared_basis import (
    _global_assignments,
    _predict_participant,
    fit_shared_basis,
)

from .plotting import make_probe_fit_figures


RETRIEVAL_FILES = {
    "M0_scalar_context": "model_M0_scalar_context.csv",
    "M1_scalar_mixture": "model_M1_scalar_mixture.csv",
    "M2_token_specific": "model_M2_token_specific.csv",
    "M3_low_rank": "model_M3_low_rank.csv",
    "M4_srn_informed": "model_M4_srn_informed.csv",
}
MODEL_LABELS = {
    "M0_scalar_context": "Scalar context",
    "M1_scalar_mixture": "Scalar retrieval",
    "M2_token_specific": "Token-specific",
    "M3_low_rank": "Low-rank retrieval (rank 5)",
    "M4_srn_informed": "Frozen SRN-informed",
    "scalar_baseline": "Scalar retrieval",
    "shared_ab_basis": "Shared A/B basis (rank 2)",
    "separate_dataset_ab_basis": "Dataset A/B basis (rank 2)",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r_squared(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator <= 1e-14:
        return float("nan")
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def _correlation(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) < 2 or np.std(actual) <= 1e-14 or np.std(predicted) <= 1e-14:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def load_retrieval_predictions(v3_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load the registered primary v3 predictions without refitting."""
    root = Path(v3_dir).expanduser().resolve()
    frames = []
    source_hashes = {}
    for expected_model, filename in RETRIEVAL_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        source_hashes[filename] = _sha256(path)
        if expected_model == "M4_srn_informed":
            frame = frame[frame.model.eq(expected_model)].copy()
        elif set(frame.model.astype(str)) != {expected_model}:
            raise ValueError(f"{filename} contains an unexpected model label.")
        if frame.empty:
            raise ValueError(f"{filename} contains no primary prediction rows.")
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True, sort=False)
    required = {
        "model", "probe_context", "dataset", "fold", "participant_uid",
        "probe_epoch", "transition_position", "unit_id", "actual_log_rt",
        "predicted_log_rt", "squared_error", "heldout_log_likelihood",
        "heldout_participant_in_group_training", "probe_split",
        "ab_profiles_frozen_before_probe_fit", "probe_rows_used_for_ab_profiles",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Frozen v3 predictions are missing columns: {missing}")
    if predictions.duplicated(["model", "probe_context", "unit_id"]).any():
        raise ValueError("Frozen retrieval predictions contain duplicate model/probe/unit rows.")
    if predictions.heldout_participant_in_group_training.astype(bool).any():
        raise ValueError("A held-out participant appears in group training.")
    if set(predictions.probe_split.astype(str)) != {"test"}:
        raise ValueError("Retrieval diagnostics must use test probe repetitions only.")
    if not predictions.ab_profiles_frozen_before_probe_fit.astype(bool).all():
        raise ValueError("A/B profiles were not frozen before probe fitting.")
    if int(predictions.probe_rows_used_for_ab_profiles.sum()) != 0:
        raise ValueError("Probe rows leaked into the A/B profiles.")

    # Each registered model must score exactly the same observations and actual values.
    for context, group in predictions.groupby("probe_context", sort=False):
        reference = None
        for model, model_rows in group.groupby("model", sort=False):
            current = model_rows.set_index("unit_id").actual_log_rt.sort_index()
            if reference is None:
                reference = current
            elif not current.index.equals(reference.index) or not np.allclose(
                current.to_numpy(), reference.to_numpy(), atol=1e-12, rtol=0
            ):
                raise ValueError(f"Models do not score identical {context} observations ({model}).")
    predictions["model_label"] = predictions.model.map(MODEL_LABELS)
    if predictions.model_label.isna().any():
        raise ValueError("An unlabeled retrieval model reached diagnostics.")
    return predictions, source_hashes


def _selected_rank(v31_dir: Path) -> tuple[int, pd.DataFrame, Path]:
    path = v31_dir / "dimension_selection.csv"
    dimensions = pd.read_csv(path)
    selected = dimensions[
        dimensions.row_type.eq("summary")
        & dimensions.selected_primary_rank.fillna(False).astype(bool)
    ]
    ranks = sorted(selected["rank"].astype(int).unique())
    if len(ranks) != 1:
        raise ValueError(f"Expected one frozen selected rank; found {ranks}.")
    return int(ranks[0]), dimensions, path


def reconstruct_rank2_predictions(
    v3_dir: str | Path,
    v31_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Recreate signed v3.1 overlays and prove equality to frozen row errors."""
    v3 = Path(v3_dir).expanduser().resolve()
    v31 = Path(v31_dir).expanduser().resolve()
    rank, _, dimension_path = _selected_rank(v31)
    if rank != 2:
        raise ValueError(f"The registered v3.1 primary rank is {rank}, not 2.")

    profiles_path = v3 / "pure_ab_token_profiles.csv"
    mixed_path = v3 / "mixed_midpoint_deviation.csv"
    novel_path = v3 / "novel_midpoint_deviation.csv"
    frozen_path = v31 / "shared_vs_separate_basis_predictions.csv"
    for path in (profiles_path, mixed_path, novel_path, frozen_path):
        if not path.exists():
            raise FileNotFoundError(path)
    profiles = pd.read_csv(profiles_path)
    probes = pd.concat([pd.read_csv(mixed_path), pd.read_csv(novel_path)], ignore_index=True)
    assignments, folds = _global_assignments(profiles, 5)
    rows = []
    for fold in range(folds):
        held_identities = {key for key, value in assignments.items() if value == fold}
        held_profile = profiles.participant_uid.astype(str).map(
            lambda value: value.split(":", 1)[-1] in held_identities
        )
        training_profiles = profiles[~held_profile]
        shared = fit_shared_basis(training_profiles, rank)
        separate = {
            dataset: fit_pca_basis(group, rank=rank)
            for dataset, group in training_profiles.groupby("dataset", sort=False)
        }
        held_probe = probes[probes.participant_uid.astype(str).map(
            lambda value: value.split(":", 1)[-1] in held_identities
        )]
        for dataset, dataset_probe in held_probe.groupby("dataset", sort=False):
            for participant, frame in dataset_probe.groupby("participant_uid", sort=False):
                for model, basis in (
                    ("scalar_baseline", None),
                    ("shared_ab_basis", shared),
                    ("separate_dataset_ab_basis", separate[dataset]),
                ):
                    test, predicted_deviation = _predict_participant(frame, basis, model)
                    for source, predicted in zip(test.itertuples(index=False), predicted_deviation):
                        actual_deviation = float(source.midpoint_deviation)
                        predicted = float(predicted)
                        residual = actual_deviation - predicted
                        rows.append({
                            "fold": fold,
                            "dataset": dataset,
                            "probe_context": source.context,
                            "model": model,
                            "model_label": MODEL_LABELS[model],
                            "participant_uid": participant,
                            "unit_id": source.unit_id,
                            "probe_epoch": source.probe_epoch,
                            "transition_position": int(source.transition_position),
                            "sequence_repetition": int(source.sequence_repetition),
                            "actual_midpoint_deviation": actual_deviation,
                            "predicted_midpoint_deviation": predicted,
                            "actual_log_rt": float(source.log_rt),
                            "predicted_log_rt": float(source.midpoint + predicted),
                            "residual": residual,
                            "squared_error": residual ** 2,
                            "absolute_error": abs(residual),
                            "basis_rank": 0 if basis is None else basis.rank,
                            "basis_source": "none" if basis is None else "pure_A_B_only",
                            "heldout_participant_in_basis": False,
                            "probe_rows_used_for_basis": 0,
                        })
    result = pd.DataFrame(rows)
    frozen = pd.read_csv(frozen_path)
    keys = ["fold", "dataset", "probe_context", "model", "participant_uid", "unit_id"]
    joined = result.merge(
        frozen[keys + ["squared_error", "absolute_error", "heldout_log_likelihood"]],
        on=keys,
        suffixes=("_reconstructed", "_frozen"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(joined._merge.astype(str)) != {"both"}:
        raise ValueError("Rank-2 reconstruction does not contain the frozen row identities.")
    squared_difference = np.abs(
        joined.squared_error_reconstructed - joined.squared_error_frozen
    )
    absolute_difference = np.abs(
        joined.absolute_error_reconstructed - joined.absolute_error_frozen
    )
    max_squared_difference = float(squared_difference.max())
    max_absolute_difference = float(absolute_difference.max())
    if max_squared_difference > 1e-10 or max_absolute_difference > 1e-10:
        raise ValueError(
            "Rank-2 reconstruction failed the frozen row-error audit: "
            f"max squared={max_squared_difference:.3g}, max absolute={max_absolute_difference:.3g}."
        )
    audit = {
        "selected_rank": rank,
        "rows": int(len(result)),
        "folds": int(folds),
        "max_squared_error_difference_from_frozen": max_squared_difference,
        "max_absolute_error_difference_from_frozen": max_absolute_difference,
        "dimension_selection_sha256": _sha256(dimension_path),
        "pure_profiles_sha256": _sha256(profiles_path),
        "mixed_probe_sha256": _sha256(mixed_path),
        "novel_probe_sha256": _sha256(novel_path),
        "frozen_prediction_errors_sha256": _sha256(frozen_path),
    }
    return result, audit


def summarize_curves(
    predictions: pd.DataFrame,
    *,
    actual_column: str,
    predicted_column: str,
) -> pd.DataFrame:
    """Participant-balanced overall and epoch-level curve summaries."""
    base = ["model", "model_label", "probe_context", "participant_uid", "transition_position"]
    participant_overall = predictions.groupby(base, as_index=False)[
        [actual_column, predicted_column]
    ].mean()
    participant_overall["probe_epoch"] = "All probe epochs"
    epoch_base = base[:-1] + ["probe_epoch", "transition_position"]
    participant_epoch = predictions.groupby(epoch_base, as_index=False)[
        [actual_column, predicted_column]
    ].mean()
    participant = pd.concat([participant_overall, participant_epoch], ignore_index=True)
    summary = participant.groupby(
        ["model", "model_label", "probe_context", "probe_epoch", "transition_position"],
        as_index=False,
    ).agg(
        actual_mean=(actual_column, "mean"),
        actual_sd=(actual_column, "std"),
        predicted_mean=(predicted_column, "mean"),
        predicted_sd=(predicted_column, "std"),
        participants=("participant_uid", "nunique"),
    )
    summary["actual_se"] = summary.actual_sd / np.sqrt(summary.participants)
    summary["predicted_se"] = summary.predicted_sd / np.sqrt(summary.participants)
    summary["actual_ci_low"] = summary.actual_mean - 1.96 * summary.actual_se
    summary["actual_ci_high"] = summary.actual_mean + 1.96 * summary.actual_se
    summary["predicted_ci_low"] = summary.predicted_mean - 1.96 * summary.predicted_se
    summary["predicted_ci_high"] = summary.predicted_mean + 1.96 * summary.predicted_se
    return summary


def fit_metrics(
    predictions: pd.DataFrame,
    *,
    actual_column: str,
    predicted_column: str,
    scale: str,
) -> pd.DataFrame:
    """Observation- and participant-balanced profile-level fit diagnostics."""
    rows = []
    for (context, model, label), group in predictions.groupby(
        ["probe_context", "model", "model_label"], sort=False
    ):
        actual = group[actual_column].to_numpy(dtype=float)
        predicted = group[predicted_column].to_numpy(dtype=float)
        participant_cells = group.groupby(
            ["participant_uid", "probe_epoch", "transition_position"], as_index=False
        )[[actual_column, predicted_column]].mean()
        cells = participant_cells.groupby(
            ["probe_epoch", "transition_position"], as_index=False
        )[[actual_column, predicted_column]].mean()
        cell_actual = cells[actual_column].to_numpy(dtype=float)
        cell_predicted = cells[predicted_column].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(actual)), actual])
        calibration = np.linalg.pinv(design) @ predicted
        rows.append({
            "probe_context": context,
            "model": model,
            "model_label": label,
            "scale": scale,
            "observations": int(len(group)),
            "participants": int(group.participant_uid.nunique()),
            "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
            "mae": float(np.mean(np.abs(actual - predicted))),
            "r_squared_observation": _r_squared(actual, predicted),
            "pearson_r_observation": _correlation(actual, predicted),
            "calibration_intercept_predicted_on_observed": float(calibration[0]),
            "calibration_slope_predicted_on_observed": float(calibration[1]),
            "profile_cells": int(len(cells)),
            "rmse_profile_cells": float(np.sqrt(np.mean((cell_actual - cell_predicted) ** 2))),
            "r_squared_profile_cells": _r_squared(cell_actual, cell_predicted),
            "pearson_r_profile_cells": _correlation(cell_actual, cell_predicted),
            "evaluation": "held-out even probe repetitions; participant-balanced profile cells",
        })
    return pd.DataFrame(rows)


def epoch_fit_metrics(
    predictions: pd.DataFrame,
    *,
    actual_column: str,
    predicted_column: str,
) -> pd.DataFrame:
    """Expose temporal mean bias that an across-epoch overlay can conceal."""
    frame = predictions.copy()
    frame["signed_error_actual_minus_predicted"] = (
        frame[actual_column] - frame[predicted_column]
    )
    frame["squared_diagnostic_error"] = frame.signed_error_actual_minus_predicted ** 2
    frame["absolute_diagnostic_error"] = frame.signed_error_actual_minus_predicted.abs()
    return frame.groupby(
        ["probe_context", "model", "model_label", "probe_epoch"], as_index=False
    ).agg(
        observations=("unit_id", "size"),
        participants=("participant_uid", "nunique"),
        mean_signed_error_actual_minus_predicted=(
            "signed_error_actual_minus_predicted", "mean"
        ),
        rmse=("squared_diagnostic_error", lambda values: float(np.sqrt(values.mean()))),
        mae=("absolute_diagnostic_error", "mean"),
    )


def load_state_space_reference(state_space_dir: str | Path | None) -> pd.DataFrame:
    """Load the already-created exploratory RT overlay metrics for context only."""
    columns = [
        "model_condition", "model", "rmse", "r_squared", "bic", "bic_weight",
        "evaluation", "outcome_scale",
    ]
    if state_space_dir is None:
        return pd.DataFrame(columns=columns)
    path = Path(state_space_dir).expanduser().resolve() / "state_space_fit_metrics.csv"
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path)
    selected = frame[
        frame.outcome.eq("normalized_rt") & frame.bic_rank.eq(1)
    ][["model_condition", "model", "rmse", "r_squared", "bic", "bic_weight"]].copy()
    selected["evaluation"] = "in-sample fit to 170 seed-averaged SRN trials"
    selected["outcome_scale"] = "normalized RT (1 - execution speed)"
    return selected[columns]


def _write_readme(
    output: Path,
    retrieval_metrics: pd.DataFrame,
    retrieval_epoch_metrics: pd.DataFrame,
    rank2_metrics: pd.DataFrame,
    state_space: pd.DataFrame,
):
    retrieval_best = retrieval_metrics.loc[
        retrieval_metrics.groupby("probe_context").rmse.idxmin(),
        ["probe_context", "model_label", "rmse", "r_squared_observation", "r_squared_profile_cells"],
    ]
    rank2_best = rank2_metrics.loc[
        rank2_metrics.groupby("probe_context").rmse.idxmin(),
        ["probe_context", "model_label", "rmse", "r_squared_observation", "r_squared_profile_cells"],
    ]
    state_text = "Not supplied."
    if len(state_space):
        state_text = state_space.to_string(index=False)
    low_rank_epochs = retrieval_epoch_metrics[
        retrieval_epoch_metrics.model.eq("M3_low_rank")
    ][[
        "probe_context", "probe_epoch", "mean_signed_error_actual_minus_predicted", "rmse"
    ]]
    readme = f"""# Probe fit diagnostics

This directory is a read-only visual audit of the frozen v3 retrieval fits and
the frozen v3.1 rank-2 linear-algebra analysis. No registered model was refit or
overwritten.

## Figures

1. `figure_01_retrieval_model_overlays`: held-out observed and predicted log RT
   across transition position for every primary retrieval model.
2. `figure_02_low_rank_retrieval_by_epoch`: the winning v3 low-rank retrieval
   model separated by probe epoch.
3. `figure_03_rank2_linear_algebra_overlays`: scalar, shared-rank-2, and
   dataset-specific-rank-2 predictions of probe midpoint deviation.
4. `figure_04_low_rank_observed_vs_predicted`: observation-level calibration for
   the v3 low-rank retrieval model and the selected v3.1 rank-2 model.
5. `figure_05_fit_metric_context`: observation/profile R-squared panels plus the
   earlier exploratory state-space RT reference.

## Best held-out retrieval RMSE

```text
{retrieval_best.to_string(index=False)}
```

## Best held-out rank-2 RMSE

```text
{rank2_best.to_string(index=False)}
```

## Low-rank epoch-level bias check

Positive signed error means the model underpredicted observed log RT; negative
means it overpredicted. This table is why the epoch-separated figure should be
retained alongside the favorable aggregate fit.

```text
{low_rank_epochs.to_string(index=False)}
```

## State-space reference

```text
{state_text}
```

## Interpretation boundary

The retrieval and rank-2 fits are out-of-sample on even probe repetitions. The
state-space values are in-sample fits to condition-level SRN means over training
trials, on a different outcome and aggregation level. Side-by-side figures are
useful for checking curve shape, but their R-squared values are not an
apples-to-apples statistical equivalence test. The corrected SRN run also failed
its prespecified execution-direction gate, so its state-space overlay remains
exploratory.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def run_probe_fit_diagnostics(
    v3_dir: str | Path,
    v31_dir: str | Path,
    output_dir: str | Path,
    *,
    state_space_dir: str | Path | None = None,
) -> dict:
    output = Path(output_dir).expanduser().resolve()
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    retrieval, retrieval_hashes = load_retrieval_predictions(v3_dir)
    retrieval_curves = summarize_curves(
        retrieval, actual_column="actual_log_rt", predicted_column="predicted_log_rt"
    )
    retrieval_metrics = fit_metrics(
        retrieval,
        actual_column="actual_log_rt",
        predicted_column="predicted_log_rt",
        scale="log RT",
    )
    retrieval_epoch_metrics = epoch_fit_metrics(
        retrieval, actual_column="actual_log_rt", predicted_column="predicted_log_rt"
    )

    rank2, rank2_audit = reconstruct_rank2_predictions(v3_dir, v31_dir)
    rank2_curves = summarize_curves(
        rank2,
        actual_column="actual_midpoint_deviation",
        predicted_column="predicted_midpoint_deviation",
    )
    rank2_metrics_deviation = fit_metrics(
        rank2,
        actual_column="actual_midpoint_deviation",
        predicted_column="predicted_midpoint_deviation",
        scale="probe midpoint deviation (log RT)",
    )
    rank2_metrics_log_rt = fit_metrics(
        rank2,
        actual_column="actual_log_rt",
        predicted_column="predicted_log_rt",
        scale="log RT including frozen participant A/B midpoint",
    )
    rank2_epoch_metrics = epoch_fit_metrics(
        rank2, actual_column="actual_midpoint_deviation",
        predicted_column="predicted_midpoint_deviation",
    )
    rank2_metrics = rank2_metrics_deviation.merge(
        rank2_metrics_log_rt[
            ["probe_context", "model", "r_squared_observation", "r_squared_profile_cells"]
        ],
        on=["probe_context", "model"],
        suffixes=("_target", "_log_rt"),
        validate="one_to_one",
    )
    # Preserve the unsuffixed target fields expected by plotting and tabular review.
    rank2_metrics["r_squared_observation"] = rank2_metrics.r_squared_observation_target
    rank2_metrics["r_squared_profile_cells"] = rank2_metrics.r_squared_profile_cells_target
    state_space = load_state_space_reference(state_space_dir)

    retrieval.to_csv(output / "retrieval_predictions_used.csv", index=False)
    retrieval_curves.to_csv(output / "retrieval_curve_summary.csv", index=False)
    retrieval_metrics.to_csv(output / "retrieval_fit_metrics.csv", index=False)
    retrieval_epoch_metrics.to_csv(output / "retrieval_epoch_fit_metrics.csv", index=False)
    rank2.to_csv(output / "rank2_predictions_reconstructed.csv", index=False)
    rank2_curves.to_csv(output / "rank2_curve_summary.csv", index=False)
    rank2_metrics.to_csv(output / "rank2_fit_metrics.csv", index=False)
    rank2_epoch_metrics.to_csv(output / "rank2_epoch_fit_metrics.csv", index=False)
    state_space.to_csv(output / "state_space_reference_metrics.csv", index=False)

    audit = {
        "analysis_status": "read-only visual diagnostic",
        "registered_models_refit": False,
        "retrieval_rows": int(len(retrieval)),
        "retrieval_observations_per_model": {
            str(context): int(group.unit_id.nunique())
            for context, group in retrieval.groupby("probe_context", sort=False)
        },
        "retrieval_source_sha256": retrieval_hashes,
        "rank2_reconstruction": rank2_audit,
        "state_space_comparison_is_apples_to_apples": False,
        "state_space_reason": (
            "State-space uses in-sample aggregate SRN training-trial means; probe models use "
            "held-out human probe observations on log-RT or midpoint-deviation scales."
        ),
        "corrected_srn_execution_gate": "failed; state-space reference is exploratory",
    }
    (output / "validation_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    make_probe_fit_figures(
        figures,
        retrieval=retrieval,
        retrieval_curves=retrieval_curves,
        retrieval_metrics=retrieval_metrics,
        rank2=rank2,
        rank2_curves=rank2_curves,
        rank2_metrics=rank2_metrics,
        state_space=state_space,
    )
    _write_readme(
        output, retrieval_metrics, retrieval_epoch_metrics, rank2_metrics, state_space
    )
    return {
        "retrieval_predictions": retrieval,
        "retrieval_curves": retrieval_curves,
        "retrieval_metrics": retrieval_metrics,
        "retrieval_epoch_metrics": retrieval_epoch_metrics,
        "rank2_predictions": rank2,
        "rank2_curves": rank2_curves,
        "rank2_metrics": rank2_metrics,
        "rank2_epoch_metrics": rank2_epoch_metrics,
        "state_space_reference": state_space,
        "audit": audit,
    }
