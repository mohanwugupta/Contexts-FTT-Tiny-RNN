from pathlib import Path

import numpy as np
import pandas as pd

from probe_fit_validation.pipeline import (
    epoch_fit_metrics,
    fit_metrics,
    load_retrieval_predictions,
    reconstruct_rank2_predictions,
    summarize_curves,
)


ROOT = Path(__file__).resolve().parents[1]


def _toy_predictions():
    rows = []
    for participant, offset in [("p1", 0.0), ("p2", 0.2)]:
        for epoch in ["Mixed1", "Mixed2"]:
            for position in [1, 2]:
                actual = offset + float(position)
                rows.append({
                    "model": "M1_scalar_mixture",
                    "model_label": "Scalar retrieval",
                    "probe_context": "Both",
                    "participant_uid": participant,
                    "probe_epoch": epoch,
                    "transition_position": position,
                    "actual": actual,
                    "predicted": actual,
                })
    return pd.DataFrame(rows)


def test_fit_metrics_reports_perfect_prediction():
    result = fit_metrics(
        _toy_predictions(), actual_column="actual", predicted_column="predicted", scale="toy"
    ).iloc[0]
    assert result.rmse == 0
    assert result.r_squared_observation == 1
    assert result.r_squared_profile_cells == 1
    np.testing.assert_allclose(
        result.calibration_intercept_predicted_on_observed, 0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.calibration_slope_predicted_on_observed, 1, atol=1e-12
    )


def test_curve_summary_is_participant_balanced_and_contains_overall():
    result = summarize_curves(
        _toy_predictions(), actual_column="actual", predicted_column="predicted"
    )
    assert set(result.probe_epoch) == {"All probe epochs", "Mixed1", "Mixed2"}
    overall = result[result.probe_epoch.eq("All probe epochs")]
    assert set(overall.participants) == {2}
    assert np.allclose(overall.actual_mean, overall.predicted_mean)


def test_epoch_metrics_preserve_signed_bias_direction():
    data = _toy_predictions()
    data["predicted"] = data.actual - 0.25
    result = epoch_fit_metrics(
        data.assign(unit_id=np.arange(len(data))),
        actual_column="actual",
        predicted_column="predicted",
    )
    np.testing.assert_allclose(result.mean_signed_error_actual_minus_predicted, 0.25)


def test_frozen_retrieval_models_use_identical_heldout_rows():
    predictions, hashes = load_retrieval_predictions(
        ROOT / "outputs/human_configured_state_v3"
    )
    assert len(predictions) == 60_060
    assert set(predictions.model) == {
        "M0_scalar_context", "M1_scalar_mixture", "M2_token_specific",
        "M3_low_rank", "M4_srn_informed",
    }
    assert predictions.groupby(["probe_context", "model"]).size().to_dict()[
        ("Both", "M3_low_rank")
    ] == 8_232
    assert len(hashes) == 5


def test_rank2_signed_reconstruction_matches_frozen_errors():
    predictions, audit = reconstruct_rank2_predictions(
        ROOT / "outputs/human_configured_state_v3",
        ROOT / "outputs/human_low_rank_interpretation_v31",
    )
    assert len(predictions) == 36_036
    assert audit["selected_rank"] == 2
    assert audit["max_squared_error_difference_from_frozen"] < 1e-10
    assert audit["max_absolute_error_difference_from_frozen"] < 1e-10
