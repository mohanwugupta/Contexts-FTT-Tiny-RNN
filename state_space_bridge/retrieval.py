"""Bridge adapters around the already-validated v3/v3.1 retrieval analyses."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from human_configured_state.config import HumanConfiguredStateConfig
from human_configured_state.data import load_human_token_data
from human_configured_state.models import fit_probe_model_ladder
from human_configured_state.pure_profiles import compute_midpoint_deviation, estimate_ab_profiles

from .vector_bridge import assert_rank_regression, validate_transition_positions


MODEL_MAP = {
    "M1_scalar_mixture": "RET_SCALAR",
    "M2_token_specific": "RET_TOKEN",
    "M3_low_rank": "RET_LOWRANK",
    "M4_srn_informed": "RET_SRN",
}


def _categorical_predictions(midpoint: pd.DataFrame) -> pd.DataFrame:
    """Winner-take-all A/B endpoint chosen from each participant's odd repetitions."""
    rows = []
    for participant, group in midpoint.groupby("participant_uid", sort=False):
        calibration = group[group.probe_split.eq("train")]
        test = group[group.probe_split.eq("test")]
        if calibration.empty or test.empty:
            continue
        sse_a = float(np.sum((calibration.log_rt - calibration.r_a) ** 2))
        sse_b = float(np.sum((calibration.log_rt - calibration.r_b) ** 2))
        selected = "A" if sse_a <= sse_b else "B"
        residual_source = calibration.log_rt - (
            calibration.r_a if selected == "A" else calibration.r_b
        )
        sigma = max(float(residual_source.std(ddof=1)), 0.05)
        for source in test.itertuples(index=False):
            prediction = float(source.r_a if selected == "A" else source.r_b)
            error = float(source.log_rt - prediction)
            rows.append({
                "model": "RET_CATEGORICAL", "source_model": "new frozen-endpoint categorical",
                "probe_context": source.context, "dataset": source.dataset,
                "participant_uid": participant, "counterbalance": source.counterbalance,
                "probe_epoch": source.probe_epoch,
                "transition_position": int(source.transition_position),
                "sequence_repetition": int(source.sequence_repetition),
                "unit_id": source.unit_id, "actual_log_rt": float(source.log_rt),
                "predicted_log_rt": prediction, "residual": error,
                "squared_error": error ** 2, "absolute_error": abs(error),
                "heldout_log_likelihood": float(
                    -0.5 * (math.log(2 * math.pi * sigma ** 2) + error ** 2 / sigma ** 2)
                ),
                "selected_endpoint": selected, "parameter_count": 0,
                "ab_profiles_frozen_before_probe_fit": True,
                "probe_rows_used_for_ab_profiles": 0,
                "target_transition_preserved": True,
            })
    return pd.DataFrame(rows)


def _default_probe_predictions(combined_path, novel_path, v3, *, participant_limit=None):
    srn = pd.read_csv(v3 / "srn_novel_profile.csv")
    if "row_type" in srn:
        srn = srn[srn.row_type.eq("all_seed_aggregate")].copy()
    predictions, midpoints = [], []
    for path, dataset in [(combined_path, "Combined"), (novel_path, "Novel")]:
        tokens = load_human_token_data(path, dataset, target_probe="Default")
        people = sorted(tokens.participant_uid.unique())
        if participant_limit is not None:
            tokens = tokens[tokens.participant_uid.isin(people[:int(participant_limit)])].copy()
        tokens["is_probe"] = tokens.context.eq("Default")
        tokens["probe_epoch"] = None
        for _, indices in tokens[tokens.is_probe].groupby("participant_uid", sort=False).groups.items():
            blocks = tokens.loc[indices].sort_values("block_order").block_id.drop_duplicates().tolist()
            mapping = {block: f"Default{number + 1}" for number, block in enumerate(blocks)}
            tokens.loc[indices, "probe_epoch"] = tokens.loc[indices, "block_id"].map(mapping)
        profiles = estimate_ab_profiles(tokens)
        midpoint = compute_midpoint_deviation(
            tokens[tokens.context.eq("Default")], profiles, endpoint_epsilon=.03
        )
        midpoints.append(midpoint)
        config = HumanConfiguredStateConfig(
            cv_folds=3 if participant_limit else 5,
            recovery_replicates=1,
            participant_limit_per_dataset=None,
            make_plots=False,
            show_progress=False,
            profile="smoke" if participant_limit else "final",
        )
        predictions.append(fit_probe_model_ladder(midpoint, profiles, srn, config=config))
    return pd.concat(predictions, ignore_index=True), pd.concat(midpoints, ignore_index=True)


def assemble_retrieval_outputs(
    v3_dir, v31_dir, *, combined_path=None, novel_path=None, participant_limit=None,
    require_rank_two=True,
):
    v3 = Path(v3_dir).expanduser().resolve()
    v31 = Path(v31_dir).expanduser().resolve()
    required_v3 = [
        "mixed_midpoint_deviation.csv", "novel_midpoint_deviation.csv",
        "mixed_model_comparison.csv", "novel_model_comparison.csv",
        "model_M1_scalar_mixture.csv", "model_M2_token_specific.csv",
        "model_M3_low_rank.csv", "model_M4_srn_informed.csv",
        "model_recovery_matrix.csv", "pure_ab_token_profiles.csv",
    ]
    required_v31 = [
        "dimension_selection.csv", "component_loadings.csv",
        "shared_vs_separate_basis_predictions.csv",
    ]
    missing = [str(v3 / name) for name in required_v3 if not (v3 / name).exists()]
    missing += [str(v31 / name) for name in required_v31 if not (v31 / name).exists()]
    if missing:
        raise FileNotFoundError("Bridge inputs missing: " + ", ".join(missing))

    predictions = []
    for filename, target in [
        ("model_M1_scalar_mixture.csv", "RET_SCALAR"),
        ("model_M2_token_specific.csv", "RET_TOKEN"),
        ("model_M3_low_rank.csv", "RET_LOWRANK"),
        ("model_M4_srn_informed.csv", "RET_SRN"),
    ]:
        frame = pd.read_csv(v3 / filename)
        # M4 contains secondary variants; retain the registered execution profile only.
        if target == "RET_SRN":
            frame = frame[frame.model.eq("M4_srn_informed")].copy()
        frame["source_model"] = frame.model
        frame["model"] = target
        predictions.append(frame)
    midpoint = pd.concat([
        pd.read_csv(v3 / "mixed_midpoint_deviation.csv"),
        pd.read_csv(v3 / "novel_midpoint_deviation.csv"),
    ], ignore_index=True)
    if combined_path is not None and novel_path is not None:
        default_predictions, default_midpoint = _default_probe_predictions(
            combined_path, novel_path, v3, participant_limit=participant_limit
        )
        for source_model, target in MODEL_MAP.items():
            selected = default_predictions[default_predictions.model.eq(source_model)].copy()
            selected["source_model"] = selected.model
            selected["model"] = target
            predictions.append(selected)
        midpoint = pd.concat([midpoint, default_midpoint], ignore_index=True, sort=False)
    predictions.append(_categorical_predictions(midpoint))
    predictions = pd.concat(predictions, ignore_index=True, sort=False)
    validate_transition_positions(predictions)

    comparison_rows = []
    for (probe, model), group in predictions.groupby(["probe_context", "model"], sort=False):
        comparison_rows.append({
            "probe_context": probe, "model": model,
            "heldout_log_likelihood": float(group.heldout_log_likelihood.sum()),
            "mean_heldout_log_likelihood": float(group.heldout_log_likelihood.mean()),
            "heldout_rmse": float(np.sqrt(group.squared_error.mean())),
            "heldout_mae": float(group.absolute_error.mean()),
            "heldout_observations": len(group),
            "heldout_participants": int(group.participant_uid.nunique()),
            "primary_metric": "heldout_log_likelihood",
        })
    comparison = pd.DataFrame(comparison_rows)
    comparison["primary_rank"] = comparison.groupby(
        "probe_context"
    ).heldout_log_likelihood.rank(ascending=False, method="min").astype(int)
    comparison = comparison.sort_values(["probe_context", "primary_rank"])

    recovery = pd.read_csv(v3 / "model_recovery_matrix.csv")
    for column in ["generating_model", "fitted_model", "selected_model", "model"]:
        if column in recovery:
            recovery[column] = recovery[column].replace({
                "M0": "RET_CATEGORICAL", "M0_scalar_context": "RET_CATEGORICAL",
                "M1": "RET_SCALAR", "M1_scalar_mixture": "RET_SCALAR",
                "M2": "RET_TOKEN", "M2_token_specific": "RET_TOKEN",
                "M3": "RET_LOWRANK", "M3_low_rank": "RET_LOWRANK",
                "M4": "RET_SRN", "M4_srn_informed": "RET_SRN",
            })
    recovery["source_note"] = (
        "Validated v3 recovery; RET_CATEGORICAL label corresponds to its endpoint-level M0 regime."
    )
    vector = pd.read_csv(v3 / "pure_ab_token_profiles.csv")
    validate_transition_positions(vector)
    dimensions = pd.read_csv(v31 / "dimension_selection.csv")
    if require_rank_two:
        assert_rank_regression(dimensions)
    basis = pd.read_csv(v31 / "component_loadings.csv")
    validate_transition_positions(basis)
    lowrank_predictions = pd.read_csv(v31 / "shared_vs_separate_basis_predictions.csv")
    return {
        "retrieval_model_comparison": comparison,
        "retrieval_predictions": predictions,
        "retrieval_recovery": recovery,
        "human_vector_AB_profiles": vector,
        "human_low_rank_rank_selection": dimensions,
        "human_low_rank_basis": basis,
        "human_low_rank_predictions": lowrank_predictions,
    }
