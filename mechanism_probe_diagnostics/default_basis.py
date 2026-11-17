"""Independent no-context basis audit and held-out Novel comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import additive_cost_rescore, score_prediction_rows


def estimate_independent_default_basis(
    default_tokens: pd.DataFrame, *, dataset: str = "Novel", value_col: str = "mean_rt",
) -> tuple[pd.DataFrame, dict]:
    """Estimate a centered default profile using Default controls only."""
    source = default_tokens[default_tokens.dataset.eq(dataset)].copy()
    conditions = set(source.condition.dropna().astype(str))
    if not conditions or conditions != {"Default"}:
        raise ValueError(f"Default basis source must contain only Default rows, found {conditions}.")
    if "Novel" in set(source.get("original_condition", pd.Series(dtype=str)).astype(str)):
        raise ValueError("Novel rows entered default-basis estimation.")
    participant = source.groupby(
        ["participant_uid", "transition_position"], as_index=False
    )[value_col].mean().rename(columns={value_col: "M"})
    participant["Z"] = participant.M - participant.groupby("participant_uid").M.transform("mean")
    basis = participant.groupby("transition_position", as_index=False).Z.mean()
    basis["Z"] -= basis.Z.mean()
    basis["unit_id"] = "human_default_basis"
    basis["condition"] = "Default"
    ordered_ids = sorted(participant.participant_uid.unique())
    first, second = set(ordered_ids[::2]), set(ordered_ids[1::2])
    split_a = participant[participant.participant_uid.isin(first)].groupby("transition_position").Z.mean()
    split_b = participant[participant.participant_uid.isin(second)].groupby("transition_position").Z.mean()
    reliability = (
        float(np.corrcoef(split_a, split_b)[0, 1])
        if len(first) >= 2 and len(second) >= 2 and split_a.std() > 0 and split_b.std() > 0
        else np.nan
    )
    audit = {
        "source_dataset": dataset,
        "source_condition": "Default",
        "rows_used": int(len(source)),
        "participants_used": int(source.participant_uid.nunique()),
        "ab_probe_rows_used": 0,
        "novel_rows_used": 0,
        "normalization": "participant-center then across-control mean; final exact centering",
        "fold_assignment": "independent disjoint control participants; fixed before Novel fit",
        "split_half_reliability": reliability,
        "is_ab_midpoint": False,
        "firewall_pass": True,
    }
    return basis, audit


def _fit_linear(train: pd.DataFrame, columns: list[str]) -> np.ndarray:
    design = np.column_stack([np.ones(len(train))] + [train[c].to_numpy(float) for c in columns])
    return np.linalg.lstsq(design, train.actual.to_numpy(float), rcond=None)[0]


def _predict_linear(frame: pd.DataFrame, columns: list[str], coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(float) for c in columns])
    return design @ coefficients


def compare_novel_default_models(
    novel_tokens: pd.DataFrame, human_profiles: pd.DataFrame, default_basis: pd.DataFrame,
    rank2_predictions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit structure on odd probe sequences and score even sequences."""
    tokens = novel_tokens[
        novel_tokens.condition.eq("Novel") & novel_tokens.dataset.eq("Novel")
    ].copy()
    endpoints = human_profiles[
        human_profiles.dataset.eq("Novel") & human_profiles.analysis_scale.eq("mean_rt_ms")
        & human_profiles.condition.isin(["A", "B"])
    ].pivot(index=["participant_uid", "transition_position"], columns="condition", values="M")
    endpoints["D"] = (endpoints.B - endpoints.B.groupby(level=0).transform("mean")) - (
        endpoints.A - endpoints.A.groupby(level=0).transform("mean")
    )
    basis_map = default_basis.set_index("transition_position").Z
    work = tokens.merge(endpoints[["D"]].reset_index(), on=["participant_uid", "transition_position"],
                        how="inner", validate="many_to_one")
    work["S"] = work.transition_position.map(basis_map)
    work["actual"] = work.mean_rt
    work["profile_id"] = (
        work.participant_uid.astype(str) + "|" + work.probe_epoch.astype(str)
        + "|r" + work.sequence_repetition.astype(str)
    )
    prediction_rows = []
    specs = {
        "N0_level_only": [], "N1_default_basis": ["S"],
        "N2_ab_contrast": ["D"], "N3_default_plus_ab": ["S", "D"],
    }
    for participant, one in work.groupby("participant_uid", sort=False):
        train = one[one.sequence_repetition.astype(int).mod(2).eq(1)].copy()
        test = one[one.sequence_repetition.astype(int).mod(2).eq(0)].copy()
        if train.empty or test.empty:
            continue
        for model, columns in specs.items():
            coefficients = _fit_linear(train, columns)
            current = test.copy()
            current["predicted"] = _predict_linear(current, columns, coefficients)
            current["model"] = model
            current["evaluation_status"] = "predictive_odd_to_even_repetitions"
            current["basis_probe_rows_used"] = 0
            prediction_rows.append(current)
        # Probe-specific transition means are retained strictly as a ceiling.
        transition_means = train.groupby("transition_position").actual.mean()
        ceiling = test.copy()
        ceiling["predicted"] = ceiling.transition_position.map(transition_means)
        ceiling["model"] = "N5_probe_specific_ceiling"
        ceiling["evaluation_status"] = "descriptive_ceiling_probe_rows_define_shape"
        ceiling["basis_probe_rows_used"] = len(train)
        prediction_rows.append(ceiling)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    raw, centered, scored = score_prediction_rows(
        predictions, actual_col="actual", predicted_col="predicted",
        profile_cols=["model", "participant_uid", "probe_epoch", "sequence_repetition"],
        group_cols=["model"],
    )
    comparison = raw.merge(
        centered[["model", "heldout_rmse", "heldout_mae", "profile_correlation",
                  "centered_shape_variance_explained"]],
        on="model", suffixes=("_raw", "_centered"), validate="one_to_one",
    )
    comparison["unit_id"] = "population"
    comparison["condition"] = "Novel"
    comparison["system"] = "Human"
    comparison["analysis_scale"] = "mean_rt_ms"
    comparison["identifiability_status"] = "identified_independent_default_control"
    comparison["evaluation_status"] = comparison.model.map(
        lambda x: "descriptive_ceiling" if x.startswith("N5") else "predictive_odd_to_even_repetitions"
    )

    if rank2_predictions is not None and len(rank2_predictions):
        rank2 = rank2_predictions[
            rank2_predictions.dataset.eq("Novel") & rank2_predictions.probe_context.eq("Novel")
            & rank2_predictions.model.eq("shared_ab_basis")
        ].copy()
        if len(rank2):
            additive, rank2_evaluation = additive_cost_rescore(rank2)
            rrow = additive[additive.augmentation.eq("structure_plus_delta")].copy()
            rank2_evaluation["actual"] = rank2_evaluation.actual_log_rt
            rank2_evaluation["predicted"] = rank2_evaluation.additive_prediction
            rank2_evaluation["model"] = "N4_existing_rank2_plus_delta"
            rank2_evaluation["condition"] = "Novel"
            rank2_evaluation["system"] = "Human"
            rank2_evaluation["analysis_scale"] = "log_rt"
            rank2_evaluation["profile_id"] = (
                rank2_evaluation.participant_uid.astype(str) + "|" + rank2_evaluation.probe_epoch.astype(str)
                + "|r" + rank2_evaluation.sequence_repetition.astype(str)
            )
            _, _, rank2_scored = score_prediction_rows(
                rank2_evaluation, actual_col="actual", predicted_col="predicted",
                profile_cols=["model", "participant_uid", "probe_epoch", "sequence_repetition"],
                group_cols=["model"],
            )
            rrow["model"] = "N4_existing_rank2_plus_delta"
            rrow["unit_id"] = "population"
            rrow["condition"] = "Novel"
            rrow["system"] = "Human"
            rrow["analysis_scale"] = "log_rt"
            rrow["identifiability_status"] = "existing_fold_safe_rank2_prediction"
            rrow["evaluation_status"] = (
                "predictive_existing_participant_folds_plus_nested_delta_holdout; "
                "not numerically comparable to ms rows"
            )
            comparison = pd.concat([comparison, rrow], ignore_index=True, sort=False)
            scored = pd.concat([scored, rank2_scored], ignore_index=True, sort=False)
    scored["condition"] = "Novel"
    scored["system"] = "Human"
    scored["analysis_scale"] = scored.get("analysis_scale", pd.Series(index=scored.index, dtype=object)).fillna("mean_rt_ms")
    scored["unit_id"] = scored.get("unit_id", scored.profile_id)
    return comparison, scored
