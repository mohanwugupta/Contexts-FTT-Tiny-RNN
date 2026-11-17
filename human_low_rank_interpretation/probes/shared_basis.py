"""Frozen-basis projections and participant-held-out shared-space tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..decomposition.estimate_basis import LowRankBasis, fit_pca_basis


def project_probe_coefficients(probe: pd.DataFrame, basis: LowRankBasis) -> pd.DataFrame:
    before = basis.components.copy()
    rows = []
    for participant, frame in probe.groupby("participant_uid", sort=False):
        profile = frame.groupby("transition_position").midpoint_deviation.mean().reindex(
            range(1, 8)
        )
        keep = profile.notna().to_numpy()
        design = basis.components[keep]
        coefficient = np.linalg.pinv(design) @ profile.to_numpy(dtype=float)[keep]
        for component, value in enumerate(coefficient, start=1):
            rows.append({
                "participant_uid": participant,
                "dataset": str(frame.dataset.iloc[0]) if "dataset" in frame else "Unknown",
                "probe_context": str(frame.context.iloc[0]) if "context" in frame else "Probe",
                "component": component,
                "coefficient": float(value),
                "observed_transitions": int(keep.sum()),
                "basis_rank": basis.rank,
                "basis_updated_from_probe": False,
                "probe_rows_used_for_basis": 0,
            })
    if not np.array_equal(before, basis.components):
        raise AssertionError("Probe projection changed the frozen A/B basis.")
    return pd.DataFrame(rows)


def summarize_probe_coefficients(probe, basis, *, replicates, random_seed):
    participant = project_probe_coefficients(probe, basis)
    rng = np.random.default_rng(random_seed)
    participants = participant.participant_uid.unique()
    bootstrap = {component: [] for component in range(1, basis.rank + 1)}
    for _ in range(int(replicates)):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        sample = pd.concat([
            participant[participant.participant_uid.eq(value)] for value in sampled
        ], ignore_index=True)
        for component, group in sample.groupby("component"):
            bootstrap[int(component)].append(float(group.coefficient.mean()))
    summary = participant.groupby(
        ["dataset", "probe_context", "component"], as_index=False
    ).agg(
        mean_coefficient=("coefficient", "mean"),
        sem_coefficient=("coefficient", "sem"),
        coefficient_sd=("coefficient", "std"),
        participants=("participant_uid", "nunique"),
    )
    summary["bootstrap_ci_lower"] = summary.component.map(
        lambda value: float(np.quantile(bootstrap[int(value)], .025))
    )
    summary["bootstrap_ci_upper"] = summary.component.map(
        lambda value: float(np.quantile(bootstrap[int(value)], .975))
    )
    summary["basis_updated_from_probe"] = False
    summary["probe_rows_used_for_basis"] = 0
    return participant, summary


def temporal_probe_coefficients(probe, basis, *, replicates, random_seed):
    rows = []
    rng = np.random.default_rng(random_seed)
    for epoch, frame in probe.dropna(subset=["probe_epoch"]).groupby("probe_epoch", sort=False):
        profile = frame.groupby("transition_position").midpoint_deviation.mean().reindex(range(1, 8))
        coefficient = np.linalg.pinv(basis.components) @ profile.to_numpy(dtype=float)
        participants = frame.participant_uid.unique()
        boot = {component: [] for component in range(basis.rank)}
        for _ in range(int(replicates)):
            sampled = rng.choice(participants, size=len(participants), replace=True)
            sample = pd.concat([frame[frame.participant_uid.eq(value)] for value in sampled])
            boot_profile = sample.groupby("transition_position").midpoint_deviation.mean().reindex(
                range(1, 8)
            )
            value = np.linalg.pinv(basis.components) @ boot_profile.to_numpy(dtype=float)
            for component in range(basis.rank):
                boot[component].append(float(value[component]))
        for component, value in enumerate(coefficient, start=1):
            rows.append({
                "dataset": str(frame.dataset.iloc[0]),
                "probe_context": str(frame.context.iloc[0]),
                "probe_epoch": epoch, "component": component,
                "coefficient": float(value),
                "bootstrap_ci_lower": float(np.quantile(boot[component - 1], .025)),
                "bootstrap_ci_upper": float(np.quantile(boot[component - 1], .975)),
                "participants": int(len(participants)),
                "basis_rank": basis.rank, "basis_updated_from_probe": False,
                "probe_rows_used_for_basis": 0,
            })
    return pd.DataFrame(rows)


def _dataset_center(profiles):
    result = profiles.copy()
    result["delta"] = result.delta - result.groupby(
        ["dataset", "transition_position"]
    ).delta.transform("mean")
    return result


def fit_shared_basis(profiles, rank):
    return fit_pca_basis(_dataset_center(profiles), rank=rank)


def _global_assignments(profiles, n_folds):
    identities = sorted({value.split(":", 1)[-1] for value in profiles.participant_uid.astype(str)})
    folds = min(int(n_folds), len(identities))
    return {identity: index % folds for index, identity in enumerate(identities)}, folds


def _sigma(training_probe):
    residual = training_probe.midpoint_deviation - training_probe.groupby(
        ["participant_uid", "transition_position"]
    ).midpoint_deviation.transform("mean")
    return max(float(residual.std(ddof=1)), .05)


def _predict_participant(frame, basis, model):
    calibration = frame[frame.probe_split.eq("train")]
    test = frame[frame.probe_split.eq("test")]
    if calibration.empty or test.empty:
        return test, np.array([])
    if model == "scalar_baseline":
        x = calibration.delta.to_numpy(dtype=float)
        y = calibration.midpoint_deviation.to_numpy(dtype=float)
        coefficient = float(x @ y / (x @ x)) if x @ x > 1e-14 else 0.0
        predicted = coefficient * test.delta.to_numpy(dtype=float)
    else:
        profile = calibration.groupby("transition_position").midpoint_deviation.mean().reindex(
            range(1, 8)
        ).to_numpy(dtype=float)
        coefficient = np.linalg.pinv(basis.components) @ profile
        predicted = basis.components[
            test.transition_position.astype(int).to_numpy() - 1
        ] @ coefficient
    return test, predicted


def compare_shared_separate_basis(profiles, probes, *, rank, n_folds):
    assignments, folds = _global_assignments(profiles, n_folds)
    all_probe = pd.concat(list(probes.values()), ignore_index=True)
    rows = []
    for fold in range(folds):
        held_identities = {key for key, value in assignments.items() if value == fold}
        is_held_profile = profiles.participant_uid.astype(str).map(
            lambda value: value.split(":", 1)[-1] in held_identities
        )
        training_profiles = profiles[~is_held_profile]
        shared = fit_shared_basis(training_profiles, rank)
        separate = {
            dataset: fit_pca_basis(group, rank=rank)
            for dataset, group in training_profiles.groupby("dataset", sort=False)
        }
        is_held_probe = all_probe.participant_uid.astype(str).map(
            lambda value: value.split(":", 1)[-1] in held_identities
        )
        training_probe = all_probe[~is_held_probe]
        heldout_probe = all_probe[is_held_probe]
        sigma = _sigma(training_probe)
        training_ids = set(training_profiles.participant_uid.astype(str))
        for dataset, dataset_probe in heldout_probe.groupby("dataset", sort=False):
            for participant, frame in dataset_probe.groupby("participant_uid", sort=False):
                for model, basis in (
                    ("scalar_baseline", None),
                    ("shared_ab_basis", shared),
                    ("separate_dataset_ab_basis", separate[dataset]),
                ):
                    test, predicted = _predict_participant(frame, basis, model)
                    for source, value in zip(test.itertuples(index=False), predicted):
                        error = float(source.midpoint_deviation - value)
                        rows.append({
                            "fold": fold, "dataset": dataset,
                            "probe_context": source.context, "model": model,
                            "participant_uid": participant, "unit_id": source.unit_id,
                            "heldout_log_likelihood": float(-.5 * (
                                math.log(2 * math.pi * sigma ** 2) + error ** 2 / sigma ** 2
                            )),
                            "squared_error": error ** 2, "absolute_error": abs(error),
                            "basis_rank": 0 if basis is None else basis.rank,
                            "heldout_participant_in_basis": participant in training_ids,
                            "heldout_identity_in_other_dataset_basis": False,
                            "probe_rows_used_for_basis": 0,
                            "basis_source": "none" if basis is None else "pure_A_B_only",
                        })
    predictions = pd.DataFrame(rows)
    summary = predictions.groupby(["dataset", "probe_context", "model"], as_index=False).agg(
        heldout_log_likelihood=("heldout_log_likelihood", "sum"),
        mean_heldout_log_likelihood=("heldout_log_likelihood", "mean"),
        heldout_rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        heldout_mae=("absolute_error", "mean"),
        heldout_observations=("unit_id", "size"),
        heldout_participants=("participant_uid", "nunique"),
        basis_rank=("basis_rank", "max"),
        heldout_participant_in_basis=("heldout_participant_in_basis", "any"),
        heldout_identity_in_other_dataset_basis=("heldout_identity_in_other_dataset_basis", "any"),
        probe_rows_used_for_basis=("probe_rows_used_for_basis", "sum"),
        basis_source=("basis_source", "first"),
    )
    fold_scores = predictions.groupby(
        ["dataset", "probe_context", "model", "fold"], as_index=False
    ).heldout_log_likelihood.sum()
    shared_fold = fold_scores[fold_scores.model.eq("shared_ab_basis")][
        ["dataset", "probe_context", "fold", "heldout_log_likelihood"]
    ].rename(columns={"heldout_log_likelihood": "shared_fold_log_likelihood"})
    fold_scores = fold_scores.merge(
        shared_fold, on=["dataset", "probe_context", "fold"], how="left"
    )
    fold_scores["gain_over_shared"] = (
        fold_scores.heldout_log_likelihood - fold_scores.shared_fold_log_likelihood
    )
    uncertainty = fold_scores.groupby(
        ["dataset", "probe_context", "model"], as_index=False
    ).agg(
        heldout_log_likelihood_gain_over_shared=("gain_over_shared", "sum"),
        mean_fold_log_likelihood_gain_over_shared=("gain_over_shared", "mean"),
        paired_gain_over_shared_se=("gain_over_shared", "sem"),
        comparison_folds=("fold", "nunique"),
    )
    summary = summary.merge(
        uncertainty, on=["dataset", "probe_context", "model"], how="left"
    )
    summary["primary_rank"] = summary.groupby("dataset").heldout_log_likelihood.rank(
        ascending=False, method="min"
    ).astype(int)
    return summary, predictions


def _loto(x, y):
    errors = []
    for held in range(len(x)):
        train = np.arange(len(x)) != held
        design = np.column_stack([np.ones(train.sum()), x[train]])
        beta = np.linalg.pinv(design.T @ design + np.diag([0, 1e-6])) @ design.T @ y[train]
        errors.append((y[held] - np.array([1, x[held]]) @ beta) ** 2)
    return float(np.sqrt(np.mean(errors)))


def participant_structural_predictions(scores, coefficients, *, shrinkage_strength):
    merged = coefficients.merge(
        scores[["participant_uid", "component", "shrunken_component_score"]],
        on=["participant_uid", "component"], how="inner",
    )
    count = merged.groupby(["probe_context", "component"]).coefficient.transform("size")
    group_mean = merged.groupby(["probe_context", "component"]).coefficient.transform("mean")
    weight = count / (count + float(shrinkage_strength))
    merged["shrunken_probe_coefficient"] = weight * merged.coefficient + (1 - weight) * group_mean
    rows = []
    for (context, component), frame in merged.groupby(["probe_context", "component"]):
        x = frame.shrunken_component_score.to_numpy(dtype=float)
        y = frame.shrunken_probe_coefficient.to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.diag([0, float(shrinkage_strength) / max(len(x), 1)])
        beta = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
        rows.append({
            "probe_context": context, "component": component,
            "participants": len(frame),
            "pearson_r": float(pearsonr(x, y).statistic)
            if np.std(x) > 1e-10 and np.std(y) > 1e-10 else np.nan,
            "spearman_rho": float(spearmanr(x, y).statistic)
            if np.std(x) > 1e-10 and np.std(y) > 1e-10 else np.nan,
            "ridge_intercept": float(beta[0]), "ridge_slope": float(beta[1]),
            "loto_rmse": _loto(x, y),
            "hierarchical_shrinkage": True,
            "shrinkage_strength": float(shrinkage_strength),
        })
    return pd.DataFrame(rows)
