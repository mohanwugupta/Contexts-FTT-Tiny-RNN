"""Cross-validated rank selection, factor robustness, and parallel analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

from .estimate_basis import LowRankBasis, fit_pca_basis, profile_matrix
from .participant_scores import participant_folds


def _factor_basis(profiles, rank, seed):
    matrix = profile_matrix(profiles)
    values = matrix.to_numpy(dtype=float)
    model = FactorAnalysis(n_components=int(rank), random_state=int(seed)).fit(values)
    components = model.components_.T
    q, _ = np.linalg.qr(components)
    components = q[:, :int(rank)]
    for column in range(components.shape[1]):
        anchor = np.argmax(np.abs(components[:, column]))
        if components[anchor, column] < 0:
            components[:, column] *= -1
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    energy = singular ** 2
    variance = energy / energy.sum() if energy.sum() else np.zeros_like(energy)
    return LowRankBasis(
        mean=model.mean_, components=components, singular_values=singular,
        variance_explained=variance, cumulative_variance=np.cumsum(variance),
        rank=int(rank), training_participants=tuple(matrix.index.astype(str)),
        training_datasets=tuple(sorted(profiles.dataset.astype(str).unique())),
    )


def _noise_sd(training_probe):
    residual = training_probe.midpoint_deviation - training_probe.groupby(
        ["participant_uid", "transition_position"]
    ).midpoint_deviation.transform("mean")
    value = float(residual.std(ddof=1))
    return max(value if np.isfinite(value) else 0, .05)


def _probe_scores(heldout_probe, basis, sigma):
    rows = []
    for participant, frame in heldout_probe.groupby("participant_uid", sort=False):
        calibration = frame[frame.probe_split.eq("train")]
        test = frame[frame.probe_split.eq("test")]
        if calibration.empty or test.empty:
            continue
        mean_profile = calibration.groupby("transition_position").midpoint_deviation.mean()
        positions = mean_profile.index.astype(int).to_numpy()
        design = basis.components[positions - 1]
        coefficient = np.linalg.pinv(design) @ mean_profile.to_numpy(dtype=float)
        predicted = basis.components[test.transition_position.astype(int).to_numpy() - 1] @ coefficient
        error = test.midpoint_deviation.to_numpy(dtype=float) - predicted
        rows.extend([{
            "participant_uid": participant,
            "squared_error": float(value ** 2),
            "log_likelihood": float(-.5 * (
                math.log(2 * math.pi * sigma ** 2) + value ** 2 / sigma ** 2
            )),
        } for value in error])
    return pd.DataFrame(rows)


def _parallel_analysis(profiles, *, replicates, seed):
    matrix = profile_matrix(profiles).to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0)
    observed = np.linalg.svd(centered, compute_uv=False) ** 2 / max(len(matrix) - 1, 1)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(replicates)):
        permuted = np.column_stack([rng.permutation(matrix[:, column]) for column in range(7)])
        null.append(np.linalg.svd(permuted - permuted.mean(axis=0), compute_uv=False) ** 2 /
                    max(len(matrix) - 1, 1))
    threshold = np.quantile(np.asarray(null), .95, axis=0)
    return observed, threshold, int(np.sum(observed > threshold))


def evaluate_dimensions(
    profiles: pd.DataFrame, probes: dict[str, pd.DataFrame], *, candidate_ranks,
    n_folds: int, parallel_replicates: int, random_seed: int,
):
    rows = []
    parallel = {}
    for dataset, dataset_profiles in profiles.groupby("dataset", sort=False):
        probe = probes[dataset]
        observed, threshold, parallel_rank = _parallel_analysis(
            dataset_profiles, replicates=parallel_replicates,
            seed=random_seed + (0 if dataset == "Combined" else 1000),
        )
        parallel[dataset] = (observed, threshold, parallel_rank)
        for method in ("PCA", "FactorAnalysis"):
            for rank in candidate_ranks:
                for fold, training, heldout in participant_folds(
                    dataset_profiles, n_folds=n_folds
                ):
                    basis = (
                        fit_pca_basis(training, rank=rank)
                        if method == "PCA" else _factor_basis(training, rank, random_seed + fold)
                    )
                    held_matrix = profile_matrix(heldout).to_numpy(dtype=float)
                    reconstruction = basis.reconstruct(held_matrix)
                    held_ids = set(heldout.participant_uid.astype(str))
                    held_probe = probe[probe.participant_uid.astype(str).isin(held_ids)]
                    training_probe = probe[~probe.participant_uid.astype(str).isin(held_ids)]
                    predictions = _probe_scores(held_probe, basis, _noise_sd(training_probe))
                    rows.append({
                        "row_type": "fold", "dataset": dataset, "method": method,
                        "rank": int(rank), "fold": fold,
                        "heldout_reconstruction_mse": float(np.mean(
                            (held_matrix - reconstruction) ** 2
                        )),
                        "heldout_probe_log_likelihood": float(predictions.log_likelihood.sum()),
                        "mean_heldout_probe_log_likelihood": float(predictions.log_likelihood.mean()),
                        "heldout_probe_rmse": float(np.sqrt(predictions.squared_error.mean())),
                        "heldout_participants": int(len(held_ids)),
                        "basis_training_participants": int(training.participant_uid.nunique()),
                        "heldout_participant_in_basis": False,
                        "probe_rows_used_for_basis": 0,
                        "parallel_analysis_rank": parallel_rank,
                        "observed_eigenvalue_at_rank": float(observed[rank - 1]),
                        "parallel_95pct_eigenvalue_at_rank": float(threshold[rank - 1]),
                    })
    folds = pd.DataFrame(rows)
    summaries = folds.groupby(["dataset", "method", "rank"], as_index=False).agg(
        heldout_reconstruction_mse=("heldout_reconstruction_mse", "mean"),
        reconstruction_mse_se=("heldout_reconstruction_mse", "sem"),
        heldout_probe_log_likelihood=("heldout_probe_log_likelihood", "sum"),
        mean_heldout_probe_log_likelihood=("mean_heldout_probe_log_likelihood", "mean"),
        probe_log_likelihood_se=("mean_heldout_probe_log_likelihood", "sem"),
        heldout_probe_rmse=("heldout_probe_rmse", "mean"),
        folds=("fold", "nunique"),
        heldout_participants=("heldout_participants", "sum"),
        parallel_analysis_rank=("parallel_analysis_rank", "first"),
        observed_eigenvalue_at_rank=("observed_eigenvalue_at_rank", "first"),
        parallel_95pct_eigenvalue_at_rank=("parallel_95pct_eigenvalue_at_rank", "first"),
    )
    primary = folds[folds.method.eq("PCA")].groupby("rank", as_index=False).agg(
        mean_score=("mean_heldout_probe_log_likelihood", "mean"),
        score_se=("mean_heldout_probe_log_likelihood", "sem"),
    )
    best = primary.loc[primary.mean_score.idxmax()]
    eligible = primary[primary.mean_score.ge(best.mean_score - best.score_se)]
    selected_rank = int(eligible["rank"].min())
    summaries["one_se_selected_rank"] = selected_rank
    summaries["selected_primary_rank"] = (
        summaries.method.eq("PCA") & summaries["rank"].eq(selected_rank)
    )
    summaries["selection_rule"] = "smallest rank within 1 SE of best dataset-fold heldout probe likelihood"
    folds["one_se_selected_rank"] = selected_rank
    folds["selected_primary_rank"] = (
        folds.method.eq("PCA") & folds["rank"].eq(selected_rank)
    )
    return pd.concat([folds, summaries.assign(row_type="summary", fold=np.nan)],
                     ignore_index=True, sort=False), selected_rank
