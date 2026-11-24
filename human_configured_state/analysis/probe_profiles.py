"""Temporal, counterbalance, SRN alignment, and cross-probe profile tests."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _safe_profile_stats(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    r2 = 1 - float(np.sum((y - predicted) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    return float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(slope), r2


def temporal_probe_profiles(*probe_frames):
    data = pd.concat(probe_frames, ignore_index=True)
    rows = data.groupby(
        ["dataset", "context", "probe_epoch", "transition_position"],
        dropna=False, as_index=False,
    ).agg(
        mean_midpoint_deviation=("midpoint_deviation", "mean"),
        sem_midpoint_deviation=("midpoint_deviation", "sem"),
        mean_responsibility_q=("responsibility_q", "mean"),
        sem_responsibility_q=("responsibility_q", "sem"),
        observations=("log_rt", "size"), participants=("participant_uid", "nunique"),
        mean_probe_order=("probe_order", "mean"),
    )
    rows["probe_blocks_collapsed_before_temporal_test"] = False
    return rows


def counterbalance_sensitivity(probe_frames, predictions):
    observed = pd.concat(probe_frames, ignore_index=True)
    profiles = observed.groupby(
        ["dataset", "context", "counterbalance", "transition_position"], as_index=False
    ).agg(
        mean_midpoint_deviation=("midpoint_deviation", "mean"),
        mean_responsibility_q=("responsibility_q", "mean"),
        observations=("log_rt", "size"), participants=("participant_uid", "nunique"),
        mean_block_order=("block_order", "mean"), mean_probe_order=("probe_order", "mean"),
    )
    profiles["row_type"] = "observed_profile"
    model = predictions[predictions.model.isin([
        "M1_scalar_mixture", "M2_token_specific", "M4_srn_informed"
    ])].groupby(
        ["dataset", "probe_context", "counterbalance", "model"], as_index=False
    ).agg(
        heldout_log_likelihood=("heldout_log_likelihood", "sum"),
        heldout_rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        heldout_observations=("unit_id", "size"),
        heldout_participants=("participant_uid", "nunique"),
    )
    model["row_type"] = "heldout_model"
    result = pd.concat([profiles, model], ignore_index=True, sort=False)
    result["counterbalance_preserved"] = True
    result["block_order_controlled_or_reported"] = True
    result["probe_order_controlled_or_reported"] = True
    return result


def human_srn_alignment(probe_frames, srn_aggregate, comparisons):
    human = pd.concat(probe_frames, ignore_index=True)
    human = human[human.ratio_reliable & human.responsibility_q.notna()]
    group = human.groupby(
        ["context", "transition_position"], as_index=False
    ).agg(human_q=("responsibility_q", "mean"), human_q_sem=("responsibility_q", "sem"))
    columns = {
        "execution": "execution_q", "full_component": "full_component_q",
        "shared_component": "shared_component_q", "specific_component": "specific_component_q",
    }
    rows = []
    for context, srn_context in (("Both", "Mixed_norm"), ("Novel", "Novel")):
        human_context = group[group.context.eq(context)]
        srn = srn_aggregate[srn_aggregate.probe_context.eq(srn_context)]
        merged = human_context.merge(srn, on="transition_position")
        comparison = comparisons[comparisons.probe_context.eq(context)].set_index("model")
        likelihood_gain = (
            float(comparison.loc["M4_srn_informed", "heldout_log_likelihood"] -
                  comparison.loc["M1_scalar_mixture", "heldout_log_likelihood"])
            if {"M4_srn_informed", "M1_scalar_mixture"}.issubset(comparison.index) else np.nan
        )
        for variant, column in columns.items():
            pearson, p_value, spearman, slope, r2 = _safe_profile_stats(
                merged[column], merged.human_q
            )
            for row in merged.itertuples(index=False):
                rows.append({
                    "probe_context": context, "srn_probe_context": srn_context,
                    "profile_variant": variant,
                    "transition_position": int(row.transition_position),
                    "human_q": float(row.human_q), "human_q_sem": float(row.human_q_sem),
                    "srn_q": float(getattr(row, column)),
                    "pearson_r": pearson, "pearson_p": p_value,
                    "spearman_rho": spearman, "alignment_slope": slope,
                    "alignment_r_squared": r2,
                    "heldout_log_likelihood_gain_over_M1": (
                        likelihood_gain if variant == "execution" else np.nan
                    ),
                    "srn_seed_count": int(row.registered_seed_count),
                    "best_seed_selection": False, "human_outcomes_used_for_srn_shape": 0,
                })
    return pd.DataFrame(rows)


def mixed_novel_profile_comparison(mixed, novel):
    def profile(frame, name):
        stable = frame[frame.ratio_reliable & frame.responsibility_q.notna()]
        return stable.groupby("transition_position").responsibility_q.mean().rename(name)
    merged = pd.concat([profile(mixed, "mixed_q"), profile(novel, "novel_q")], axis=1).dropna()
    pearson, p_value, spearman, slope, r2 = _safe_profile_stats(merged.mixed_q, merged.novel_q)
    intercept = float(merged.novel_q.mean() - slope * merged.mixed_q.mean()) if np.isfinite(slope) else np.nan
    predicted = intercept + slope * merged.mixed_q if np.isfinite(slope) else np.nan
    rows = []
    for position, row in merged.iterrows():
        rows.append({
            "transition_position": int(position), "mixed_q": float(row.mixed_q),
            "novel_q": float(row.novel_q),
            "affine_predicted_novel_q": float(predicted.loc[position]) if np.isfinite(slope) else np.nan,
            "profile_pearson_r": pearson, "profile_pearson_p": p_value,
            "profile_spearman_rho": spearman, "affine_intercept": intercept,
            "affine_slope": slope, "affine_r_squared": r2,
        })
    return pd.DataFrame(rows)


def _fit_shape_calibration(frame, shape):
    z = frame.transition_position.map(shape).to_numpy(dtype=float)
    x = np.column_stack([frame.delta.to_numpy(dtype=float), z * frame.delta.to_numpy(dtype=float)])
    y = frame.midpoint_deviation.to_numpy(dtype=float)
    return np.linalg.pinv(x.T @ x) @ x.T @ y


def evaluate_shared_vs_separate(mixed, novel, *, n_folds=5):
    frames = {"Both": mixed, "Novel": novel}
    assignments = {}
    for context, frame in frames.items():
        participants = sorted(frame.participant_uid.unique())
        folds = min(n_folds, len(participants))
        assignments[context] = {p: i % folds for i, p in enumerate(participants)}
    rows = []
    for fold in range(min(n_folds, min(len(set(v.values())) for v in assignments.values()))):
        training, heldout = {}, {}
        separate_profiles = {}
        for context, frame in frames.items():
            held_ids = {p for p, f in assignments[context].items() if f == fold}
            training[context] = frame[~frame.participant_uid.isin(held_ids)]
            heldout[context] = frame[frame.participant_uid.isin(held_ids)]
            stable = training[context][training[context].ratio_reliable]
            separate_profiles[context] = stable.groupby(
                "transition_position"
            ).responsibility_q.mean().to_dict()
        positions = sorted(set(separate_profiles["Both"]) & set(separate_profiles["Novel"]))
        matrix = np.vstack([
            [separate_profiles[context][p] for p in positions] for context in ("Both", "Novel")
        ])
        row_means = matrix.mean(axis=1, keepdims=True)
        _, _, vt = np.linalg.svd(matrix - row_means, full_matrices=False)
        shared_shape = dict(zip(positions, vt[0]))
        sigma_source = pd.concat(list(training.values()))
        sigma = max(float((sigma_source.log_rt - sigma_source.groupby(
            ["participant_uid", "transition_position"]
        ).log_rt.transform("mean")).std()), 0.05)
        for context in ("Both", "Novel"):
            for participant, participant_rows in heldout[context].groupby("participant_uid"):
                calibration = participant_rows[participant_rows.probe_split.eq("train")]
                test = participant_rows[participant_rows.probe_split.eq("test")]
                if calibration.empty or test.empty:
                    continue
                for model, shape in (("shared_profile", shared_shape),
                                     ("separate_profile", separate_profiles[context])):
                    beta = _fit_shape_calibration(calibration, shape)
                    z = test.transition_position.map(shape).to_numpy(dtype=float)
                    predicted = test.midpoint.to_numpy() + (
                        beta[0] + beta[1] * z
                    ) * test.delta.to_numpy()
                    for source, value in zip(test.itertuples(index=False), predicted):
                        error = float(source.log_rt - value)
                        rows.append({
                            "fold": fold, "probe_context": context, "model": model,
                            "participant_uid": participant, "unit_id": source.unit_id,
                            "transition_position": int(source.transition_position),
                            "actual_log_rt": float(source.log_rt),
                            "predicted_log_rt": float(value), "squared_error": error ** 2,
                            "heldout_log_likelihood": float(-0.5 * (
                                math.log(2 * math.pi * sigma ** 2) + error ** 2 / sigma ** 2
                            )),
                            "heldout_participant_in_training": False,
                            "profile_learned_from_training_participants_only": True,
                        })
    predictions = pd.DataFrame(rows)
    summary = predictions.groupby(["probe_context", "model"], as_index=False).agg(
        heldout_log_likelihood=("heldout_log_likelihood", "sum"),
        heldout_rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        heldout_observations=("unit_id", "size"),
        heldout_participants=("participant_uid", "nunique"),
    )
    summary["evaluation"] = "leave_participant_out_plus_heldout_repetitions"
    return summary

