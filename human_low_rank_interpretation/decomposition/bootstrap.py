"""Participant bootstrap and fold stability for aligned human components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .align_components import align_components
from .estimate_basis import LowRankBasis, fit_pca_basis
from .participant_scores import participant_folds


def _resampled_profiles(profiles, participants, rng, replicate):
    sampled = rng.choice(participants, size=len(participants), replace=True)
    frames = []
    for draw, participant in enumerate(sampled):
        frame = profiles[profiles.participant_uid.astype(str).eq(str(participant))].copy()
        frame["participant_uid"] = frame.participant_uid.astype(str) + f"|boot{replicate}|draw{draw}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bootstrap_components(
    profiles: pd.DataFrame, reference: LowRankBasis, *, replicates: int,
    n_folds: int, random_seed: int, stability_threshold: float,
):
    rng = np.random.default_rng(random_seed)
    participants = profiles.participant_uid.astype(str).unique()
    loading_rows, match_rows = [], []
    for replicate in range(int(replicates)):
        sample = _resampled_profiles(profiles, participants, rng, replicate)
        fitted = fit_pca_basis(sample, rank=reference.rank)
        aligned, audit = align_components(reference.components, fitted.components)
        for component in range(reference.rank):
            match_rows.append({
                "resample_type": "bootstrap", "resample": replicate,
                "component": component + 1,
                "matched_component": int(audit.candidate_component[component]),
                "matched_cosine": float(audit.matched_cosine[component]),
                "sign_flipped": bool(audit.sign_flipped[component]),
            })
            for position in range(7):
                loading_rows.append({
                    "bootstrap": replicate, "component": component + 1,
                    "transition_position": position + 1,
                    "aligned_loading": float(aligned[position, component]),
                    "matched_cosine": float(audit.matched_cosine[component]),
                })
    for fold, training, _heldout in participant_folds(profiles, n_folds=n_folds):
        fitted = fit_pca_basis(training, rank=reference.rank)
        _aligned, audit = align_components(reference.components, fitted.components)
        for component in range(reference.rank):
            match_rows.append({
                "resample_type": "participant_fold", "resample": fold,
                "component": component + 1,
                "matched_component": int(audit.candidate_component[component]),
                "matched_cosine": float(audit.matched_cosine[component]),
                "sign_flipped": bool(audit.sign_flipped[component]),
            })
    boot = pd.DataFrame(loading_rows)
    matches = pd.DataFrame(match_rows)
    loading_summary = boot.groupby(
        ["component", "transition_position"], as_index=False
    ).agg(
        bootstrap_mean_loading=("aligned_loading", "mean"),
        bootstrap_ci_lower=("aligned_loading", lambda values: float(np.quantile(values, .025))),
        bootstrap_ci_upper=("aligned_loading", lambda values: float(np.quantile(values, .975))),
        sign_consistency=("aligned_loading", lambda values: float(max(
            np.mean(np.asarray(values) >= 0), np.mean(np.asarray(values) <= 0)
        ))),
    )
    stability = matches.groupby(["resample_type", "component"], as_index=False).agg(
        median_matched_cosine=("matched_cosine", "median"),
        mean_matched_cosine=("matched_cosine", "mean"),
        minimum_matched_cosine=("matched_cosine", "min"),
        sign_flip_rate=("sign_flipped", "mean"),
        resamples=("resample", "nunique"),
    )
    overall = matches.groupby("component", as_index=False).agg(
        median_matched_cosine=("matched_cosine", "median"),
        mean_matched_cosine=("matched_cosine", "mean"),
        minimum_matched_cosine=("matched_cosine", "min"),
        sign_flip_rate=("sign_flipped", "mean"),
        resamples=("resample", "size"),
    )
    overall["resample_type"] = "all"
    stability = pd.concat([stability, overall], ignore_index=True)
    stability["stability_threshold"] = float(stability_threshold)
    stability["stable_component"] = stability.median_matched_cosine.ge(stability_threshold)
    return boot, loading_summary, stability
