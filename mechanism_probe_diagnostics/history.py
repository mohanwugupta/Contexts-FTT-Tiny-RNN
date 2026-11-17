"""History derivation, identifiability gates, and synthetic separation checks."""

from __future__ import annotations

import json
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import ttest_rel


HISTORY_CANDIDATES = (
    ("immediately_preceding_context", "categorical", "derived previous sequence context"),
    ("most_recent_unambiguous_ab", "categorical", "last preceding sequence in A or B"),
    ("signed_ab_recency", "continuous", "sequences_since_A minus sequences_since_B"),
    ("sequences_since_last_a", "continuous", "sequence count since preceding A exposure"),
    ("sequences_since_last_b", "continuous", "sequence count since preceding B exposure"),
    ("probe_order", "continuous", "canonical probe block order"),
    ("sequence_repetition", "continuous", "position within current probe block"),
    ("time_since_last_ab", "continuous", "minimum sequence count since A or B"),
    ("test_condition_order", "continuous", "canonical probe block order"),
)


def derive_sequence_history(
    sequences: pd.DataFrame, *, participant_col: str = "participant_uid",
    context_col: str = "condition", block_order_col: str = "block_order",
    sequence_col: str = "sequence_repetition",
) -> pd.DataFrame:
    """Derive history using only observations preceding the current sequence."""
    required = {participant_col, context_col, block_order_col, sequence_col}
    missing = sorted(required.difference(sequences))
    if missing:
        raise ValueError(f"Sequence history source is missing columns: {missing}")
    out_rows = []
    for participant, one in sequences.groupby(participant_col, sort=False):
        ordered = one.sort_values([block_order_col, sequence_col], kind="stable").copy()
        last_a = last_b = None
        prior_context = None
        for chronological_index, (_, row) in enumerate(ordered.iterrows()):
            since_a = np.nan if last_a is None else chronological_index - last_a
            since_b = np.nan if last_b is None else chronological_index - last_b
            if last_a is None and last_b is None:
                most_recent = None
            elif last_b is None or (last_a is not None and last_a > last_b):
                most_recent = "A"
            else:
                most_recent = "B"
            record = row.to_dict()
            record.update({
                "chronological_sequence_index": chronological_index,
                "immediately_preceding_context": prior_context,
                "most_recent_unambiguous_ab": most_recent,
                "sequences_since_last_a": since_a,
                "sequences_since_last_b": since_b,
                "signed_ab_recency": (
                    since_a - since_b if np.isfinite(since_a) and np.isfinite(since_b) else np.nan
                ),
                "time_since_last_ab": np.nanmin([since_a, since_b])
                if np.isfinite(since_a) or np.isfinite(since_b) else np.nan,
                "test_condition_order": row.get("probe_order", np.nan),
            })
            out_rows.append(record)
            current = str(row[context_col])
            if current == "A":
                last_a = chronological_index
            elif current == "B":
                last_b = chronological_index
            prior_context = current
    return pd.DataFrame(out_rows)


def sequence_table_from_tokens(tokens: pd.DataFrame) -> pd.DataFrame:
    keys = ["participant_uid", "block_id", "original_sequence_in_block"]
    columns = keys + [
        "condition", "original_condition", "dataset", "block_order",
        "sequence_repetition", "probe_epoch", "probe_order", "previous_context",
    ]
    sequence = tokens[columns].drop_duplicates(keys)
    if sequence.duplicated(keys).any():
        raise ValueError("History source is not unique at the sequence level.")
    return derive_sequence_history(sequence)


def _within_fraction(frame: pd.DataFrame, predictor: str, kind: str, threshold: int) -> tuple[float, float]:
    varied, supported = [], []
    for _, one in frame.groupby("participant_uid", sort=False):
        values = one[predictor].dropna()
        varied.append(values.nunique() >= 2)
        if kind == "categorical":
            counts = values.astype(str).value_counts()
            supported.append(len(counts) >= 2 and bool((counts >= threshold).all()))
        else:
            supported.append(values.nunique() >= threshold)
    return float(np.mean(varied)), float(np.mean(supported))


def _epoch_determined(frame: pd.DataFrame, predictor: str) -> bool:
    groups = frame.groupby(["participant_uid", "probe_epoch"], dropna=False)[predictor].nunique(dropna=True)
    return bool(len(groups) and groups.max() <= 1)


def _order_determined(frame: pd.DataFrame, predictor: str, kind: str) -> bool:
    if predictor == "sequence_repetition":
        return True
    if kind != "continuous":
        return predictor in {"probe_order", "sequence_repetition", "test_condition_order"}
    correlations = []
    for _, one in frame.groupby(["participant_uid", "probe_epoch"], dropna=False):
        pair = one[[predictor, "sequence_repetition"]].dropna().drop_duplicates()
        if len(pair) >= 3 and pair[predictor].nunique() > 1:
            correlations.append(abs(float(pair.corr().iloc[0, 1])))
    return bool(correlations and np.mean(np.asarray(correlations) > .999) >= .5)


def _encode(series: pd.Series, kind: str) -> pd.Series:
    if kind == "continuous":
        return pd.to_numeric(series, errors="coerce")
    levels = {value: number for number, value in enumerate(sorted(series.dropna().astype(str).unique()))}
    return series.astype(str).map(levels).where(series.notna())


def history_identifiability_audit(
    history: pd.DataFrame, *, thresholds: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply preregistered support, order/epoch, rank, correlation, and VIF gates."""
    probe = history[history.condition.eq("Mixed")].copy()
    # One row per probe sequence for support; history is already sequence-level.
    audit_rows = []
    encoded = {}
    for predictor, kind, derivation in HISTORY_CANDIDATES:
        if predictor not in probe:
            audit_rows.append({
                "unit_id": predictor, "condition": "Mixed", "predictor": predictor,
                "kind": kind, "derivation": derivation, "status": "missing",
                "eligible": False, "reason": "source field unavailable",
            })
            continue
        values = probe[predictor]
        within, support = _within_fraction(
            probe, predictor, kind,
            int(thresholds["categorical_min_sequences_per_level"] if kind == "categorical"
                else thresholds["continuous_min_distinct_values"]),
        )
        epoch_determined = _epoch_determined(probe, predictor)
        order_determined = _order_determined(probe, predictor, kind)
        numeric = _encode(values, kind)
        encoded[predictor] = numeric
        # Rank of predictor with intercept, probe epoch, and sequence order.
        design = pd.concat([
            pd.Series(1.0, index=probe.index, name="intercept"),
            numeric.rename(predictor),
            pd.get_dummies(probe.probe_epoch.astype(str), prefix="epoch", drop_first=True, dtype=float),
            pd.to_numeric(probe.sequence_repetition, errors="coerce").rename("trial_order"),
        ], axis=1).dropna()
        full_rank = bool(len(design) and np.linalg.matrix_rank(design.to_numpy(float)) == design.shape[1])
        required_fraction = float(thresholds["participant_fraction"])
        reasons = []
        if within < required_fraction:
            reasons.append("insufficient within-participant variation")
        if support < required_fraction:
            reasons.append("insufficient repeated support")
        if epoch_determined:
            reasons.append("perfectly determined by probe epoch")
        if order_determined:
            reasons.append("perfectly determined by trial/order")
        if not full_rank:
            reasons.append("rank-deficient with epoch/order covariates")
        eligible = not reasons
        audit_rows.append({
            "unit_id": predictor, "condition": "Mixed", "predictor": predictor,
            "kind": kind, "source_column": predictor, "derivation": derivation,
            "missing_fraction": float(values.isna().mean()),
            "within_participant_variation_fraction": within,
            "repeated_support_fraction": support,
            "value_count_or_range": (
                json.dumps(values.astype(str).value_counts().to_dict(), sort_keys=True)
                if kind == "categorical" else
                json.dumps({"min": float(numeric.min()), "max": float(numeric.max()),
                            "distinct": int(numeric.nunique())})
            ),
            "association_with_probe_condition": "not_estimable_Mixed_primary_only",
            "association_with_transition": "none_by_construction_history_constant_within_sequence",
            "perfectly_determined_by_probe_epoch": epoch_determined,
            "perfectly_determined_by_trial_order": order_determined,
            "design_matrix_rank": int(np.linalg.matrix_rank(design.to_numpy(float))) if len(design) else 0,
            "design_matrix_columns": int(design.shape[1]),
            "design_matrix_full_rank": full_rank,
            "randomization_status": (
                "counterbalanced" if predictor in {"most_recent_unambiguous_ab", "immediately_preceding_context"}
                else "observational_or_deterministic_order"
            ),
            "eligible": eligible, "status": "eligible" if eligible else "ineligible",
            "reason": "; ".join(reasons) if reasons else "passes all preregistered gates",
        })
    audit = pd.DataFrame(audit_rows)

    usable = [c for c in encoded if encoded[c].notna().sum()]
    matrix = pd.DataFrame({c: encoded[c] for c in usable})
    correlations = matrix.corr()
    confound_rows = []
    for left in usable:
        for right in usable:
            correlation = float(correlations.loc[left, right]) if pd.notna(correlations.loc[left, right]) else np.nan
            confound_rows.append({
                "unit_id": f"{left}|{right}", "condition": "Mixed",
                "predictor_a": left, "predictor_b": right,
                "correlation": correlation,
                "exceeds_threshold": bool(left != right and np.isfinite(correlation)
                                          and abs(correlation) >= float(thresholds["max_abs_correlation"])),
            })
    confounds = pd.DataFrame(confound_rows)
    max_correlations = {}
    vifs = {}
    complete = matrix.dropna()
    for predictor in usable:
        others = [c for c in usable if c != predictor]
        correlations_one = [
            abs(float(correlations.loc[predictor, other])) for other in others
            if pd.notna(correlations.loc[predictor, other])
        ]
        max_correlations[predictor] = max(correlations_one, default=np.nan)
        if len(complete) and others and complete[predictor].nunique() > 1:
            y = complete[predictor].to_numpy(float)
            x = np.column_stack([np.ones(len(complete)), complete[others].to_numpy(float)])
            predicted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
            denominator = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - float(np.sum((y - predicted) ** 2)) / denominator if denominator > 0 else np.nan
            vifs[predictor] = np.inf if np.isfinite(r2) and r2 >= 1 - 1e-12 else (
                1 / (1 - r2) if np.isfinite(r2) else np.nan
            )
        else:
            vifs[predictor] = np.nan
    audit["max_absolute_predictor_correlation"] = audit.predictor.map(max_correlations)
    audit["variance_inflation_factor"] = audit.predictor.map(vifs)
    for index, row in audit.iterrows():
        if not bool(row.get("eligible", False)):
            continue
        reasons = []
        correlation = row.max_absolute_predictor_correlation
        vif = row.variance_inflation_factor
        if np.isfinite(correlation) and correlation >= float(thresholds["max_abs_correlation"]):
            reasons.append("absolute predictor correlation exceeds threshold")
        if np.isfinite(vif) and vif >= float(thresholds["max_vif"]):
            reasons.append("variance inflation factor exceeds threshold")
        if reasons:
            audit.loc[index, "eligible"] = False
            audit.loc[index, "status"] = "ineligible"
            audit.loc[index, "reason"] = "; ".join(reasons)
    eligible = audit.loc[audit.eligible.fillna(False), "predictor"].tolist()
    status = {
        "stage": 3, "condition": "Mixed", "status": "pass" if eligible else "stopped_nonidentifiable",
        "eligible_predictors": eligible,
        "reason": ("At least one history predictor passed every preregistered gate."
                   if eligible else "No history predictor passed support, epoch/order, and full-rank gates."),
        "psychological_null_inferred": False,
    }
    return audit, confounds, status


def _history_design(series: pd.Series, kind: str) -> tuple[np.ndarray, list[str]]:
    if kind == "continuous":
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        mean, sd = float(np.nanmean(values)), float(np.nanstd(values))
        sd = sd if sd > 1e-12 else 1.0
        return ((values - mean) / sd)[:, None], ["standardized_history"]
    dummies = pd.get_dummies(series.astype(str), drop_first=True, dtype=float)
    if dummies.shape[1] == 0:
        return np.zeros((len(series), 1)), ["constant_history"]
    return dummies.to_numpy(float), [f"level_{c}" for c in dummies.columns]


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, float)
    result = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result
    order = finite[np.argsort(values[finite])]
    adjusted = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def fit_identified_history_models(
    tokens: pd.DataFrame, sequence_history: pd.DataFrame, human_profiles: pd.DataFrame,
    eligible_audit: pd.DataFrame, *, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit individually frozen H0/Hdelta/Hq/Hboth models with block holdout.

    Shared history coefficients accompany participant-specific delta and q
    intercepts.  The observation equation contains no second intercept.
    """
    eligible = eligible_audit[eligible_audit.eligible.fillna(False)]
    if eligible.empty:
        raise ValueError("History models require at least one predictor that passed the audit.")
    sequence_keys = ["participant_uid", "block_id", "original_sequence_in_block"]
    history_columns = sequence_keys + ["probe_order", "probe_epoch"] + eligible.predictor.tolist()
    observations = tokens[tokens.condition.eq("Mixed")].merge(
        sequence_history[history_columns], on=sequence_keys, how="inner", validate="many_to_one",
        suffixes=("", "_history"),
    )
    endpoint = human_profiles[
        human_profiles.analysis_scale.eq("mean_rt_ms") & human_profiles.condition.isin(["A", "B"])
    ].pivot(index=["participant_uid", "transition_position"], columns="condition", values="M")
    observations = observations.merge(endpoint.reset_index(), on=["participant_uid", "transition_position"],
                                      how="inner", validate="many_to_one")
    observations = observations.reset_index(drop=True)
    observations["actual"] = observations.mean_rt.astype(float)
    observations["profile_id"] = (
        observations.participant_uid.astype(str) + "|b" + observations.block_id.astype(str)
        + "|s" + observations.original_sequence_in_block.astype(str)
    )
    comparison_rows, parameter_rows, prediction_rows = [], [], []
    rng = np.random.default_rng(seed)

    candidate_specs = [(row.predictor, row.kind, False) for row in eligible.itertuples(index=False)]
    if "sequence_repetition" in observations:
        candidate_specs.append(("sequence_repetition", "continuous", True))
    for predictor, kind, is_order_control in candidate_specs:
        x_all, x_names = _history_design(observations[predictor], kind)
        variants = [("trial_order_only" if is_order_control else "observed", x_all)]
        if not is_order_control:
            permuted = x_all.copy()
            permutation_frame = observations[["participant_uid", "probe_epoch"]].copy()
            for _, indices in permutation_frame.groupby(["participant_uid", "probe_epoch"], dropna=False).groups.items():
                order = np.asarray(list(indices), int)
                permuted[order] = permuted[rng.permutation(order)]
            variants.append(("within_participant_epoch_permutation", permuted))

        for control, x_values in variants:
            fold_metrics = []
            for held_block in sorted(pd.to_numeric(observations.probe_order, errors="coerce").dropna().unique()):
                test_mask = pd.to_numeric(observations.probe_order, errors="coerce").eq(held_block).to_numpy()
                train_mask = ~test_mask
                train_participants = sorted(observations.loc[train_mask, "participant_uid"].unique())
                pmap = {p: i for i, p in enumerate(train_participants)}
                train_indices = observations.loc[train_mask, "participant_uid"].map(pmap).to_numpy(int)
                test_valid = test_mask & observations.participant_uid.isin(pmap).to_numpy()
                test_indices = observations.loc[test_valid, "participant_uid"].map(pmap).to_numpy(int)
                a_train, b_train = observations.loc[train_mask, "A"].to_numpy(float), observations.loc[train_mask, "B"].to_numpy(float)
                y_train = observations.loc[train_mask, "actual"].to_numpy(float)
                a_test, b_test = observations.loc[test_valid, "A"].to_numpy(float), observations.loc[test_valid, "B"].to_numpy(float)
                y_test = observations.loc[test_valid, "actual"].to_numpy(float)
                x_train, x_test = x_values[train_mask], x_values[test_valid]
                n_participants, n_history = len(train_participants), x_values.shape[1]

                for model in ("H0", "Hdelta", "Hq", "Hboth"):
                    use_delta = model in {"Hdelta", "Hboth"}
                    use_q = model in {"Hq", "Hboth"}
                    n_parameters = 2 * n_participants + n_history * (use_delta + use_q)
                    initial = np.zeros(n_parameters)
                    midpoint_train = .5 * (a_train + b_train)
                    for participant, index in pmap.items():
                        mask = train_indices == index
                        initial[index] = float(np.mean(y_train[mask] - midpoint_train[mask]))

                    def predict(params, pidx, a, b, x):
                        delta = params[pidx]
                        theta = params[n_participants + pidx]
                        cursor = 2 * n_participants
                        if use_delta:
                            delta = delta + x @ params[cursor:cursor + n_history]
                            cursor += n_history
                        if use_q:
                            theta = theta + x @ params[cursor:cursor + n_history]
                        q = 1 / (1 + np.exp(-np.clip(theta, -30, 30)))
                        return delta + (1 - q) * a + q * b

                    fit = least_squares(
                        lambda params: y_train - predict(params, train_indices, a_train, b_train, x_train),
                        initial, max_nfev=800,
                    )
                    train_residual = y_train - predict(fit.x, train_indices, a_train, b_train, x_train)
                    predicted = predict(fit.x, test_indices, a_test, b_test, x_test)
                    residual = y_test - predicted
                    sigma = max(float(np.sqrt(np.mean(train_residual ** 2))), 1e-8)
                    current = observations.loc[test_valid].copy()
                    current["predicted"] = predicted
                    current["residual"] = residual
                    current["predictor"] = predictor
                    current["control"] = control
                    current["model"] = model
                    prediction_rows.append(current)
                    levels = current.assign(predicted=predicted).groupby("profile_id").agg(
                        actual=("actual", "mean"), predicted=("predicted", "mean")
                    )
                    centered_actual = y_test - current.groupby("profile_id").actual.transform("mean").to_numpy()
                    centered_predicted = predicted - current.assign(predicted=predicted).groupby("profile_id").predicted.transform("mean").to_numpy()
                    centered_sst = float(np.sum(centered_actual ** 2))
                    fold_metrics.append({
                        "unit_id": f"{predictor}|{control}|{model}|block{int(held_block)}",
                        "condition": "Mixed", "predictor": predictor, "control": control,
                        "model": model, "heldout_probe_block": int(held_block),
                        "n_test_rows": len(y_test), "n_participants": n_participants,
                        "heldout_rmse": float(np.sqrt(np.mean(residual ** 2))),
                        "heldout_mae": float(np.mean(np.abs(residual))),
                        "heldout_log_likelihood": float(np.sum(-.5 * np.log(2 * np.pi * sigma ** 2)
                                                                  - .5 * residual ** 2 / sigma ** 2)),
                        "level_calibration_correlation": float(levels.corr().iloc[0, 1]),
                        "centered_shape_correlation": float(np.corrcoef(centered_actual, centered_predicted)[0, 1]),
                        "centered_shape_variance_explained": (
                            1 - float(np.sum((centered_actual - centered_predicted) ** 2)) / centered_sst
                            if centered_sst > 0 else np.nan
                        ),
                        "delta_intercept_sd": float(np.std(fit.x[:n_participants])),
                        "q_intercept_sd": float(np.std(1 / (1 + np.exp(-fit.x[n_participants:2*n_participants])))),
                    })
                    cursor = 2 * n_participants
                    if use_delta:
                        for name, estimate in zip(x_names, fit.x[cursor:cursor + n_history]):
                            parameter_rows.append({
                                "unit_id": f"{predictor}|{control}|{model}|delta|{name}|{int(held_block)}",
                                "condition": "Mixed", "predictor": predictor, "control": control,
                                "model": model, "heldout_probe_block": int(held_block),
                                "parameter": "history_delta", "term": name, "estimate": float(estimate),
                            })
                        cursor += n_history
                    if use_q:
                        for name, estimate in zip(x_names, fit.x[cursor:cursor + n_history]):
                            parameter_rows.append({
                                "unit_id": f"{predictor}|{control}|{model}|q|{name}|{int(held_block)}",
                                "condition": "Mixed", "predictor": predictor, "control": control,
                                "model": model, "heldout_probe_block": int(held_block),
                                "parameter": "history_q", "term": name, "estimate": float(estimate),
                            })
            comparison_rows.extend(fold_metrics)

    comparison = pd.DataFrame(comparison_rows)
    parameters_fold = pd.DataFrame(parameter_rows)
    if len(parameters_fold):
        parameters = parameters_fold.groupby(
            ["condition", "predictor", "control", "model", "parameter", "term"], as_index=False
        ).estimate.agg(["mean", "std", "count"]).reset_index().rename(columns={"mean": "estimate"})
        parameters["standard_error"] = parameters["std"] / np.sqrt(parameters["count"])
        parameters["ci_lower"] = parameters.estimate - 1.96 * parameters.standard_error
        parameters["ci_upper"] = parameters.estimate + 1.96 * parameters.standard_error
        parameters["unit_id"] = parameters.predictor + "|" + parameters.model + "|" + parameters.parameter + "|" + parameters.term
        parameters["interval_basis"] = "leave-one-probe-block fold dispersion"
    else:
        parameters = pd.DataFrame()

    # Exploratory individual-predictor FDR comparisons against matched H0 folds.
    comparison["p_value_vs_H0"] = np.nan
    primary = comparison[comparison.control.eq("observed")]
    tests = []
    for (predictor, model), one in primary[~primary.model.eq("H0")].groupby(["predictor", "model"]):
        baseline = primary[(primary.predictor.eq(predictor)) & primary.model.eq("H0")]
        joined = one.merge(baseline[["heldout_probe_block", "heldout_rmse"]], on="heldout_probe_block",
                           suffixes=("", "_H0"))
        p = float(ttest_rel(joined.heldout_rmse, joined.heldout_rmse_H0,
                            alternative="less").pvalue) if len(joined) >= 2 else np.nan
        tests.append((predictor, model, p))
    adjusted = _bh_adjust(np.array([x[2] for x in tests]))
    for (predictor, model, p), q_value in zip(tests, adjusted):
        mask = comparison.predictor.eq(predictor) & comparison.model.eq(model) & comparison.control.eq("observed")
        comparison.loc[mask, "p_value_vs_H0"] = p
        comparison.loc[mask, "fdr_q_value"] = q_value
    return comparison, parameters, pd.concat(prediction_rows, ignore_index=True)


def synthetic_history_separation(
    a: np.ndarray, b: np.ndarray, history: np.ndarray, observations: np.ndarray,
) -> dict[str, float]:
    """Small recovery helper used by T11 and pipeline preflight."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    h, y = np.asarray(history, float), np.asarray(observations, float)
    n_t, k = y.shape

    def residual(params, model):
        delta0, theta0 = params[:2]
        cursor = 2
        gamma = params[cursor] if model in {"Hdelta", "Hboth"} else 0.0
        cursor += int(model in {"Hdelta", "Hboth"})
        beta = params[cursor] if model in {"Hq", "Hboth"} else 0.0
        delta = delta0 + gamma * h
        q = 1 / (1 + np.exp(-(theta0 + beta * h)))
        pred = delta[:, None] + (1 - q[:, None]) * a + q[:, None] * b
        return (y - pred).ravel()

    result = {}
    for model in ("H0", "Hdelta", "Hq", "Hboth"):
        count = 2 + int(model in {"Hdelta", "Hboth"}) + int(model in {"Hq", "Hboth"})
        fit = least_squares(lambda p: residual(p, model), np.zeros(count), max_nfev=3000)
        result[model] = float(np.sqrt(np.mean(residual(fit.x, model) ** 2)))
    return result
