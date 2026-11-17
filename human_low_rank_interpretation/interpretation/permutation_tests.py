"""Exact transition-label permutation inference for semantic associations."""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


NUMERIC_FEATURES = (
    "transition_position", "mean_pure_logRT", "A_minus_B", "error_rate",
    "early_learning_slope", "late_learning_slope", "switch_cost",
    "post_switch_recovery", "distance_since_current_key",
    "distance_since_next_key", "second_half_indicator", "initial_log_rt",
    "asymptotic_log_rt",
)
CATEGORICAL_FEATURES = ("current_key", "next_key", "cyclic_boundary")


def _pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.std(x[keep]) < 1e-12 or np.std(y[keep]) < 1e-12:
        return np.nan
    return float(pearsonr(x[keep], y[keep]).statistic)


@lru_cache(maxsize=8)
def _permutation_indices(size):
    return np.asarray(list(itertools.permutations(range(int(size)))), dtype=int)


def _numeric_null(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x - x.mean(); y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator < 1e-14:
        return np.array([])
    return np.abs(y[_permutation_indices(len(y))] @ x / denominator)


def exact_permutation_pvalue(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    observed = abs(_pearson(x, y))
    if len(x) < 3 or not np.isfinite(observed):
        return np.nan
    null = _numeric_null(x, y)
    return float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))


def _loto_rmse_numeric(x, y):
    errors = []
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    for held in range(len(x)):
        train = np.arange(len(x)) != held
        design = np.column_stack([np.ones(train.sum()), x[train]])
        beta = np.linalg.pinv(design) @ y[train]
        errors.append((y[held] - np.array([1, x[held]]) @ beta) ** 2)
    return float(np.sqrt(np.mean(errors))) if errors else np.nan


def _eta_squared(categories, values):
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)
    keep = pd.notna(categories) & np.isfinite(values)
    categories, values = categories[keep], values[keep]
    if len(values) < 3 or len(np.unique(categories)) < 2 or np.var(values) < 1e-14:
        return np.nan
    grand = values.mean()
    between = sum(
        len(values[categories == level]) * (values[categories == level].mean() - grand) ** 2
        for level in np.unique(categories)
    )
    total = float(np.sum((values - grand) ** 2))
    return float(between / total) if total > 1e-14 else np.nan


def _loto_rmse_categorical(categories, values):
    categories = np.asarray(categories)
    values = np.asarray(values, dtype=float)
    errors = []
    for held in range(len(values)):
        train = np.arange(len(values)) != held
        same = train & (categories == categories[held])
        prediction = values[same].mean() if same.any() else values[train].mean()
        errors.append((values[held] - prediction) ** 2)
    return float(np.sqrt(np.mean(errors)))


def _bootstrap_ci(bootstrap, component, feature_values, categorical):
    statistics = []
    subset = bootstrap[bootstrap.component.eq(component)]
    for _, group in subset.groupby("bootstrap"):
        loading = group.sort_values("transition_position").aligned_loading.to_numpy()
        value = (
            _eta_squared(feature_values, loading)
            if categorical else _pearson(feature_values, loading)
        )
        if np.isfinite(value):
            statistics.append(value)
    if not statistics:
        return np.nan, np.nan
    return float(np.quantile(statistics, .025)), float(np.quantile(statistics, .975))


def component_feature_associations(loadings, stability, features, bootstrap):
    stable = stability[
        stability.resample_type.eq("all") & stability.stable_component.astype(bool)
    ].set_index("component")
    feature_table = features.sort_values("transition_position")
    associations, permutations = [], []
    for component, component_rows in loadings.groupby("component", sort=True):
        loading = component_rows.sort_values("transition_position").loading.to_numpy(dtype=float)
        component_stable = int(component) in stable.index
        for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            values = feature_table[feature].to_numpy()
            categorical = feature in CATEGORICAL_FEATURES
            estimable = (
                pd.Series(values).notna().sum() >= 3
                and pd.Series(values).dropna().nunique() >= 2
            )
            if not estimable:
                associations.append({
                    "component": component, "feature": feature,
                    "feature_kind": "categorical" if categorical else "numeric",
                    "feature_estimable": False, "pearson_r": np.nan,
                    "spearman_rho": np.nan, "association_statistic": np.nan,
                    "loto_rmse": np.nan, "bootstrap_ci_lower": np.nan,
                    "bootstrap_ci_upper": np.nan, "permutation_p": np.nan,
                    "permutation_count": 0, "permutation_mode": "not_estimable",
                    "component_stable": component_stable,
                    "semantic_association_supported": False,
                })
                continue
            complete = pd.notna(values) & np.isfinite(loading)
            complete_values = values[complete]
            complete_loading = loading[complete]
            if categorical:
                observed = _eta_squared(complete_values, complete_loading)
                pearson = spearman = np.nan
                loto = _loto_rmse_categorical(complete_values, complete_loading)
            else:
                observed = abs(_pearson(complete_values, complete_loading))
                pearson = _pearson(complete_values, complete_loading)
                spearman = float(spearmanr(complete_values, complete_loading).statistic)
                loto = _loto_rmse_numeric(complete_values, complete_loading)
            indices = _permutation_indices(len(complete_values))
            if categorical:
                null = np.asarray([
                    _eta_squared(complete_values[index], complete_loading) for index in indices
                ])
            else:
                null = _numeric_null(complete_loading, complete_values)
            for index, statistic in enumerate(null):
                permutations.append({
                    "component": component, "feature": feature,
                    "permutation": index, "null_statistic": float(statistic),
                    "observed_statistic": observed,
                })
            null = np.asarray(null, dtype=float)
            p_value = float((np.sum(null >= observed - 1e-12) + 1) / (len(null) + 1))
            lower, upper = _bootstrap_ci(bootstrap, component, values, categorical)
            supported = bool(
                component_stable and p_value < .05 and
                np.isfinite(lower) and np.isfinite(upper) and (lower > 0 or upper < 0)
            )
            associations.append({
                "component": component, "feature": feature,
                "feature_kind": "categorical" if categorical else "numeric",
                "feature_estimable": True, "pearson_r": pearson,
                "spearman_rho": spearman, "association_statistic": observed,
                "loto_rmse": loto, "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper, "permutation_p": p_value,
                "permutation_count": len(null),
                "permutation_mode": f"exact_{len(complete_values)}_factorial",
                "component_stable": component_stable,
                "semantic_association_supported": supported,
            })
    return pd.DataFrame(associations), pd.DataFrame(permutations)
