"""Generic-difficulty control for human component interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .permutation_tests import exact_permutation_pvalue


def residualize_against_difficulty(component, difficulty):
    component = np.asarray(component, dtype=float)
    difficulty = np.asarray(difficulty, dtype=float)
    keep = np.isfinite(component) & np.isfinite(difficulty)
    result = np.full(len(component), np.nan)
    if keep.sum() < 2:
        return result
    x = np.column_stack([np.ones(keep.sum()), difficulty[keep]])
    beta = np.linalg.pinv(x) @ component[keep]
    result[keep] = component[keep] - x @ beta
    return result


def _profile(frame, column):
    return frame.groupby("transition_position")[column].mean().reindex(range(1, 8)).to_numpy()


def difficulty_residualized_results(loadings, features, mixed, novel, stability):
    targets = {
        "A_minus_B": features.set_index("transition_position").A_minus_B.reindex(range(1, 8)).to_numpy(),
        "switch_cost": features.set_index("transition_position").switch_cost.reindex(range(1, 8)).to_numpy(),
        "Mixed_midpoint_deviation": _profile(mixed, "midpoint_deviation"),
        "Novel_midpoint_deviation": _profile(novel, "midpoint_deviation"),
    }
    difficulty = features.set_index("transition_position").mean_pure_logRT.reindex(
        range(1, 8)
    ).to_numpy()
    stable_map = stability[stability.resample_type.eq("all")].set_index(
        "component"
    ).stable_component.to_dict()
    rows = []
    for component, group in loadings.groupby("component"):
        original = group.sort_values("transition_position").loading.to_numpy()
        residual = residualize_against_difficulty(original, difficulty)
        for position, value in enumerate(residual, start=1):
            rows.append({
                "row_type": "residual_loading", "component": component,
                "transition_position": position, "target": "component_loading",
                "original_loading": float(original[position - 1]),
                "difficulty_residualized_loading": float(value),
                "pearson_r": np.nan, "spearman_rho": np.nan,
                "permutation_p": np.nan,
                "component_stable": bool(stable_map.get(component, False)),
                "beyond_difficulty_supported": False,
            })
        for target, values in targets.items():
            keep = np.isfinite(residual) & np.isfinite(values)
            if keep.sum() >= 3 and np.std(residual[keep]) > 1e-12 and np.std(values[keep]) > 1e-12:
                pearson = float(pearsonr(residual[keep], values[keep]).statistic)
                spearman = float(spearmanr(residual[keep], values[keep]).statistic)
                p_value = exact_permutation_pvalue(residual[keep], values[keep])
            else:
                pearson = spearman = p_value = np.nan
            supported = bool(
                stable_map.get(component, False) and np.isfinite(p_value) and p_value < .05
            )
            rows.append({
                "row_type": "target_association", "component": component,
                "transition_position": np.nan, "target": target,
                "original_loading": np.nan,
                "difficulty_residualized_loading": np.nan,
                "pearson_r": pearson, "spearman_rho": spearman,
                "permutation_p": p_value,
                "component_stable": bool(stable_map.get(component, False)),
                "beyond_difficulty_supported": supported,
            })
    return pd.DataFrame(rows)
