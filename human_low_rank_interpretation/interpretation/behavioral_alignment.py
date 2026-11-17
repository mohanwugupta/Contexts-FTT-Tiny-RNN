"""Project switch and learning profiles into the fixed human A/B basis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..decomposition.estimate_basis import LowRankBasis
from .permutation_tests import exact_permutation_pvalue


def switch_cost_alignment(switch_profiles: pd.DataFrame, basis: LowRankBasis):
    rows = []
    for participant, frame in switch_profiles.groupby("participant_uid", sort=False):
        profile = frame.set_index("transition_position").switch_cost.reindex(range(1, 8))
        keep = profile.notna().to_numpy()
        coefficient = np.linalg.pinv(basis.components[keep]) @ profile.to_numpy(dtype=float)[keep]
        for component, value in enumerate(coefficient, start=1):
            rows.append({
                "row_type": "participant_projection", "participant_uid": participant,
                "dataset": str(frame.dataset.iloc[0]), "component": component,
                "coefficient": float(value), "pearson_r": np.nan,
                "permutation_p": np.nan, "observed_transitions": int(keep.sum()),
                "basis_source": "pure_A_B_only", "basis_updated_from_switch": False,
            })
    group = switch_profiles.groupby("transition_position").switch_cost.mean().reindex(range(1, 8))
    for component in range(basis.rank):
        loading = basis.components[:, component]
        correlation = np.corrcoef(loading, group.to_numpy(dtype=float))[0, 1]
        rows.append({
            "row_type": "group_association", "participant_uid": np.nan,
            "dataset": "Pooled", "component": component + 1,
            "coefficient": float((np.linalg.pinv(basis.components) @ group.to_numpy())[component]),
            "pearson_r": float(correlation),
            "permutation_p": exact_permutation_pvalue(loading, group.to_numpy()),
            "observed_transitions": int(group.notna().sum()),
            "basis_source": "pure_A_B_only", "basis_updated_from_switch": False,
        })
    return pd.DataFrame(rows)


def learning_alignment(associations: pd.DataFrame):
    learning = {
        "early_learning_slope", "late_learning_slope", "initial_log_rt",
        "asymptotic_log_rt", "post_switch_recovery",
    }
    result = associations[associations.feature.isin(learning)].copy()
    result["row_type"] = "component_feature_association"
    result["basis_source"] = "pure_A_B_only"
    result["basis_updated_from_learning"] = False
    return result
