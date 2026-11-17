"""Read-only human/SRN comparison at dimensionality and subspace level."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

from ..decomposition.estimate_basis import LowRankBasis


def _effective_dimension(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2)) if np.sum(
        eigenvalues ** 2
    ) > 1e-20 else 0.0


def compare_human_srn_dimensionality(human_basis: LowRankBasis, srn_profile: pd.DataFrame):
    before = srn_profile.copy(deep=True)
    seed_rows = srn_profile[srn_profile.row_type.eq("seed")].copy()
    seed_rows["srn_delta"] = seed_rows.a_rt - seed_rows.b_rt
    matrix = seed_rows.pivot_table(
        index="seed", columns="transition_position", values="srn_delta", aggfunc="first"
    ).reindex(columns=range(1, 8)).dropna()
    values = matrix.to_numpy(dtype=float)
    centered = values - values.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = singular ** 2
    if energy.sum() > 1e-20:
        variance = energy / energy.sum()
    else:
        variance = np.zeros_like(energy)
        variance[0] = 1.0
    srn_rank90 = int(np.searchsorted(np.cumsum(variance), .90) + 1)
    human_variance = human_basis.variance_explained
    human_rank90 = int(np.searchsorted(np.cumsum(human_variance), .90) + 1)
    dimensionality = pd.DataFrame([
        {
            "system": "Human", "rank_90pct": human_rank90,
            "effective_dimensionality": _effective_dimension(human_variance),
            "observations": len(human_basis.training_participants),
            "transition_count": 7, "srn_profile_frozen": True,
            "human_outcomes_used_to_modify_srn": 0,
        },
        {
            "system": "SRN", "rank_90pct": srn_rank90,
            "effective_dimensionality": _effective_dimension(variance),
            "observations": len(matrix), "transition_count": 7,
            "srn_profile_frozen": True, "human_outcomes_used_to_modify_srn": 0,
        },
    ])
    shared_specific = pd.DataFrame([
        {
            "system": "Human",
            "shared_pc1_variance_fraction": float(human_variance[0]),
            "state_specific_residual_variance_fraction": float(1 - human_variance[0]),
            "variance_partition_definition": "PC1_shared_vs_higher_components_state_specific",
        },
        {
            "system": "SRN", "shared_pc1_variance_fraction": float(variance[0]),
            "state_specific_residual_variance_fraction": float(1 - variance[0]),
            "variance_partition_definition": "PC1_shared_vs_higher_components_state_specific",
        },
    ])
    compare_rank = min(human_basis.rank, srn_rank90, vt.shape[0])
    srn_basis = vt[:compare_rank].T
    human_compare = human_basis.components[:, :compare_rank]
    angles = np.degrees(subspace_angles(human_compare, srn_basis))
    canonical = np.linalg.svd(human_compare.T @ srn_basis, compute_uv=False)
    similarity = pd.DataFrame([{
        "human_rank": human_basis.rank, "srn_rank_90pct": srn_rank90,
        "comparison_rank": compare_rank,
        "mean_principal_angle_degrees": float(np.mean(angles)),
        "maximum_principal_angle_degrees": float(np.max(angles)),
        "principal_angles_degrees": "|".join(f"{value:.10g}" for value in angles),
        "mean_canonical_correlation": float(np.mean(canonical)),
        "minimum_canonical_correlation": float(np.min(canonical)),
        "variance_partition_difference": float(abs(human_variance[0] - variance[0])),
        "exact_token_profile_was_tuned": False,
        "srn_profile_frozen": True,
        "human_outcomes_used_to_modify_srn": 0,
    }])
    pd.testing.assert_frame_equal(srn_profile, before)
    return dimensionality, shared_specific, similarity
