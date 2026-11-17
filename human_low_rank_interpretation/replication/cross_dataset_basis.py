"""Independent Combined/Novel bases and post-fit subspace alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles

from ..decomposition.align_components import align_components
from ..decomposition.estimate_basis import fit_pca_basis


def fit_independent_bases(profiles: pd.DataFrame, *, rank: int):
    return {
        dataset: fit_pca_basis(frame, rank=rank)
        for dataset, frame in profiles.groupby("dataset", sort=False)
    }


def compare_cross_dataset_bases(profiles: pd.DataFrame, *, rank: int):
    bases = fit_independent_bases(profiles, rank=rank)
    combined, novel = bases["Combined"], bases["Novel"]
    aligned_novel, audit = align_components(combined.components, novel.components)
    rows = []
    for component in range(rank):
        for position in range(7):
            rows.append({
                "component": component + 1, "transition_position": position + 1,
                "combined_loading": float(combined.components[position, component]),
                "novel_loading_unaligned": float(novel.components[position, component]),
                "novel_loading_aligned": float(aligned_novel[position, component]),
                "matched_novel_component": int(audit.candidate_component[component]),
                "matched_cosine": float(audit.matched_cosine[component]),
                "sign_flipped": bool(audit.sign_flipped[component]),
                "combined_training_datasets": "|".join(combined.training_datasets),
                "novel_training_datasets": "|".join(novel.training_datasets),
                "independent_before_alignment": True,
            })
    angles = np.degrees(subspace_angles(combined.components, novel.components))
    projection_fraction = float(
        np.linalg.norm(combined.components.T @ novel.components, "fro") ** 2 / rank
    )
    summary = pd.DataFrame([{
        "rank": rank,
        "mean_principal_angle_degrees": float(np.mean(angles)),
        "maximum_principal_angle_degrees": float(np.max(angles)),
        "minimum_principal_angle_degrees": float(np.min(angles)),
        "principal_angles_degrees": "|".join(f"{value:.10g}" for value in angles),
        "mean_matched_component_cosine": float(np.mean(audit.matched_cosine)),
        "minimum_matched_component_cosine": float(np.min(audit.matched_cosine)),
        "shared_subspace_variance_fraction": projection_fraction,
        "combined_participants": len(combined.training_participants),
        "novel_participants": len(novel.training_participants),
        "bases_fit_independently_before_alignment": True,
    }])
    return bases, pd.DataFrame(rows), summary
