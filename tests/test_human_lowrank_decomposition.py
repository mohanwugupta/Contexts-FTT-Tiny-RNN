import numpy as np
import pandas as pd
import pytest

from human_low_rank_interpretation.decomposition.align_components import align_components
from human_low_rank_interpretation.decomposition.estimate_basis import (
    assert_pure_profile_contract,
    fit_pca_basis,
)
from human_low_rank_interpretation.decomposition.participant_scores import participant_folds
from human_low_rank_interpretation.replication.cross_dataset_basis import (
    fit_independent_bases,
)


def _profiles():
    rows = []
    for dataset in ("Combined", "Novel"):
        for participant_index in range(8):
            for position in range(1, 8):
                rows.append({
                    "participant_uid": f"{dataset}:p{participant_index}",
                    "dataset": dataset,
                    "transition_position": position,
                    "delta": (
                        (participant_index - 3.5) * (position - 4) * 0.02
                        + np.sin(position) * 0.1
                    ),
                    "midpoint": 6 + position * 0.01,
                    "pure_source_contexts": "A|B",
                    "probe_rows_used_for_profile": 0,
                })
    return pd.DataFrame(rows)


def test_pure_basis_contract_rejects_probe_contamination():
    profiles = _profiles()
    assert_pure_profile_contract(profiles)
    contaminated = profiles.copy()
    contaminated.loc[0, "probe_rows_used_for_profile"] = 1
    with pytest.raises(ValueError, match="probe"):
        assert_pure_profile_contract(contaminated)


def test_participant_folds_have_no_participant_leakage():
    profiles = _profiles().query("dataset == 'Combined'")
    for fold, training, heldout in participant_folds(profiles, n_folds=4):
        assert set(training.participant_uid).isdisjoint(heldout.participant_uid)
        assert fold in {0, 1, 2, 3}


def test_sign_and_order_changes_are_aligned_by_hungarian_assignment():
    reference = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    candidate = reference[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    aligned, audit = align_components(reference, candidate)
    np.testing.assert_allclose(aligned, reference)
    assert np.allclose(audit.matched_cosine, 1.0)


def test_full_rank_pca_reconstructs_original_profiles():
    profiles = _profiles().query("dataset == 'Combined'")
    basis = fit_pca_basis(profiles, rank=7)
    matrix = profiles.pivot(
        index="participant_uid", columns="transition_position", values="delta"
    ).sort_index(axis=1)
    reconstructed = basis.reconstruct(matrix.to_numpy())
    np.testing.assert_allclose(reconstructed, matrix.to_numpy(), atol=1e-10)


def test_cross_dataset_bases_are_fit_independently_before_alignment():
    result = fit_independent_bases(_profiles(), rank=2)
    assert set(result) == {"Combined", "Novel"}
    assert result["Combined"].training_datasets == ("Combined",)
    assert result["Novel"].training_datasets == ("Novel",)

