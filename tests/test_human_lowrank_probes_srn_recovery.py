import numpy as np
import pandas as pd

from human_low_rank_interpretation.decomposition.estimate_basis import fit_pca_basis
from human_low_rank_interpretation.probes.shared_basis import project_probe_coefficients
from human_low_rank_interpretation.recovery.recover_semantics import (
    apply_semantic_recovery_gate,
)
from human_low_rank_interpretation.srn.dimensionality_comparison import (
    compare_human_srn_dimensionality,
)


def _profiles():
    rows = []
    for participant_index in range(10):
        for position in range(1, 8):
            rows.append({
                "participant_uid": f"p{participant_index}",
                "dataset": "Combined",
                "transition_position": position,
                "delta": np.sin(position) + participant_index * .02 * (position - 4),
                "midpoint": 6.0,
                "pure_source_contexts": "A|B",
                "probe_rows_used_for_profile": 0,
            })
    return pd.DataFrame(rows)


def test_probe_projection_estimates_coefficients_without_changing_basis():
    basis = fit_pca_basis(_profiles(), rank=2)
    before = basis.components.copy()
    probe = pd.DataFrame({
        "participant_uid": np.repeat(["p0", "p1"], 7),
        "transition_position": np.tile(np.arange(1, 8), 2),
        "midpoint_deviation": np.tile(basis.components[:, 0], 2),
    })
    coefficients = project_probe_coefficients(probe, basis)
    np.testing.assert_allclose(basis.components, before)
    assert coefficients.basis_updated_from_probe.eq(False).all()
    assert coefficients.probe_rows_used_for_basis.eq(0).all()


def test_human_srn_comparison_is_read_only_and_audited():
    basis = fit_pca_basis(_profiles(), rank=2)
    srn = pd.DataFrame([
        {"seed": seed, "transition_position": position,
         "a_rt": .3 + .01 * position, "b_rt": .7 - .01 * position,
         "row_type": "seed"}
        for seed in range(3) for position in range(1, 8)
    ])
    before = srn.copy(deep=True)
    dimensionality, _variance, _similarity = compare_human_srn_dimensionality(
        basis, srn
    )
    pd.testing.assert_frame_equal(srn, before)
    assert dimensionality.human_outcomes_used_to_modify_srn.eq(0).all()
    assert dimensionality.srn_profile_frozen.eq(True).all()


def test_semantic_labels_are_blocked_when_recovery_is_inadequate():
    recovery = pd.DataFrame({
        "generating_semantic": ["difficulty", "context_sensitivity", "random"],
        "recovered_semantic": ["difficulty", "context_sensitivity", "difficulty"],
        "recovery_probability": [.45, .40, .12],
    })
    gate = apply_semantic_recovery_gate(
        recovery, minimum_recovery=.60, maximum_random_false_label=.05
    )
    assert not gate.semantic_labels_allowed
    assert "recovery" in gate.reason.lower() or "false" in gate.reason.lower()
