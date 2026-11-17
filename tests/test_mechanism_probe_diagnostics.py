"""PRD mechanism-first diagnostics T1--T14 contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mechanism_probe_diagnostics.analysis import (
    bootstrap_population, canonical_condition, canonicalize_whole_profiles,
    decompose_profiles, deterministic_fold, fit_probe_q_delta,
    score_prediction_rows,
)
from mechanism_probe_diagnostics.data import sha256_file
from mechanism_probe_diagnostics.default_basis import estimate_independent_default_basis
from mechanism_probe_diagnostics.history import derive_sequence_history, synthetic_history_separation


def _profiles(q=.3, delta=10.0, offset=0.0, weak=False):
    a = np.array([1., -1., .5, -.5, .8, -.8, 0.]) + 100
    b = a.copy() if weak else np.array([-.5, .5, -1., 1., -.2, .2, 0.]) + 120
    p = delta + (1 - q) * a + q * b + offset
    rows = []
    for condition, values in (("A", a), ("B", b), ("Mixed", p)):
        for k, value in enumerate(values, 1):
            rows.append({"unit_id": "u1", "condition": condition,
                         "transition_position": k, "M": value})
    return pd.DataFrame(rows)


def test_t1_condition_aliases_are_deterministic():
    assert canonical_condition("Combined") == "Mixed"
    assert canonical_condition("Both") == "Mixed"
    assert canonical_condition("Third") == "Novel"
    assert canonical_condition("Novel") == "Novel"
    with pytest.raises(ValueError):
        canonical_condition("mystery")


def test_t2_exact_decomposition():
    result = decompose_profiles(_profiles())
    assert np.allclose(result.M, result.L + result.Z)
    assert np.allclose(result.groupby(["unit_id", "condition"]).Z.sum(), 0)


def test_t3_offset_invariance():
    fit = fit_probe_q_delta(decompose_profiles(_profiles(offset=0)), probe="Mixed").iloc[0]
    shifted = fit_probe_q_delta(decompose_profiles(_profiles(offset=17)), probe="Mixed").iloc[0]
    assert shifted.q_unconstrained == pytest.approx(fit.q_unconstrained)
    assert shifted.delta - fit.delta == pytest.approx(17)


def test_t4_synthetic_q_recovery_noiseless_and_noisy():
    fit = fit_probe_q_delta(decompose_profiles(_profiles(q=.72, delta=4)), probe="Mixed").iloc[0]
    assert fit.q_unconstrained == pytest.approx(.72)
    noisy = _profiles(q=.72, delta=4)
    mask = noisy.condition.eq("Mixed")
    noisy.loc[mask, "M"] += np.array([-.01, .01, 0, .005, -.005, .01, -.01])
    recovered = fit_probe_q_delta(decompose_profiles(noisy), probe="Mixed").iloc[0]
    assert recovered.q_unconstrained == pytest.approx(.72, abs=.02)


def test_t5_synthetic_delta_recovery_without_q_bias():
    fit = fit_probe_q_delta(decompose_profiles(_profiles(q=.2, delta=35)), probe="Mixed").iloc[0]
    assert fit.q_unconstrained == pytest.approx(.2)
    assert fit.delta == pytest.approx(35)


def test_t6_weak_contrast_is_flagged_and_finite_diagnostic():
    fit = fit_probe_q_delta(
        decompose_profiles(_profiles(weak=True)), probe="Mixed", weak_epsilon=1e-4
    ).iloc[0]
    assert bool(fit.weak_ab_contrast)
    assert np.isnan(fit.q_unconstrained)
    assert np.isfinite(fit.delta)


def test_t7_whole_profile_canonicalization_never_splits_transitions():
    source = _profiles()[lambda x: x.condition.isin(["A", "B"])]
    canonical, mapping = canonicalize_whole_profiles(source)
    assert set(canonical.condition) == {"Fast", "Slow"}
    original_for_fast = mapping.fast_original_condition.iloc[0]
    assert set(canonical.loc[canonical.condition.eq("Fast"), "original_condition"]) == {original_for_fast}
    assert not bool(mapping.canonicalization_used_probe_rows.iloc[0])


def test_t8_fold_firewall_fields_and_heldout_rows_are_preserved():
    rows = pd.DataFrame({
        "model": ["m"] * 14, "dataset": ["d"] * 14, "probe_context": ["Both"] * 14,
        "participant_uid": ["p"] * 14, "probe_epoch": ["e"] * 14,
        "sequence_repetition": [2] * 7 + [4] * 7,
        "transition_position": list(range(1, 8)) * 2,
        "actual": np.arange(14.), "predicted": np.arange(14.),
        "heldout_participant_in_basis": [False] * 14,
        "probe_rows_used_for_basis": [0] * 14,
    })
    assert not rows.heldout_participant_in_basis.any()
    assert rows.probe_rows_used_for_basis.sum() == 0
    _, _, scored = score_prediction_rows(
        rows, actual_col="actual", predicted_col="predicted",
        profile_cols=["participant_uid", "probe_epoch", "sequence_repetition"],
        group_cols=["model"],
    )
    assert len(scored) == len(rows)


def test_t9_centering_removes_constant_offset_only():
    rows = pd.DataFrame({
        "participant_uid": ["p"] * 7, "condition": ["Mixed"] * 7,
        "transition_position": range(1, 8), "actual": np.arange(7.),
        "predicted": np.arange(7.) + 5,
    })
    raw, centered, _ = score_prediction_rows(
        rows, actual_col="actual", predicted_col="predicted",
        profile_cols=["participant_uid", "condition"], group_cols=["condition"],
    )
    assert raw.heldout_rmse.iloc[0] == pytest.approx(5)
    assert centered.heldout_rmse.iloc[0] == pytest.approx(0)


def test_t10_history_derivation_on_toy_sequence():
    toy = pd.DataFrame({
        "participant_uid": ["p"] * 5,
        "condition": ["A", "A", "Mixed", "B", "Mixed"],
        "block_order": [1, 1, 2, 3, 4], "sequence_repetition": [1, 2, 1, 1, 1],
        "probe_order": [np.nan, np.nan, 1, np.nan, 2],
    })
    out = derive_sequence_history(toy)
    assert out.immediately_preceding_context.tolist() == [None, "A", "A", "Mixed", "B"]
    assert out.most_recent_unambiguous_ab.tolist() == [None, "A", "A", "A", "B"]
    assert out.loc[4, "sequences_since_last_a"] == 3
    assert out.loc[4, "time_since_last_ab"] == 1
    assert out.loc[4, "test_condition_order"] == 2


def test_t11_synthetic_history_delta_q_separation():
    rng = np.random.default_rng(4)
    a = np.array([-.7, -.4, -.1, 0, .2, .4, .6])
    b = np.array([.6, .2, 0, -.2, -.4, -.1, .3])
    h = np.linspace(-1, 1, 30)
    q = .4
    delta_only = 2 + .8 * h[:, None] + (1 - q) * a + q * b + rng.normal(0, .002, (30, 7))
    scores_delta = synthetic_history_separation(a, b, h, delta_only)
    assert scores_delta["Hdelta"] < scores_delta["Hq"]
    q_t = 1 / (1 + np.exp(-(-.2 + 1.4 * h)))
    q_only = 2 + (1 - q_t[:, None]) * a + q_t[:, None] * b + rng.normal(0, .002, (30, 7))
    scores_q = synthetic_history_separation(a, b, h, q_only)
    assert scores_q["Hq"] < scores_q["Hdelta"]


def test_t12_default_source_firewall_rejects_novel_rows():
    rows = []
    for participant in ("d1", "d2", "d3", "d4"):
        for k in range(1, 8):
            rows.append({"participant_uid": participant, "dataset": "Novel",
                         "condition": "Default", "original_condition": "Default",
                         "transition_position": k, "mean_rt": 100 + k})
    basis, audit = estimate_independent_default_basis(pd.DataFrame(rows))
    assert audit["novel_rows_used"] == 0 and audit["ab_probe_rows_used"] == 0
    contaminated = pd.DataFrame(rows)
    contaminated.loc[0, "condition"] = "Novel"
    with pytest.raises(ValueError):
        estimate_independent_default_basis(contaminated)
    assert abs(basis.Z.sum()) < 1e-10


def test_t13_deterministic_resampling_and_folds():
    units = pd.DataFrame({
        "unit_id": [f"u{i}" for i in range(8)], "condition": ["Mixed"] * 8,
        "system": ["Human"] * 8, "delta": np.arange(8.),
    })
    s1, d1 = bootstrap_population(units, group_cols=["system", "condition"],
                                  metrics=["delta"], draws=30, seed=9)
    s2, d2 = bootstrap_population(units, group_cols=["system", "condition"],
                                  metrics=["delta"], draws=30, seed=9)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(d1, d2)
    assert deterministic_fold("participant", 5, 20) == deterministic_fold("participant", 5, 20)


def test_t14_source_hash_is_unchanged_by_read_only_analysis(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    before = sha256_file(source)
    pd.read_csv(source).assign(c=3)
    after = sha256_file(source)
    assert before == after
