from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phase1f_shared_backbone.analysis import (
    assert_canonical_metadata, frozen_runs, geometry_tables, hidden_patches,
    load_canonical, module_hash, task_patches,
)
from phase1f_shared_backbone.config import smoke_config
from phase1f_shared_backbone.math import (
    backbone, centered_logits, decompose, orthogonal_matched,
    prediction_metrics, wrong_positions,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_seed(tmp_path_factory):
    config = smoke_config(tmp_path_factory.mktemp("phase1f"))
    config = config.__class__(**{**config.__dict__,
        "checkpoint_dir": ROOT/"outputs/confirmatory_srn_final/confirmatory_checkpoints",
        "timeline_audit_dir": ROOT/"analysis/srn_training_timeline_audit"})
    model, policy, _, _, checkpoint_hash = load_canonical(config, "High_monotonic", 0)
    runs = frozen_runs(config, model)
    task, hidden, specificity, arrays = geometry_tables(
        config, "High_monotonic", 0, checkpoint_hash, runs)
    before = module_hash(model)
    task_patch = task_patches(config, "High_monotonic", 0, policy, checkpoint_hash, runs, arrays)
    hidden_patch = hidden_patches(
        config, "High_monotonic", 0, model, policy, checkpoint_hash, runs, arrays)
    return config, model, runs, task, hidden, specificity, task_patch, hidden_patch, before


def test_noncanonical_srn_fails_closed():
    with pytest.raises(AssertionError, match="FAIL: NONCANONICAL_SRN"):
        assert_canonical_metadata(6, 2., "High_monotonic")


def test_centered_logit_identity():
    values = centered_logits([[1., 2., 7., -3.], [5., 5., 5., 5.]])
    assert np.allclose(values.sum(axis=1), 0.)


def test_backbone_exactly_reconstructs_A_and_B():
    rng = np.random.default_rng(2); a = rng.normal(size=(7, 4)); b = rng.normal(size=(7, 4))
    shared, modulation = backbone(a, b)
    assert np.allclose(shared + modulation, a)
    assert np.allclose(shared - modulation, b)


def test_projection_residual_is_orthogonal():
    result = decompose([2., 3., -1.], [0., 1., 0.], [1., -2., 1.])
    assert abs(result["residual"] @ np.array([1., -2., 1.])) < 1e-8


def test_wrong_position_control_excludes_true_position():
    for position in range(7):
        wrong = wrong_positions(position)
        assert position not in wrong and len(wrong) == 6


def test_orthogonal_control_is_displacement_matched():
    correct = np.array([1., 2., -1., -2.])
    wrong = orthogonal_matched(correct)
    assert abs(np.linalg.norm(wrong)-np.linalg.norm(correct)) < 1e-8
    assert abs(wrong @ correct) < 1e-8


def test_all_causal_evaluations_are_frozen(real_seed):
    _, model, _, _, _, _, _, _, before = real_seed
    assert module_hash(model) == before


def test_raw_loss_matches_canonical_logits(real_seed):
    _, _, runs, _, _, _, _, _, _ = real_seed
    for run in runs.values():
        exported = prediction_metrics(run["logits"], run["targets"])["raw_loss"]
        assert np.isclose(exported, run["loss"].mean(), atol=1e-10)


def test_all_interventions_have_seven_transitions(real_seed):
    *_, task_patch, hidden_patch, _ = real_seed
    for frame in (task_patch, hidden_patch):
        counts = frame.groupby(["probe", "intervention"]).transition.nunique()
        assert counts.eq(7).all()


def test_probe_labels_remain_distinct(real_seed):
    _, _, _, task, hidden, specificity, task_patch, hidden_patch, _ = real_seed
    expected = {"Mixed", "Novel", "NoContextProbe"}
    for frame in (task, hidden, specificity, task_patch, hidden_patch):
        assert set(frame.probe.unique()) == expected
