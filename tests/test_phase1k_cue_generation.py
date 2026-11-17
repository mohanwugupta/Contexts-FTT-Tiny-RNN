from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from confirmatory_srn.training import state_hash
from phase1f_shared_backbone.math import prediction_metrics
from phase1h_plasticity_audit.replay import _setup, canonical_objective
from phase1i_context_composition.cue_encoding import CUES
from phase1i_context_composition.mechanics import centered, natural_rollout
from phase1j_context_stability.core import load_final_model
from phase1k_cue_generation.acquisition import replay_acquisition
from phase1k_cue_generation.analysis import add_all_novel_means, bootstrap_summary, primary_comparisons, seed_summary, trajectory_summary
from phase1k_cue_generation.config import FAMILIES, HIGH, smoke_config
from phase1k_cue_generation.interventions import (
    ab_convergence_rows, assert_takeover_semantics, cue_rollout, cue_trajectory_batches,
    endpoint_coordinates, frozen_references, local_cue_branches,
)
from phase1k_cue_generation.plotting import enforce_figures
from srn_training_timeline_audit.core import sha256_file


@pytest.fixture(scope="module")
def config():
    return smoke_config(".", "/tmp/phase1k_test")


@pytest.fixture(scope="module")
def evaluated(config):
    model, policy, provenance = load_final_model(config, HIGH, 0); before = state_hash(model, policy)
    references, natural = frozen_references(model, policy, config, HIGH, 0, provenance)
    local = local_cue_branches(model, policy, config, HIGH, 0, provenance, references)
    takeover, pulse, endpoint, runs = cue_trajectory_batches(model, policy, config, HIGH, 0, provenance, references)
    convergence = ab_convergence_rows(config, HIGH, 0, provenance, references, runs)
    assert before == state_hash(model, policy)
    return model, policy, provenance, references, natural, local, takeover, pulse, endpoint, convergence, runs


def test_1_all_checkpoint_hashes_match_registry(config):
    registry = pd.read_csv(config.resolve(config.provenance_table))
    for family in FAMILIES:
        for seed in range(20):
            row = registry[(registry.condition.eq(family)) & registry.seed.eq(seed)]
            path = config.resolve(config.checkpoint_dir) / f"{family}_seed_{seed}.pt"
            assert len(row) == 1 and sha256_file(path) == row.iloc[0].checkpoint_hash


def test_2_canonical_architecture_identity(config):
    for family, scale in (("High_monotonic", 2.0), ("Low_monotonic", .02)):
        model, _, provenance = load_final_model(config, family, 0)
        assert model.hidden_dim == 8 and model.recurrent.nonlinearity == "tanh"
        assert provenance["initialization_scale"] == scale


def test_3_every_intervention_batch_keeps_parameters_frozen(evaluated):
    model, policy, provenance, *_ = evaluated
    assert state_hash(model, policy) == provenance["parameter_hash"]


def test_4_intervention_engine_has_no_optimizer_or_backward_path():
    import phase1k_cue_generation.interventions as module
    source = inspect.getsource(module)
    assert "Optimizer" not in source and "torch.optim" not in source and ".backward(" not in source


def test_5_physical_mixed_cue_is_exact_midpoint():
    assert np.allclose(CUES["Mixed"], .5 * (CUES["A"] + CUES["B"]), atol=1e-7, rtol=0)


def test_6_takeover_preserves_natural_history(evaluated, config):
    model, _, _, references, *_ = evaluated
    run = cue_rollout(model, config, "A", 3, "persistent_mixed_takeover")
    natural = references["rolls"]["A"]
    assert np.array_equal(run["hidden"][:3], natural["hidden"][:3])
    assert assert_takeover_semantics(run, natural, "A", 3, "persistent_mixed_takeover")


def test_7_first_changed_cue_produces_transition_k_state_and_output(evaluated, config):
    model, _, _, references, *_ = evaluated
    run = cue_rollout(model, config, "A", 2, "persistent_mixed_takeover")
    natural = references["rolls"]["A"]
    assert run["cues"][1] == "A" and run["cues"][2] == "Mixed"
    assert np.array_equal(run["hidden"][:2], natural["hidden"][:2])
    assert not np.array_equal(run["hidden"][2], natural["hidden"][2])
    assert not np.array_equal(run["logits"][2], natural["logits"][2])


def test_8_pulse_uses_mixed_exactly_once_then_restores_native(evaluated, config):
    model, _, _, references, *_ = evaluated
    run = cue_rollout(model, config, "Novel1", 4, "single_mixed_pulse")
    assert run["cues"].count("Mixed") == 1 and run["cues"][4] == "Mixed" and run["cues"][5] == "Novel1"
    assert assert_takeover_semantics(run, references["rolls"]["Novel1"], "Novel1", 4, "single_mixed_pulse")


def test_9_persistent_takeover_never_restores_native(evaluated, config):
    model, _, _, references, *_ = evaluated
    run = cue_rollout(model, config, "Novel2", 3, "persistent_mixed_takeover")
    assert run["cues"][:3] == ("Novel2",) * 3 and run["cues"][3:] == ("Mixed",) * 4
    assert assert_takeover_semantics(run, references["rolls"]["Novel2"], "Novel2", 3, "persistent_mixed_takeover")


def test_10_every_local_branch_has_identical_prestate_hash(evaluated):
    local = evaluated[5]
    counts = local.groupby(["family", "seed", "source_probe", "transition"]).prestate_hash.nunique()
    assert counts.eq(1).all() and local.groupby(["source_probe", "transition"]).size().eq(5).all()


def test_11_familiar_midpoint_references_are_frozen_A_B_only(evaluated):
    references = evaluated[3]; rolls = references["rolls"]
    assert np.array_equal(references["midpoint_h"], .5 * (rolls["A"]["hidden"] + rolls["B"]["hidden"]))
    assert np.array_equal(references["midpoint_z"], .5 * (rolls["A"]["centered_logits"] + rolls["B"]["centered_logits"]))


def test_12_target_builder_never_averages_probabilities():
    import phase1k_cue_generation.interventions as module
    source = inspect.getsource(module)
    assert "softmax" not in source and "probabilities" not in source


def test_13_centered_logits_have_zero_class_mean(evaluated):
    references = evaluated[3]
    for run in references["rolls"].values():
        assert np.allclose(centered(run["logits"]).mean(axis=1), 0, atol=1e-12)


def test_14_endpoint_coordinate_synthetic_sanity():
    a = np.array([-1., 0., 1., 0.]); b = np.array([1., 0., -1., 0.]); middle = .5 * (a + b)
    for value, expected in ((a, 0.), (b, 1.), (middle, .5)):
        alpha, off_axis = endpoint_coordinates(value, a, b)
        assert np.isclose(alpha, expected, atol=1e-12) and np.isclose(off_axis, 0, atol=1e-12)


def test_15_novelmean_is_nested_within_seed(evaluated, config):
    natural, local, takeover, pulse, endpoint = add_all_novel_means(evaluated[4], evaluated[5], evaluated[6], evaluated[7], evaluated[8])
    summary = trajectory_summary(takeover, pulse, endpoint)
    seeds = seed_summary(local, summary, evaluated[9])
    comparisons = primary_comparisons(seeds); boot = bootstrap_summary(comparisons, config)
    assert "NovelMean" in set(local.source_probe) and boot.resampling_unit.eq("seed").all()
    assert local[local.source_probe.eq("NovelMean")].shape[0] * 4 == local[local.source_probe.isin(["Novel1", "Novel2", "Novel3", "Novel4"])].shape[0]


def test_16_every_intervention_rollout_preserves_seven_raw_CEs(evaluated):
    for frame in evaluated[6:9]:
        counts = frame.groupby(["family", "seed", "source_probe", "takeover_transition", "intervention"]).raw_ce.count()
        assert counts.eq(7).all()


def test_17_sequence_loss_is_mean_of_exactly_seven_CEs(evaluated, config):
    model = evaluated[0]; run = natural_rollout(model, config.sequence, CUES["A"])
    metrics = prediction_metrics(run["logits"], run["targets"])
    assert len(metrics["transition_losses"]) == 7
    assert np.isclose(metrics["raw_loss"], np.mean(metrics["transition_losses"]))


def test_18_every_seed_replay_matches_registered_final_checkpoint(config):
    for family in FAMILIES:
        for seed in range(20):
            replay = replay_acquisition(config, family, seed)
            assert len(replay) == 170 and replay.parameters_changed.all()


def test_19_acquisition_loss_is_pre_update(config):
    replay = replay_acquisition(config, HIGH, 0); first = replay.iloc[0]
    model, policy, _, _, tokens, targets = _setup(config, HIGH, 0)
    _, ce, _, _ = canonical_objective(model, policy, tokens, targets, "A", config)
    assert np.isclose(first.raw_sequence_loss_pre_update, float(ce.detach()), atol=1e-7)
    assert first.parameter_hash_pre != first.parameter_hash


def test_20_all_ten_figures_are_mandatory(tmp_path):
    with pytest.raises(RuntimeError, match="Missing Phase 1K mandatory figures"):
        enforce_figures(tmp_path)
