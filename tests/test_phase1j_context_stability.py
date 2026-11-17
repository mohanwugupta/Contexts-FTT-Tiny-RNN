from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from confirmatory_srn.training import state_hash
from phase1f_shared_backbone.math import prediction_metrics
from phase1h_plasticity_audit.replay import assert_canonical_payload
from phase1i_context_composition.cue_encoding import CUES, PROBES
from phase1i_context_composition.mechanics import derangement, natural_rollout, orthogonal_matched
from phase1j_context_stability.analysis import bootstrap_summary
from phase1j_context_stability.config import FAMILIES, HIGH, smoke_config
from phase1j_context_stability.core import (
    CLAMPS,
    add_novel_mean,
    all_intervention_rows,
    assert_mixed_cue,
    intervention_rollout,
    load_final_model,
    midpoint_targets,
    natural_reference_rows,
    replay_acquisition,
)
from phase1j_context_stability.plotting import enforce_figures


@pytest.fixture(scope="module")
def config():
    return smoke_config(".", "/tmp/phase1j_test")


@pytest.fixture(scope="module")
def frozen(config):
    model, policy, provenance = load_final_model(config, HIGH, 0)
    parameter_hash = provenance["parameter_hash"]
    natural, rolls, midpoint, midpoint_task = natural_reference_rows(
        model, policy, config, HIGH, 0, provenance["source_checkpoint_hash"], parameter_hash
    )
    interventions = all_intervention_rows(
        model, policy, config, HIGH, 0, provenance["source_checkpoint_hash"],
        parameter_hash, rolls, midpoint, midpoint_task,
    )
    return model, policy, provenance, natural, interventions, rolls, midpoint, midpoint_task


def test_1_every_checkpoint_has_canonical_architecture(config):
    for family in FAMILIES:
        for seed in range(20):
            path = config.resolve(config.checkpoint_dir) / f"{family}_seed_{seed}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert assert_canonical_payload(payload, family, config)
    model, _, _ = load_final_model(config, HIGH, 0)
    assert model.hidden_dim == 8 and model.recurrent.nonlinearity == "tanh"


def test_2_no_parameter_or_rng_mutation(frozen):
    model, policy, provenance, *_ = frozen
    assert state_hash(model, policy) == provenance["parameter_hash"]


def test_3_exact_midpoint_for_every_seed_and_transition(config):
    for family in FAMILIES:
        for seed in range(20):
            model, _, _ = load_final_model(config, family, seed)
            a = natural_rollout(model, config.sequence, CUES["A"])["hidden"]
            b = natural_rollout(model, config.sequence, CUES["B"])["hidden"]
            midpoint = midpoint_targets(a, b)
            assert np.allclose(midpoint, (a + b) / 2, atol=0, rtol=0)


def test_4_mixed_cue_is_exact_composite():
    assert assert_mixed_cue()


def _single(frozen, probe, clamp, position=2):
    model, _, _, _, _, rolls, midpoint, _ = frozen
    return intervention_rollout(
        model, (0, 1, 2, 3, 0, 2, 1, 3), probe, midpoint,
        rolls["A"]["hidden"], rolls["B"]["hidden"], "single_midpoint",
        clamp, 0, position,
    )


def test_5_mixed_d0_equals_d2_transitionwise(frozen):
    d0 = _single(frozen, "Mixed", "D0_natural")
    d2 = _single(frozen, "Mixed", "D2_composite")
    assert np.array_equal(d0["hidden"], d2["hidden"])
    assert np.array_equal(d0["logits"], d2["logits"])


def test_6_nocontext_d0_equals_d1_transitionwise(frozen):
    d0 = _single(frozen, "NoContext", "D0_natural")
    d1 = _single(frozen, "NoContext", "D1_zero")
    assert np.array_equal(d0["hidden"], d1["hidden"])
    assert np.array_equal(d0["logits"], d1["logits"])


def test_7_exactly_one_patch_at_registered_position(frozen):
    run = _single(frozen, "Novel1", "D0_natural", position=4)
    assert run["patches"] == (4,)


def test_8_patched_state_is_next_recurrent_prestate(frozen):
    run = _single(frozen, "Novel2", "D0_natural", position=3)
    assert np.array_equal(run["prestates"][4], run["hidden"][3])


def test_9_teacher_forced_tokens_are_identical(frozen):
    reference = _single(frozen, "Novel3", "D0_natural")["tokens"]
    for clamp in CLAMPS:
        assert np.array_equal(reference, _single(frozen, "Novel3", clamp)["tokens"])


def test_10_raw_ce_uses_exactly_seven_predictions(frozen):
    run = _single(frozen, "Novel4", "D0_natural")
    metrics = prediction_metrics(run["logits"], run["targets"])
    assert len(metrics["transition_losses"]) == 7
    assert np.isclose(metrics["raw_loss"], metrics["transition_losses"].mean())


def test_11_wrong_position_mapping_is_fixed_derangement():
    mapping = derangement()
    assert len(mapping) == 7 and np.all(mapping != np.arange(7))
    assert np.array_equal(mapping, derangement())


def test_12_orthogonal_control_is_orthogonal_and_norm_matched():
    target = np.arange(1, 9, dtype=float)
    control = orthogonal_matched(target, 11, 5)
    assert abs(target @ control) < 1e-8
    assert np.isclose(np.linalg.norm(target), np.linalg.norm(control), atol=1e-8)


def test_13_novelmean_uses_all_four_variants_once(frozen):
    _, _, _, natural, interventions, *_ = frozen
    source = pd.concat([natural, interventions], ignore_index=True)
    combined = add_novel_mean(source)
    novel = source[source.probe.isin(("Novel1", "Novel2", "Novel3", "Novel4"))]
    derived = combined[combined.probe.eq("NovelMean")]
    assert len(derived) == len(novel) // 4
    assert tuple(probe for probe in PROBES if probe.startswith("Novel")) == (
        "Novel1", "Novel2", "Novel3", "Novel4"
    )


def test_14_targets_cannot_receive_behavioral_losses():
    assert set(inspect.signature(midpoint_targets).parameters) == {"a_hidden", "b_hidden"}
    assert "loss" not in inspect.getsource(midpoint_targets).lower()


def test_15_bootstrap_resamples_seed_level_values_only(config):
    comparisons = pd.DataFrame({
        "family": [HIGH] * 5,
        "seed": np.arange(5),
        "comparison": ["synthetic"] * 5,
        "value": np.arange(5, dtype=float),
        "predicted_direction_observed": [False, True, True, True, True],
    })
    result = bootstrap_summary(comparisons, config)
    assert result.loc[0, "n_seeds"] == 5
    assert result.loc[0, "resampling_unit"] == "seed"


def test_16_acquisition_is_true_changing_parameter_replay(config):
    acquisition = replay_acquisition(config, HIGH, 0)
    assert len(acquisition) == 170
    assert acquisition.parameters_changed.all()
    assert acquisition.parameter_hash.nunique() == 170
    assert acquisition.parameter_hash_pre.nunique() == 170


def test_mandatory_figures_are_enforced(tmp_path):
    with pytest.raises(RuntimeError, match="Missing Phase 1J mandatory figures"):
        enforce_figures(tmp_path)
