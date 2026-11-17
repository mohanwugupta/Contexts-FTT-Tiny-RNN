from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest
import torch

from context_controlled_recurrent_architectures.models import build_model
from human_context_control_transfer.analysis import canonicalize, passing_keys, probe_metrics
from human_context_control_transfer.config import ALL_GROUPS, EXPECTED_SEEDS, load_config
from human_context_control_transfer.engine import make_batch, probe_run, train_transfer, visible_features
from human_context_control_transfer.timeline import build_timeline, training_schedule
from human_svg_geometry_transfer.geometry import load_coordinate_registry


@pytest.fixture(scope="module")
def assets():
    config = load_config()
    registry = load_coordinate_registry(config.coordinate_registry, config.payload["coordinate_assignment"])
    return config, registry


@pytest.fixture(scope="module")
def trained_runs(assets):
    config, registry = assets
    a = train_transfer(config, registry, config.groups[0], 20, build_model("A", 20).state_dict())
    rp = train_transfer(config, registry, config.groups[4], 20, build_model("B", 20).state_dict())
    return a, rp


def test_locked_registry_and_independent_seed_bank(assets):
    config, _ = assets
    assert tuple(x["group"] for x in config.groups) == ALL_GROUPS
    assert config.seeds == EXPECTED_SEEDS
    assert not config.payload["optional_random_controls_included"]


def test_exact_human_timeline_maps_primary_probes():
    frame = build_timeline(); probes = frame.dropna(subset=["probe_name"]).set_index("probe_name")
    assert len(training_schedule(frame)) == 120
    assert probes.loc["Mixed1", "model_training_sequence_index"] == 40
    assert probes.loc["Mixed2", "model_training_sequence_index"] == 40
    assert probes.loc["MixedTest", "model_training_sequence_index"] == 120
    assert probes.weights_frozen_before_probe.all()


def test_no_label_or_target_channel_in_model_interface(assets):
    config, registry = assets
    batch = make_batch(registry, "Mixed", config.payload["human_training"]["token_sequence"])
    assert batch[0].shape == (1, 7, 4) and batch[1].shape == (1, 7, 42)
    assert "target" not in inspect.signature(build_model("B", 1).forward).parameters
    assert visible_features(registry, "NoContext").shape == (7, 42)


def test_training_uses_only_familiar_rows_and_exact_updates(trained_runs):
    for run in trained_runs:
        assert len(run.training) == 120
        assert set(run.training.context) == {"A", "B"}
        assert not run.training.latent_context_label_input.any()
        assert not run.training.target_passed_to_controller.any()
        assert run.training.optimizer_steps.iloc[-1] == 120
        assert sorted(run.snapshots) == [0, 20, 40, 50, 90, 120]


def test_matched_schedule_and_fresh_execution_heads(trained_runs):
    a, rp = trained_runs
    assert a.training[["training_sequence", "context", "human_block"]].reset_index(drop=True).equals(
        rp.training[["training_sequence", "context", "human_block"]].reset_index(drop=True))
    assert a.audit["execution_readout_freshly_initialized"] and rp.audit["execution_readout_freshly_initialized"]
    assert a.audit["initial_readout_sha256"] == rp.audit["initial_readout_sha256"]


def test_restricted_plasticity_freezes_shared_elman_core(trained_runs):
    _, run = trained_runs; before = run.snapshots[0]["model_state_dict"]; after = run.snapshots[120]["model_state_dict"]
    frozen = [k for k in before if k.startswith("input_to_core") or k.startswith("recurrent_core") or k == "core_bias"]
    adaptable = [k for k in before if k.startswith("context_gru") or k.startswith("context_to_core") or k.startswith("sequence_head")]
    assert frozen and all(torch.equal(before[k], after[k]) for k in frozen)
    assert any(not torch.equal(before[k], after[k]) for k in adaptable)


def test_frozen_probe_has_first_exposure_and_no_updates(assets, trained_runs):
    config, registry = assets; run = trained_runs[0]
    frame = probe_run(run, registry, 40, "Mixed1", ("A", "B", "Mixed"), tuple(config.payload["human_training"]["token_sequence"]))
    assert len(frame) == 21 and frame.optimizer_updates.eq(0).all()
    assert frame.probe_exposure_index.eq(1).all() and frame.primary_first_exposure.all()
    assert frame.weight_policy.eq("frozen_clone_discarded").all()


def test_canonicalization_swaps_whole_profiles_only():
    rows = []
    for context, values in {"A": [.7, .1, .7, .1, .7, .1, .7], "B": [.4] * 7, "Mixed": [.5] * 7}.items():
        for position, value in enumerate(values, 1):
            rows.append({"group":"g","architecture":"A","prior_condition":"x","seed":1,"checkpoint":40,"probe_name":"Mixed1",
                         "context":context,"transition_position":position,"normalized_RT":value})
    result = canonicalize(pd.DataFrame(rows))
    assert result.fast_source_context.nunique() == 1 and result.slow_source_context.nunique() == 1
    assert result.fast_source_context.iloc[0] == "B"  # lower whole-profile mean
    assert set(result[result.context == "B"].canonical_context) == {"Fast"}


def test_gate_filter_requires_both_seed_and_group_pass():
    frame = pd.DataFrame([
        {"row_type":"seed","group":"A-S","seed":20,"seed_gate_passed":True},
        {"row_type":"seed","group":"A-S","seed":21,"seed_gate_passed":False},
        {"row_type":"seed","group":"B-S","seed":20,"seed_gate_passed":True},
        {"row_type":"group","group":"A-S","seed":np.nan,"group_gate_passed":True},
        {"row_type":"group","group":"B-S","seed":np.nan,"group_gate_passed":False},
    ])
    assert passing_keys(frame) == {("A-S", 20)}


def test_probe_metrics_accept_mixed_hidden_widths(assets, trained_runs):
    config, registry = assets; run = trained_runs[0]; sequence = tuple(config.payload["human_training"]["token_sequence"])
    frame = probe_run(run, registry, 40, "Mixed1", ("A", "B", "Mixed"), sequence)
    canonical = canonicalize(frame); mixed, novel, no_context, hidden, controller = probe_metrics(canonical, {"Mixed": np.arange(7, dtype=float)})
    summary = mixed[mixed.row_type == "seed_summary"]
    assert len(summary) == 1 and summary.probe_name.iloc[0] == "Mixed1"
    assert len(hidden) == 7 and novel.empty and no_context.empty and controller.empty


def test_actual_preflight_artifact_contains_all_80_hashes():
    path = load_config().repo_root / "outputs/human_context_control_transfer_preflight/human_transfer_checkpoint_hashes.json"
    if not path.exists(): pytest.skip("read-only integration preflight has not been run")
    payload = json.loads(path.read_text())
    assert len(payload["checkpoints"]) == 80
    assert all(len(item["sha256"]) == 64 for item in payload["checkpoints"].values())
