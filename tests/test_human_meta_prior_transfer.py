"""T1--T10 contracts for locked-meta-prior target transfer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from human_meta_prior_transfer.analysis import canonicalize_familiar_profiles, enforce_familiar_gate
from human_meta_prior_transfer.checkpoints import file_sha256, load_locked_checkpoint
from human_meta_prior_transfer.config import load_human_meta_transfer_config
from human_meta_prior_transfer.pipeline import REQUIRED, run_human_meta_transfer
from human_meta_prior_transfer.training import ablated_state


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    output = tmp_path_factory.mktemp("human_meta_transfer")
    config = load_human_meta_transfer_config(ROOT / "configs/human_meta_transfer_config.json", smoke=True, output=output)
    config.payload["smoke"]["seeds"] = [0]
    config.payload["smoke"]["checkpoint_materialization_families"] = 1
    tables = run_human_meta_transfer(config)
    return config, output, tables


def test_t1_checkpoint_identity_is_hashed_and_exact(result):
    config, output, _ = result; registry = json.loads((output / "human_meta_transfer_checkpoint_hashes.json").read_text())
    assert len(registry["checkpoints"]) == 3
    for row in registry["checkpoints"]:
        payload, path = load_locked_checkpoint(config, row["training_condition"], row["seed"])
        assert row["file_sha256"] == file_sha256(path)
        assert row["model_state_sha256"] == payload["model_state_sha256"]


def test_t2_optimizer_and_t3_readout_are_reset(result):
    audit = result[2]["human_meta_transfer_training_audit.csv"]
    assert audit.target_optimizer_empty_at_start.all()
    assert not audit.synthetic_optimizer_state_carried.any()
    assert audit.execution_readout_freshly_initialized.all()
    assert audit.initial_readout_raw_slope.eq(0).all() and audit.initial_readout_intercept.eq(0).all()


def test_t4_probe_firewall_excludes_probe_contexts_from_training(result):
    tables = result[2]; assert set(tables["human_meta_transfer_training.csv"].context) == {"A", "B"}
    audit = tables["human_meta_transfer_training_audit.csv"]
    assert audit.mixed_training_updates.eq(0).all() and audit.novel_training_updates.eq(0).all()
    reference = tables["human_meta_transfer_human_profile_reference.csv"]
    assert len(reference) == 7


def test_t5_frozen_probes_hash_all_parameters(result):
    audit = result[2]["human_meta_transfer_probe_audit.csv"]
    assert audit.parameters_unchanged.all() and audit.optimizer_unchanged.all()
    assert (audit.parameter_hash_before == audit.parameter_hash_after).all()


def test_t6_hidden_state_is_reset_for_every_probe(result):
    probes = result[2]["human_meta_transfer_probe_raw.csv"]
    assert probes.hidden_state_policy.eq("zero_reset").all()
    assert result[2]["human_meta_transfer_training_audit.csv"].hidden_state_reset_each_sequence.all()


def test_t7_primary_rows_are_first_exposures(result):
    first = result[2]["human_meta_transfer_first_exposure.csv"]
    assert first.probe_exposure_index.eq(1).all() and first.primary_first_exposure.all()


def test_t8_whole_profile_canonicalization_never_swaps_transitions_individually():
    rows = []
    for context, offset in (("A", .8), ("B", .2), ("Mixed", .5), ("Novel", .7), ("Neutral", .6)):
        for position in range(1, 8):
            rows.append({"group": "Gx", "group_label": "x", "source_condition": "x", "hidden_size": 8,
                         "seed": 0, "checkpoint": 170, "context": context, "transition_position": position,
                         "normalized_RT": offset + position / 100})
    canonical = canonicalize_familiar_profiles(pd.DataFrame(rows))
    assert canonical[canonical.context.eq("B")].canonical_context.eq("Fast").all()
    assert canonical[canonical.context.eq("A")].canonical_context.eq("Slow").all()
    assert canonical.fast_source_context.nunique() == canonical.slow_source_context.nunique() == 1


def test_t9_failed_familiar_runs_cannot_enter_primary_inference():
    metrics = pd.DataFrame({"group": ["G1", "G2"], "seed": [0, 0], "value": [1., 2.]})
    gates = pd.DataFrame([{"row_type": "run", "group": "G1", "seed": 0, "run_gate_passed_excluding_group_F4": True},
                          {"row_type": "run", "group": "G2", "seed": 0, "run_gate_passed_excluding_group_F4": True},
                          {"row_type": "group", "group": "G1", "seed": None, "gate_passed": True},
                          {"row_type": "group", "group": "G2", "seed": None, "gate_passed": False}])
    eligible = enforce_familiar_gate(metrics, gates)
    assert eligible.set_index("group").primary_inference_eligible.to_dict() == {"G1": True, "G2": False}


def test_t10_G1_to_G4_training_contracts_and_schedules_are_identical(result):
    training = result[2]["human_meta_transfer_training.csv"]
    primary = training[training.group.isin(["G1", "G2", "G3", "G4"])]
    assert primary.target_contract_sha256.nunique() == 1
    assert primary.groupby("group").size().nunique() == 1
    schedules = primary.pivot(index="training_sequence", columns="group", values="context")
    assert schedules.apply(lambda row: row.nunique() == 1, axis=1).all()


def test_registered_outputs_and_figures_exist(result):
    _, output, tables = result
    assert all((output / name).exists() for name in REQUIRED)
    assert len(list(output.glob("figure_*.png"))) == 10
    assert set(tables["human_meta_transfer_group_registry.csv"].group) == {"G1", "G2", "G3", "G4", "G5"}


def test_conditional_ablation_resets_only_the_named_subsystem(result):
    config = result[0]; payload, _ = load_locked_checkpoint(config, "P2_meta", 0)
    original = payload["model_state_dict"]
    reset, keys = ablated_state(config, original, "recurrent", 0)
    assert set(keys) == {"recurrent.weight_hh_l0", "recurrent.bias_hh_l0"}
    assert any(not torch.equal(reset[name], original[name]) for name in keys)
    assert all(torch.equal(reset[name], original[name]) for name in original if name not in keys)
