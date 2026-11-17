from dataclasses import replace
import json

import numpy as np
import pandas as pd
import torch

from execution_readout_diagnostic.config import (
    READOUT_CONDITIONS,
    screen_config,
    smoke_config,
)
from execution_readout_diagnostic.pipeline import run_readout_diagnostic
from execution_readout_diagnostic.policies import (
    MonotonicConfidencePolicy,
    deterministic_normative_speed,
)
from execution_readout_diagnostic.state_space import fit_passing_readouts
from execution_readout_diagnostic.training import (
    initialize_readout,
    train_readout_condition,
)


def test_scientific_config_freezes_scale_seeds_schedule_and_normative_mapping():
    config = screen_config()
    config.validate()
    assert config.high_init_scale == 2
    assert config.seeds == tuple(range(5))
    assert config.training_sequences == 170
    assert config.normative_candidate.c_time == .5
    assert config.normative_candidate.c_risk == 1
    assert config.normative_candidate.gamma == 2


def test_monotonic_policy_has_exactly_two_parameters_and_strict_positive_slope():
    policy = MonotonicConfidencePolicy()
    assert sum(parameter.numel() for parameter in policy.parameters()) == 2
    confidence = torch.linspace(.01, .99, 101)
    speed = policy(confidence)
    assert policy.positive_slope > 0
    assert torch.all(torch.diff(speed) > 0)
    speed.sum().backward()
    assert all(parameter.grad is not None for parameter in policy.parameters())


def test_deterministic_policy_exactly_equals_analytical_optimum_and_detaches():
    config = smoke_config()
    confidence = torch.tensor([.25, .50, .70, .90], requires_grad=True)
    speed = deterministic_normative_speed(confidence, config.normative_candidate)
    expected = np.clip(.5 / (2 * (1 - confidence.detach().numpy())), 0, 1)
    np.testing.assert_allclose(speed.numpy(), expected)
    assert not speed.requires_grad
    assert np.all(np.diff(speed.numpy()) >= 0)


def test_readouts_share_initial_srn_and_unrestricted_head_is_exactly_zero():
    config = smoke_config()
    initialized = {
        condition: initialize_readout(condition, config, seed=3)
        for condition in READOUT_CONDITIONS
    }
    shared = {
        item.audit["shared_initial_parameters_sha256"] for item in initialized.values()
    }
    assert len(shared) == 1
    free = initialized["unrestricted_zero_initialized"]
    assert free.audit["execution_head_zero_initialized"]
    assert free.audit["execution_head_initial_abs_max"] == 0
    assert torch.count_nonzero(free.model.execution_head.weight) == 0
    assert torch.count_nonzero(free.model.execution_head.bias) == 0


def test_training_outputs_are_trial_aligned_nonhuman_and_respect_rt_identity():
    config = smoke_config()
    deterministic = train_readout_condition(
        "deterministic_normative", config, seed=0
    )
    unrestricted = train_readout_condition(
        "unrestricted_zero_initialized", config, seed=0
    )
    assert deterministic.optimizer_steps == unrestricted.optimizer_steps == 20
    assert len(deterministic.sequence_results) == 20
    assert len(deterministic.token_results) == 140
    np.testing.assert_allclose(
        deterministic.token_results.execution_speed,
        deterministic.token_results.objective_optimal_speed,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        unrestricted.sequence_results.normalized_rt,
        1 - unrestricted.sequence_results.execution_speed,
        atol=1e-7,
    )
    assert unrestricted.sequence_results.iloc[0].execution_speed == .5
    assert deterministic.sequence_results.external_rt_targets.eq(0).all()
    assert deterministic.sequence_results.human_probe_rows_used.eq(0).all()
    assert deterministic.sequence_results.uncertainty_detached_from_logits.all()


def test_state_space_firewall_only_fits_conditions_marked_as_passing():
    rows = []
    for seed in range(2):
        for trial in range(1, 41):
            block = (trial - 1) // 10 + 1
            rows.append({
                "readout_condition": "deterministic_normative",
                "seed": seed,
                "global_sequence": trial,
                "block_index": block,
                "sequence_in_block": (trial - 1) % 10 + 1,
                "context": "A" if block % 2 else "B",
                "normalized_rt": .8 * np.exp(-.02 * trial) + .01 * seed,
            })
    sequence = pd.DataFrame(rows)
    failed = pd.DataFrame([{
        "readout_condition": "deterministic_normative", "screening_passed": False
    }])
    assert fit_passing_readouts(sequence, failed)["state_space_model_fits"].empty
    passed = failed.assign(screening_passed=True)
    tables = fit_passing_readouts(sequence, passed)
    assert set(tables["state_space_model_fits"].model) == {
        "SS_LUMP", "SS_CROSS", "SS_SPLIT"
    }
    assert len(tables["state_space_predictions"]) == 3 * 40
    assert tables["state_space_model_fits"].human_rows_used.eq(0).all()
    assert tables["state_space_model_fits"].convergence_verified.all()


def test_pipeline_writes_audited_outputs_and_never_starts_confirmatory_run(tmp_path):
    config = replace(smoke_config(), show_progress=False, make_plots=False)
    result = run_readout_diagnostic(tmp_path, config=config)
    required = (
        "EXECUTION_READOUT_DIAGNOSTIC_REPORT.md",
        "execution_readout_diagnostic_config.json",
        "readout_sequence_results.csv",
        "readout_token_results.csv",
        "readout_initialization_audit.csv",
        "readout_gate_summary.csv",
        "state_space_model_fits.csv",
        "state_space_predictions.csv",
    )
    assert all((tmp_path / name).exists() for name in required)
    assert set(result["readout_gate_summary"].readout_condition) == set(
        READOUT_CONDITIONS
    )
    audit = result["readout_initialization_audit"]
    assert audit.groupby("seed").shared_initial_parameters_sha256.nunique().eq(1).all()
    record = json.loads(
        (tmp_path / "execution_readout_diagnostic_config.json").read_text()
    )
    assert record["objective_search_frozen"] is True
    assert record["objective_search_source_hashes_before"] == (
        record["objective_search_source_hashes_after"]
    )
    assert record["human_rows_used"] == 0
    assert record["confirmatory_20_seed_rerun_started"] is False
