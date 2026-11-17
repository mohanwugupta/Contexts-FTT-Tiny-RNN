from dataclasses import replace
import json

import numpy as np
import torch

from execution_objective_search.config import (
    ObjectiveCandidate,
    build_candidate_grid,
    post_failure_policy_candidates,
    smoke_config,
    stronger_primary_candidates,
)
from execution_objective_search.objectives import (
    calibrate_candidate,
    execution_loss,
    optimal_speed_numpy,
    summarize_calibration,
)
from execution_objective_search.pipeline import run_objective_search
from execution_objective_search.training import train_candidate


def test_candidate_a_has_closed_form_monotone_optimum():
    candidate = ObjectiveCandidate("A", c_time=.1, c_risk=.5, gamma=2)
    uncertainty = np.array([.8, .4, .1, 0.0])
    observed = optimal_speed_numpy(uncertainty, candidate)
    expected = np.clip(.1 / (2 * .5 * np.where(uncertainty > 0, uncertainty, 1)), 0, 1)
    expected[-1] = 1
    np.testing.assert_allclose(observed, expected)
    assert np.all(np.diff(observed) >= 0)


def test_execution_uncertainty_is_detached_from_sequence_logits():
    torch.manual_seed(1)
    logits = torch.randn(1, 7, 4, requires_grad=True)
    raw_speed = torch.randn(1, 7, requires_grad=True)
    speed = torch.sigmoid(raw_speed)
    targets = torch.tensor([1, 2, 0, 3, 1, 2, 3])
    candidate = ObjectiveCandidate("C", c_time=.1, c_risk=1, gamma=2)
    cost, optimum, statistics = execution_loss(logits, targets, speed, candidate)
    cost.mean().backward()
    assert logits.grad is None or torch.count_nonzero(logits.grad) == 0
    assert raw_speed.grad is not None
    assert torch.count_nonzero(raw_speed.grad) > 0
    assert not optimum.requires_grad
    assert not statistics["objective_uncertainty"].requires_grad


def test_post_failure_policy_bce_uses_only_detached_risk_optimum():
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0, -2.0], [0.0, 1.0, -1.0, -2.0]]],
        requires_grad=True,
    )
    raw_speed = torch.tensor([[0.2, -0.4]], requires_grad=True)
    speed = torch.sigmoid(raw_speed)
    candidate = ObjectiveCandidate("E", c_time=.1, c_risk=1, gamma=2)
    cost, optimum, statistics = execution_loss(
        logits, torch.tensor([0, 1]), speed, candidate
    )
    expected = torch.nn.functional.binary_cross_entropy(
        speed, optimum, reduction="none"
    )
    torch.testing.assert_close(cost, expected)
    cost.mean().backward()
    assert logits.grad is None or torch.count_nonzero(logits.grad) == 0
    assert torch.count_nonzero(raw_speed.grad) > 0
    assert not optimum.requires_grad
    assert not statistics["objective_uncertainty"].requires_grad


def test_post_failure_candidates_reuse_candidate_a_normative_mapping():
    candidates = post_failure_policy_candidates()
    assert len(candidates) == 10
    certainty = np.linspace(.3, .9, 13)
    uncertainty = 1 - certainty
    for policy in candidates:
        economic = ObjectiveCandidate(
            "A", policy.c_time, policy.c_risk, policy.gamma, policy.lambda_exec
        )
        np.testing.assert_allclose(
            optimal_speed_numpy(uncertainty, policy),
            optimal_speed_numpy(uncertainty, economic),
        )
    assert max(candidate.c_time / candidate.c_risk for candidate in candidates) == .75


def test_grid_contains_all_four_families_without_duplicate_ids():
    config = smoke_config()
    candidates = build_candidate_grid(config)
    assert {candidate.family for candidate in candidates} == {"A", "B", "C", "D"}
    assert len(candidates) == 5
    assert len({candidate.candidate_id for candidate in candidates}) == len(candidates)


def test_stronger_primary_followup_preserves_optima_while_scaling_gradients():
    candidates = stronger_primary_candidates()
    assert len(candidates) == 6
    weak = ObjectiveCandidate("A", c_time=.05, c_risk=1, gamma=2)
    strong = next(item for item in candidates if item.c_time == .5 and item.c_risk == 10)
    uncertainty = np.linspace(.05, .95, 11)
    np.testing.assert_allclose(
        optimal_speed_numpy(uncertainty, weak),
        optimal_speed_numpy(uncertainty, strong),
    )
    assert strong.c_time / weak.c_time == strong.c_risk / weak.c_risk == 10


def test_calibration_accepts_usable_primary_candidate_and_rejects_flat_one():
    config = smoke_config()
    primary = ObjectiveCandidate("A", c_time=.1, c_risk=1, gamma=2)
    primary_table = calibrate_candidate(primary, config)
    primary_summary = summarize_calibration(primary_table, primary, config)
    assert primary_summary["calibration_passed"]
    flat = ObjectiveCandidate("A", c_time=.005, c_risk=1, gamma=2)
    flat_summary = summarize_calibration(calibrate_candidate(flat, config), flat, config)
    assert not flat_summary["calibration_usable_dynamic_range"]
    assert not flat_summary["calibration_passed"]


def test_entropy_monotonicity_is_evaluated_from_four_choice_chance_upward():
    config = smoke_config()
    candidate = ObjectiveCandidate("C", c_time=.1, c_risk=.5, gamma=2)
    summary = summarize_calibration(calibrate_candidate(candidate, config), candidate, config)
    assert summary["calibration_monotone"]
    assert summary["calibration_positive_slope"]


def test_isolated_trainer_preserves_architecture_schedule_and_required_outputs():
    config = replace(
        smoke_config(), n_blocks=2, seeds=(0,), profile="unit", make_plots=False
    )
    candidate = ObjectiveCandidate("A", c_time=.1, c_risk=1, gamma=2)
    result = train_candidate(candidate, config, seed=0)
    assert result.optimizer_steps == 20
    assert len(result.sequence_results) == 20
    assert len(result.token_results) == 140
    assert set(result.sequence_results.context) == {"A", "B"}
    assert result.sequence_results.high_init_scale.eq(2).all()
    assert result.sequence_results.external_rt_targets.eq(0).all()
    assert result.sequence_results.human_probe_rows_used.eq(0).all()
    assert result.sequence_results.uncertainty_detached_from_logits.all()
    np.testing.assert_allclose(
        result.sequence_results.normalized_rt,
        1.0 - result.sequence_results.execution_speed,
        atol=1e-7,
    )


def test_pipeline_writes_required_deliverables_without_confirmatory_rerun(tmp_path):
    config = replace(
        smoke_config(), n_blocks=2, seeds=(0,), profile="unit", make_plots=False,
        show_progress=False,
    )
    candidate = ObjectiveCandidate("A", c_time=.1, c_risk=1, gamma=2)
    result = run_objective_search(tmp_path, config=config, candidates=(candidate,))
    for filename in (
        "EXECUTION_OBJECTIVE_SEARCH_REPORT.md",
        "objective_calibration.csv",
        "objective_screening_results.csv",
        "objective_candidate_summary.csv",
        "objective_search_config.json",
    ):
        assert (tmp_path / filename).exists()
    record = json.loads((tmp_path / "objective_search_config.json").read_text())
    assert record["confirmatory_20_seed_rerun_started"] is False
    assert record["human_target_rt_rows"] == 0
    assert record["mixed_novel_training_or_probe_rows"] == 0
    assert len(result["objective_screening_results"]) == 20
