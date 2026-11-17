"""Read-only contract tests for the posthoc transition/switch analysis."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from posthoc_transition_switch.analysis import (
    primary_statistics,
    profile_separation,
    switch_cost_by_transition,
    transition_profiles,
)
from posthoc_transition_switch.config import (
    HIGH_MONOTONIC, LOW_MONOTONIC, default_config,
)
from posthoc_transition_switch.data import validate_transition_history
from posthoc_transition_switch.pipeline import (
    MissingHistoricalTransitionData, REQUIRED_OUTPUTS,
    run_posthoc_switch_analysis,
)


def _synthetic_config(**changes):
    settings = {
        "expected_seeds": (0, 1, 2), "n_blocks": 5,
        "make_plots": False, "show_progress": False,
    }
    settings.update(changes)
    return replace(default_config(), **settings)


def _synthetic_history(config):
    rows = []
    for condition in config.conditions:
        for seed in config.expected_seeds:
            context_amplitude = (
                .045 + .001 * seed
                if condition == HIGH_MONOTONIC else .012 + .0003 * seed
            )
            recovery_amplitude = (
                .050 + .0015 * seed
                if condition == HIGH_MONOTONIC else .018 + .0004 * seed
            )
            for block in range(1, config.n_blocks + 1):
                context = "A" if block % 2 else "B"
                context_sign = 1 if context == "A" else -1
                for within in range(1, config.block_length + 1):
                    decay = (config.block_length - within) / 9
                    sequence = (block - 1) * config.block_length + within
                    for position in config.transition_positions:
                        rows.append({
                            "stage": "stage2_layout",
                            "cohort": "confirmatory",
                            "overlap_count": 0,
                            "condition": condition,
                            "seed": seed,
                            "training_sequence": sequence,
                            "block_index": block,
                            "context": context,
                            "within_block_sequence": within,
                            "is_context_switch": bool(block > 1 and within == 1),
                            "transition_position": position,
                            "normalized_RT": (
                                .35 + .008 * position + .0005 * seed
                                + context_sign * context_amplitude * decay
                                + recovery_amplitude * decay
                            ),
                        })
    return pd.DataFrame(rows)


def _synthetic_frozen(config):
    rows = []
    for condition in config.conditions:
        for seed in config.expected_seeds:
            amplitude = .004 if condition == HIGH_MONOTONIC else .002
            for context, sign in (("A", 1), ("B", -1)):
                for position in config.transition_positions:
                    rows.append({
                        "stage": "stage2_layout", "cohort": "confirmatory",
                        "condition": condition, "seed": seed,
                        "probe_context": context,
                        "transition_position": position,
                        "normalized_RT": .35 + .008 * position + sign * amplitude,
                    })
    return pd.DataFrame(rows)


def _write_registered_config(path, config):
    path.write_text(json.dumps({
        "analysis_version": "ordinal-layout-ecology-1.0.0",
        "n_blocks": config.n_blocks,
        "block_length": config.block_length,
        "confirmatory_seeds": list(config.expected_seeds),
        "human_rows_used": 0,
        "layout_or_context_noise_sigma": 0.0,
        "blocked_by": None,
    }))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_history_contract_and_string_boolean_parsing():
    config = _synthetic_config()
    history = _synthetic_history(config)
    history["is_context_switch"] = history.is_context_switch.astype(str)
    validated = validate_transition_history(history, config)
    assert len(validated) == config.expected_transition_rows
    assert not validated[validated.block_index.eq(1)].is_context_switch.any()
    assert validated[
        validated.block_index.gt(1)
        & validated.within_block_sequence.eq(1)
    ].is_context_switch.all()


def test_registered_windows_and_block_one_exclusion_are_exact():
    config = _synthetic_config()
    history = validate_transition_history(_synthetic_history(config), config)
    profiles, recovery = transition_profiles(history, config)
    assert len(profiles) == 2 * 3 * 2 * 7
    assert len(recovery) == 2 * 3 * 2 * 7 * 10
    assert set(profiles.switch_blocks) == {2}
    assert set(profiles.early_observations) == {4}
    assert set(profiles.settled_observations) == {6}
    selected = history[
        history.condition.eq(HIGH_MONOTONIC)
        & history.seed.eq(0) & history.context.eq("A")
        & history.transition_position.eq(1) & history.block_index.ge(2)
    ]
    expected = selected[
        selected.within_block_sequence.isin((8, 9, 10))
    ].normalized_RT.mean()
    actual = profiles[
        profiles.condition.eq(HIGH_MONOTONIC)
        & profiles.seed.eq(0) & profiles.context.eq("A")
        & profiles.transition_position.eq(1)
    ].RT_settled.item()
    assert actual == pytest.approx(expected)


def test_switch_costs_and_profile_separation_recover_signal():
    config = _synthetic_config()
    profiles, _ = transition_profiles(_synthetic_history(config), config)
    costs = switch_cost_by_transition(profiles)
    separation = profile_separation(profiles, _synthetic_frozen(config))
    statistics = primary_statistics(costs, separation)
    high_cost = costs[costs.condition.eq(HIGH_MONOTONIC)].switch_cost.mean()
    low_cost = costs[costs.condition.eq(LOW_MONOTONIC)].switch_cost.mean()
    assert high_cost > low_cost > 0
    assert separation[
        separation.condition.eq(HIGH_MONOTONIC)
    ].D_switch_minus_settled.gt(0).all()
    assert separation.D_frozen.notna().all()
    paired = statistics[
        statistics.comparison.eq("High minus Low mean switch cost")
    ].iloc[0]
    assert paired.n_paired_seeds == 3
    assert paired.mean_difference > 0


def test_missing_history_fails_without_retraining_or_fake_outputs(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    config = _synthetic_config(
        transition_source=str(missing),
        frozen_probe_source=str(tmp_path / "probe.csv"),
        ordinal_config_source=str(tmp_path / "config.json"),
    )
    output = tmp_path / "blocked"
    with pytest.raises(MissingHistoricalTransitionData):
        run_posthoc_switch_analysis(output, config=config)
    audit = json.loads((output / "posthoc_source_audit.json").read_text())
    assert audit["status"] == "blocked_missing_historical_transition_data"
    assert audit["optimizer_steps"] == audit["parameter_updates"] == 0
    assert not audit["model_retraining"] and not audit["checkpoint_loading"]
    assert (output / "POSTHOC_SWITCH_TRANSITION_REPORT.md").exists()
    assert not (output / "posthoc_switch_transition_profiles.csv").exists()


def test_synthetic_pipeline_is_read_only_and_writes_tabular_contract(tmp_path):
    base = _synthetic_config()
    transition_path, probe_path = tmp_path / "transitions.csv", tmp_path / "probes.csv"
    ordinal_path = tmp_path / "ordinal.json"
    _synthetic_history(base).to_csv(transition_path, index=False)
    _synthetic_frozen(base).to_csv(probe_path, index=False)
    _write_registered_config(ordinal_path, base)
    before = [_sha(path) for path in (transition_path, probe_path, ordinal_path)]
    config = replace(
        base, transition_source=str(transition_path),
        frozen_probe_source=str(probe_path),
        ordinal_config_source=str(ordinal_path),
    )
    output = tmp_path / "complete"
    tables = run_posthoc_switch_analysis(output, config=config)
    after = [_sha(path) for path in (transition_path, probe_path, ordinal_path)]
    assert before == after
    assert len(tables["posthoc_AB_profile_separation"]) == 6
    required_tables = [
        name for name in REQUIRED_OUTPUTS
        if name.endswith(".csv") or name.endswith(".md")
    ]
    assert all((output / name).exists() for name in required_tables)
    audit = json.loads((output / "posthoc_source_audit.json").read_text())
    assert audit["status"] == "complete"
    assert audit["source_hashes_unchanged"]
    assert audit["optimizer_steps"] == audit["parameter_updates"] == 0


def test_all_registered_figures_render_from_synthetic_data(tmp_path):
    config = _synthetic_config(make_plots=True)
    profiles, recovery = transition_profiles(_synthetic_history(config), config)
    costs = switch_cost_by_transition(profiles)
    separation = profile_separation(profiles, _synthetic_frozen(config))
    assert not separation.empty
    from posthoc_transition_switch.plotting import make_figures
    make_figures(tmp_path, profiles, costs, recovery, _synthetic_frozen(config))
    for name in (
        "figure_posthoc_transition_profiles.png",
        "figure_posthoc_switch_costs.png",
        "figure_posthoc_block_recovery.png",
        "figure_posthoc_frozen_comparison.png",
    ):
        assert (tmp_path / name).stat().st_size > 1_000


def test_analysis_package_does_not_import_model_or_training_modules():
    import posthoc_transition_switch.pipeline as pipeline_module
    source = open(pipeline_module.__file__, encoding="utf-8").read()
    assert "ordinal_layout_ecology.training" not in source
    assert "torch" not in source
    assert "optimizer.step" not in source
