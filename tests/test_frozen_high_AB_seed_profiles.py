"""Tests for the frozen High-scale A/B seed-profile posthoc."""

from __future__ import annotations

import numpy as np
import pandas as pd

from posthoc_transition_switch.frozen_seed_profiles import (
    calculate_seed_profile_outputs, load_frozen_high_profiles,
    run_frozen_seed_profile_posthoc,
)


def _frozen_table():
    rows = []
    for seed in range(20):
        for context in ("A", "B"):
            for position in range(1, 8):
                delta = (seed + 1) * position / 1000
                rows.append({
                    "stage": "stage2_layout", "cohort": "confirmatory",
                    "condition": "High_monotonic", "seed": seed,
                    "probe_context": context, "transition_position": position,
                    "normalized_RT": .4 + (delta if context == "B" else 0),
                    "optimizer_steps": 0, "parameter_updates": 0,
                    "human_rows_used": 0,
                })
    return pd.DataFrame(rows)


def test_delta_magnitude_mean_and_correlations_are_exact(tmp_path):
    source = tmp_path / "probes.csv"
    _frozen_table().to_csv(source, index=False)
    frozen = load_frozen_high_profiles(source)
    results = calculate_seed_profile_outputs(frozen)
    expected_seed0 = np.arange(1, 8) / 1000
    np.testing.assert_allclose(
        results["wide"].iloc[0, 1:].to_numpy(float), expected_seed0,
    )
    assert np.isclose(
        results["magnitude"].iloc[0].D, np.linalg.norm(expected_seed0)
    )
    assert np.isclose(results["mean_pairwise_correlation"], 1)
    assert np.isclose(results["median_pairwise_correlation"], 1)


def test_pipeline_writes_contract_without_changing_source(tmp_path):
    source = tmp_path / "probes.csv"
    _frozen_table().to_csv(source, index=False)
    before = source.read_bytes()
    output = tmp_path / "output"
    run_frozen_seed_profile_posthoc(source, output, quiet=True)
    assert source.read_bytes() == before
    for name in (
        "frozen_high_AB_delta_by_seed_transition.csv",
        "frozen_high_AB_profile_magnitude.csv",
        "frozen_high_AB_mean_delta.csv",
        "frozen_high_AB_seed_correlation_matrix.csv",
        "figure_frozen_high_AB_delta_heatmap.png",
        "FROZEN_HIGH_AB_SEED_PROFILE_SUMMARY.md",
        "frozen_high_AB_source_audit.json",
    ):
        assert (output / name).exists()
