"""Tests for whole-profile frozen A/B label canonicalization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from posthoc_transition_switch.frozen_seed_profiles import load_frozen_high_profiles
from posthoc_transition_switch.label_canonicalization import (
    _load_saved_mixed, canonicalize_whole_profiles,
    run_label_canonicalization_posthoc,
)


def _source():
    rows = []
    base_profile = np.arange(1, 8, dtype=float) / 100
    for seed in range(20):
        fast_context = "A" if seed % 2 == 0 else "B"
        for context in ("A", "B", "Mixed"):
            for position, base in enumerate(base_profile, start=1):
                if context == "Mixed":
                    rt = .5 + base
                else:
                    rt = .2 + base + (.1 if context != fast_context else 0)
                rows.append({
                    "stage": "stage2_layout", "cohort": "confirmatory",
                    "condition": "High_monotonic", "seed": seed,
                    "probe_context": context, "transition_position": position,
                    "normalized_RT": rt, "optimizer_steps": 0,
                    "parameter_updates": 0, "human_rows_used": 0,
                })
    return pd.DataFrame(rows)


def test_one_seed_level_swap_aligns_full_profiles(tmp_path):
    source = tmp_path / "probes.csv"
    _source().to_csv(source, index=False)
    ab, mixed = load_frozen_high_profiles(source), _load_saved_mixed(source)
    results = canonicalize_whole_profiles(ab, mixed)
    labels = results["labels"]
    assert (labels.Fast_original_context == np.where(labels.seed % 2 == 0, "A", "B")).all()
    delta = results["wide"].set_index("seed")
    np.testing.assert_allclose(delta.to_numpy(float), .1)
    comparison = results["comparison"].set_index("labeling")
    assert np.isclose(
        comparison.loc["Canonical Slow-Fast", "mean_pairwise_correlation"], 1
    )
    assert np.isclose(results["magnitudes"].D_original, results["magnitudes"].D_canonical).all()


def test_read_only_pipeline_writes_outputs_and_audit(tmp_path):
    source = tmp_path / "probes.csv"
    _source().to_csv(source, index=False)
    before = source.read_bytes()
    output = tmp_path / "output"
    run_label_canonicalization_posthoc(source, output, quiet=True)
    assert source.read_bytes() == before
    for name in (
        "canonical_seed_labels.csv",
        "canonical_slow_minus_fast_by_seed_transition.csv",
        "canonical_mean_transition_profiles.csv",
        "canonical_seed_correlation_matrix.csv",
        "canonicalization_original_comparison.csv",
        "figure_canonical_slow_minus_fast_heatmap.png",
        "figure_canonical_fast_slow_mixed_profiles.png",
        "FROZEN_HIGH_AB_LABEL_CANONICALIZATION_SUMMARY.md",
        "canonicalization_source_audit.json",
    ):
        assert (output / name).exists()
