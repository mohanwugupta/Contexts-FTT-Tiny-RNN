"""Frozen High-scale A/B profile heterogeneity analysis (no model evaluation)."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


EXPECTED_SEEDS = tuple(range(20))
EXPECTED_POSITIONS = tuple(range(1, 8))
REQUIRED_COLUMNS = (
    "stage", "cohort", "condition", "seed", "probe_context",
    "transition_position", "normalized_RT", "optimizer_steps",
    "parameter_updates", "human_rows_used",
)


def _progress(message: str, quiet: bool):
    if not quiet:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] FROZEN-A/B | {message}", flush=True)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_high_profiles(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the saved 20-seed frozen probe table."""
    source = Path(path).expanduser().resolve()
    frame = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen probe table is missing columns: {missing}")
    selected = frame[
        frame.stage.eq("stage2_layout")
        & frame.cohort.eq("confirmatory")
        & frame.condition.eq("High_monotonic")
        & frame.probe_context.isin(["A", "B"])
    ].copy()
    if len(selected) != 20 * 2 * 7:
        raise ValueError(f"Expected 280 frozen High A/B rows, found {len(selected)}.")
    if set(selected.seed.astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError("Frozen High A/B probe seed set is not exactly 0--19.")
    if set(selected.transition_position.astype(int)) != set(EXPECTED_POSITIONS):
        raise ValueError("Frozen High A/B transition positions are not exactly 1--7.")
    keys = ["seed", "probe_context", "transition_position"]
    if selected.duplicated(keys).any():
        raise ValueError("Frozen High A/B seed-context-position keys are not unique.")
    if not np.isfinite(selected.normalized_RT.to_numpy(float)).all():
        raise ValueError("Frozen High A/B probe RT contains missing/nonfinite values.")
    if not selected.optimizer_steps.eq(0).all() or not selected.parameter_updates.eq(0).all():
        raise ValueError("Selected rows are not registered zero-update frozen probes.")
    if not selected.human_rows_used.eq(0).all():
        raise ValueError("Selected frozen model rows unexpectedly reference human data.")
    selected["seed"] = selected.seed.astype(int)
    selected["transition_position"] = selected.transition_position.astype(int)
    return selected.sort_values(keys).reset_index(drop=True)


def calculate_seed_profile_outputs(
    frozen: pd.DataFrame,
) -> dict[str, pd.DataFrame | float]:
    pivot = frozen.pivot(
        index="seed", columns=["probe_context", "transition_position"],
        values="normalized_RT",
    ).sort_index()
    delta = (pivot["B"] - pivot["A"]).reindex(
        index=EXPECTED_SEEDS, columns=EXPECTED_POSITIONS
    )
    if delta.isna().any().any():
        raise ValueError("The 20 x 7 B-minus-A delta matrix is incomplete.")

    wide = delta.rename(columns=lambda k: f"delta_P{k}").reset_index()
    magnitude_values = np.sqrt(np.square(delta).sum(axis=1))
    magnitude = pd.DataFrame({
        "seed": delta.index.astype(int), "D": magnitude_values.to_numpy(float),
    })
    mean_delta = pd.DataFrame({
        "transition_position": list(EXPECTED_POSITIONS),
        "mean_delta": delta.mean(axis=0).to_numpy(float),
        "SD_delta": delta.std(axis=0, ddof=1).to_numpy(float),
        "positive_seed_count": (delta > 0).sum(axis=0).to_numpy(int),
        "negative_seed_count": (delta < 0).sum(axis=0).to_numpy(int),
    })
    mean_delta["SE_delta"] = mean_delta.SD_delta / np.sqrt(len(EXPECTED_SEEDS))
    mean_delta["CI95_lower"] = mean_delta.mean_delta - 1.96 * mean_delta.SE_delta
    mean_delta["CI95_upper"] = mean_delta.mean_delta + 1.96 * mean_delta.SE_delta
    mean_delta["majority_sign_fraction"] = mean_delta[
        ["positive_seed_count", "negative_seed_count"]
    ].max(axis=1) / len(EXPECTED_SEEDS)

    correlations = delta.T.corr()
    correlations.index.name = "seed"
    correlations.columns = [f"seed_{int(seed)}" for seed in correlations.columns]
    upper = correlations.to_numpy()[np.triu_indices(len(EXPECTED_SEEDS), k=1)]
    grand_mean_magnitude = float(np.linalg.norm(delta.mean(axis=0).to_numpy(float)))
    mean_individual_magnitude = float(magnitude.D.mean())
    return {
        "wide": wide,
        "magnitude": magnitude,
        "mean_delta": mean_delta,
        "correlations": correlations.reset_index(),
        "mean_pairwise_correlation": float(np.mean(upper)),
        "median_pairwise_correlation": float(np.median(upper)),
        "negative_pair_fraction": float(np.mean(upper < 0)),
        "grand_mean_magnitude": grand_mean_magnitude,
        "mean_individual_magnitude": mean_individual_magnitude,
        "cancellation_ratio": grand_mean_magnitude / mean_individual_magnitude,
    }


def _make_figures(output: Path, results: dict[str, pd.DataFrame | float]):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "frozen-ab-mpl")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = results["wide"].set_index("seed")
    values = delta.to_numpy(float)
    limit = float(np.max(np.abs(values)))
    fig, axis = plt.subplots(figsize=(9, 9))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(7), [f"P{k}" for k in EXPECTED_POSITIONS])
    axis.set_yticks(range(20), [str(seed) for seed in EXPECTED_SEEDS])
    axis.set_xlabel("Ordinal transition position")
    axis.set_ylabel("Seed")
    axis.set_title("Frozen High-scale A/B RT profiles: Δ = RT(B) − RT(A)")
    colorbar = fig.colorbar(image, ax=axis, shrink=.82)
    colorbar.set_label("Normalized RT difference")
    fig.tight_layout()
    fig.savefig(output / "figure_frozen_high_AB_delta_heatmap.png", dpi=220)
    fig.savefig(output / "figure_frozen_high_AB_delta_heatmap.pdf")
    plt.close(fig)

    corr = results["correlations"].set_index("seed").to_numpy(float)
    fig, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ticks = range(20)
    axis.set_xticks(ticks, EXPECTED_SEEDS, rotation=90)
    axis.set_yticks(ticks, EXPECTED_SEEDS)
    axis.set_xlabel("Seed")
    axis.set_ylabel("Seed")
    axis.set_title("Pairwise correlation of seven-position Δ profiles")
    colorbar = fig.colorbar(image, ax=axis, shrink=.82)
    colorbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output / "figure_frozen_high_AB_seed_correlations.png", dpi=220)
    fig.savefig(output / "figure_frozen_high_AB_seed_correlations.pdf")
    plt.close(fig)


def _write_report(output: Path, results: dict[str, pd.DataFrame | float]):
    mean_delta = results["mean_delta"]
    sign_range = (
        float(mean_delta.majority_sign_fraction.min()),
        float(mean_delta.majority_sign_fraction.max()),
    )
    report = f"""# Frozen High-scale A/B seed-profile posthoc

## Data and method

This analysis read the saved frozen A/B probe RTs for the 20 High-monotonic
Stage 2 confirmatory seeds. It did not load a checkpoint, retrain a model, or
rerun a probe. For seed `s` and transition position `k`,
`delta[s,k] = RT(B,s,k) - RT(A,s,k)`.

## Results

- Mean individual profile magnitude `D_s`: {results['mean_individual_magnitude']:.6f}
- Median individual profile magnitude: {results['magnitude'].D.median():.6f}
- Magnitude of the across-seed mean profile: {results['grand_mean_magnitude']:.6f}
- Grand-mean / mean-individual magnitude ratio: {results['cancellation_ratio']:.3f}
- Mean pairwise seed-profile correlation: {results['mean_pairwise_correlation']:.3f}
- Median pairwise seed-profile correlation: {results['median_pairwise_correlation']:.3f}
- Fraction of pairwise correlations below zero: {results['negative_pair_fraction']:.3f}
- Majority-sign agreement across positions: {sign_range[0]:.2f}--{sign_range[1]:.2f}

Across-seed mean deltas:

```text
{mean_delta[['transition_position', 'mean_delta', 'positive_seed_count', 'negative_seed_count', 'majority_sign_fraction']].to_string(index=False)}
```

## Interpretation

**The networks have strong but idiosyncratic A/B transition profiles that
substantially cancel in the grand mean.** The typical seed-level magnitude is
large, while the grand-mean profile retains only
{100 * results['cancellation_ratio']:.1f}% of that magnitude. Pairwise profile
correlations center near zero, {100 * results['negative_pair_fraction']:.1f}%
are negative, and no transition position has more than
{100 * sign_range[1]:.0f}% of seeds sharing one sign. The sign and seven-position
pattern therefore vary substantially rather than being consistently aligned
across seeds.
"""
    (output / "FROZEN_HIGH_AB_SEED_PROFILE_SUMMARY.md").write_text(
        report, encoding="utf-8"
    )


def run_frozen_seed_profile_posthoc(
    probe_path: str | Path, output_dir: str | Path, *, quiet: bool = False,
) -> dict[str, pd.DataFrame | float]:
    source = Path(probe_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = _hash(source)
    _progress(f"Reading saved probes: {source}", quiet)
    frozen = load_frozen_high_profiles(source)
    results = calculate_seed_profile_outputs(frozen)
    results["wide"].to_csv(output / "frozen_high_AB_delta_by_seed_transition.csv", index=False)
    results["magnitude"].to_csv(output / "frozen_high_AB_profile_magnitude.csv", index=False)
    results["mean_delta"].to_csv(output / "frozen_high_AB_mean_delta.csv", index=False)
    results["correlations"].to_csv(output / "frozen_high_AB_seed_correlation_matrix.csv", index=False)
    _make_figures(output, results)
    _write_report(output, results)
    after = _hash(source)
    if before != after:
        raise AssertionError("The frozen probe source changed during analysis.")
    audit = {
        "analysis_version": "frozen-high-ab-seed-profiles-1.0.0",
        "source": str(source), "source_sha256_before": before,
        "source_sha256_after": after, "source_unchanged": True,
        "selected_probe_rows": len(frozen), "seeds": 20,
        "transition_positions": 7, "condition": "High_monotonic",
        "checkpoint_loading": False, "model_retraining": False,
        "probes_rerun": False, "optimizer_steps": 0,
        "parameter_updates": 0, "human_rows_used": 0,
    }
    (output / "frozen_high_AB_source_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _progress("Complete; saved 20 x 7 delta matrix and correlation outputs", quiet)
    return results
