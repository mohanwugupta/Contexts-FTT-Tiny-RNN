"""Read-only whole-profile Fast/Slow canonicalization of frozen High A/B probes."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .frozen_seed_profiles import (
    EXPECTED_POSITIONS, EXPECTED_SEEDS, load_frozen_high_profiles,
)


def _progress(message: str, quiet: bool):
    if not quiet:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] FAST/SLOW | {message}", flush=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_saved_mixed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "stage", "cohort", "condition", "seed", "probe_context",
        "transition_position", "normalized_RT", "optimizer_steps",
        "parameter_updates", "human_rows_used",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen probe table is missing Mixed columns: {missing}")
    mixed = frame[
        frame.stage.eq("stage2_layout")
        & frame.cohort.eq("confirmatory")
        & frame.condition.eq("High_monotonic")
        & frame.probe_context.eq("Mixed")
    ].copy()
    if len(mixed) != 20 * 7:
        raise ValueError(f"Expected 140 frozen High Mixed rows, found {len(mixed)}.")
    keys = ["seed", "transition_position"]
    if mixed.duplicated(keys).any():
        raise ValueError("Frozen High Mixed seed-position keys are not unique.")
    if set(mixed.seed.astype(int)) != set(EXPECTED_SEEDS):
        raise ValueError("Frozen High Mixed seed set is not exactly 0--19.")
    if set(mixed.transition_position.astype(int)) != set(EXPECTED_POSITIONS):
        raise ValueError("Frozen High Mixed positions are not exactly 1--7.")
    if not np.isfinite(mixed.normalized_RT.to_numpy(float)).all():
        raise ValueError("Frozen High Mixed RT contains missing/nonfinite values.")
    if not mixed.optimizer_steps.eq(0).all() or not mixed.parameter_updates.eq(0).all():
        raise ValueError("Selected Mixed rows are not saved zero-update probes.")
    if not mixed.human_rows_used.eq(0).all():
        raise ValueError("Selected Mixed rows unexpectedly reference human data.")
    mixed["seed"] = mixed.seed.astype(int)
    mixed["transition_position"] = mixed.transition_position.astype(int)
    return mixed.sort_values(keys).reset_index(drop=True)


def _delta_statistics(delta: pd.DataFrame) -> dict:
    magnitude = np.sqrt(np.square(delta).sum(axis=1))
    correlations = delta.T.corr()
    upper = correlations.to_numpy()[np.triu_indices(len(delta), k=1)]
    grand_magnitude = float(np.linalg.norm(delta.mean(axis=0).to_numpy(float)))
    mean_individual = float(magnitude.mean())
    return {
        "magnitude": magnitude,
        "correlations": correlations,
        "mean_pairwise_correlation": float(np.mean(upper)),
        "median_pairwise_correlation": float(np.median(upper)),
        "negative_pair_fraction": float(np.mean(upper < 0)),
        "grand_mean_magnitude": grand_magnitude,
        "mean_individual_magnitude": mean_individual,
        "magnitude_ratio": grand_magnitude / mean_individual,
    }


def canonicalize_whole_profiles(
    frozen_ab: pd.DataFrame, frozen_mixed: pd.DataFrame,
) -> dict[str, pd.DataFrame | dict]:
    """Assign one Fast/Slow label per seed, then retain that swap at all positions."""
    pivot = frozen_ab.pivot(
        index="seed", columns=["probe_context", "transition_position"],
        values="normalized_RT",
    ).sort_index()
    a, b = pivot["A"].copy(), pivot["B"].copy()
    mean_a, mean_b = a.mean(axis=1), b.mean(axis=1)
    if np.isclose(mean_a.to_numpy(), mean_b.to_numpy(), atol=1e-12).any():
        tied = mean_a.index[np.isclose(mean_a, mean_b, atol=1e-12)].tolist()
        raise ValueError(f"Fast/Slow label is undefined for tied seed(s): {tied}")
    fast_is_a = mean_a < mean_b
    labels = pd.DataFrame({
        "seed": list(EXPECTED_SEEDS), "mean_RT_A": mean_a.to_numpy(float),
        "mean_RT_B": mean_b.to_numpy(float),
        "Fast_original_context": np.where(fast_is_a, "A", "B"),
        "Slow_original_context": np.where(fast_is_a, "B", "A"),
        "whole_profile_swapped": ~fast_is_a.to_numpy(bool),
    })

    fast = a.where(fast_is_a, b, axis=0)
    slow = b.where(fast_is_a, a, axis=0)
    original_delta = b - a
    canonical_delta = slow - fast
    original_stats = _delta_statistics(original_delta)
    canonical_stats = _delta_statistics(canonical_delta)

    long_parts = []
    for seed in EXPECTED_SEEDS:
        for canonical_label, values, contexts in (
            ("Fast", fast.loc[seed], labels.set_index("seed").loc[seed, "Fast_original_context"]),
            ("Slow", slow.loc[seed], labels.set_index("seed").loc[seed, "Slow_original_context"]),
        ):
            long_parts.append(pd.DataFrame({
                "seed": seed, "canonical_label": canonical_label,
                "original_context": contexts,
                "transition_position": list(EXPECTED_POSITIONS),
                "normalized_RT": values.to_numpy(float),
            }))
    canonical_long = pd.concat(long_parts, ignore_index=True)
    mixed_long = frozen_mixed[["seed", "transition_position", "normalized_RT"]].copy()
    mixed_long["canonical_label"] = "Mixed"
    mixed_long["original_context"] = "Mixed"
    all_profiles = pd.concat([canonical_long, mixed_long], ignore_index=True)
    profile_summary = all_profiles.groupby(
        ["canonical_label", "transition_position"], as_index=False,
    ).agg(
        mean_RT=("normalized_RT", "mean"), SD_RT=("normalized_RT", "std"),
        seeds=("seed", "nunique"),
    )
    profile_summary["SE_RT"] = profile_summary.SD_RT / np.sqrt(profile_summary.seeds)
    profile_summary["CI95_lower"] = profile_summary.mean_RT - 1.96 * profile_summary.SE_RT
    profile_summary["CI95_upper"] = profile_summary.mean_RT + 1.96 * profile_summary.SE_RT

    sign_rows = []
    for label, matrix in (("Original B-A", original_delta), ("Canonical Slow-Fast", canonical_delta)):
        for position in EXPECTED_POSITIONS:
            values = matrix[position]
            positive, negative = int((values > 0).sum()), int((values < 0).sum())
            sign_rows.append({
                "labeling": label, "transition_position": position,
                "mean_delta": float(values.mean()),
                "positive_seed_count": positive, "negative_seed_count": negative,
                "majority_sign_fraction": max(positive, negative) / len(values),
            })
    sign_agreement = pd.DataFrame(sign_rows)
    comparison = pd.DataFrame([
        {
            "labeling": label,
            "mean_pairwise_correlation": stats["mean_pairwise_correlation"],
            "median_pairwise_correlation": stats["median_pairwise_correlation"],
            "negative_pair_fraction": stats["negative_pair_fraction"],
            "grand_mean_profile_magnitude": stats["grand_mean_magnitude"],
            "mean_individual_profile_magnitude": stats["mean_individual_magnitude"],
            "grand_mean_over_mean_individual": stats["magnitude_ratio"],
        }
        for label, stats in (
            ("Original B-A", original_stats),
            ("Canonical Slow-Fast", canonical_stats),
        )
    ])
    wide = canonical_delta.rename(columns=lambda k: f"delta_P{k}").reset_index()
    magnitudes = pd.DataFrame({
        "seed": list(EXPECTED_SEEDS),
        "D_original": original_stats["magnitude"].to_numpy(float),
        "D_canonical": canonical_stats["magnitude"].to_numpy(float),
    })
    correlations = canonical_stats["correlations"].copy()
    correlations.index.name = "seed"
    correlations.columns = [f"seed_{seed}" for seed in correlations.columns]
    return {
        "labels": labels, "wide": wide, "magnitudes": magnitudes,
        "canonical_long": canonical_long, "profile_summary": profile_summary,
        "sign_agreement": sign_agreement, "comparison": comparison,
        "correlations": correlations.reset_index(),
        "original_delta": original_delta, "canonical_delta": canonical_delta,
    }


def _plot(output: Path, results: dict[str, pd.DataFrame | dict]):
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fast-slow-mpl")
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    canonical = results["canonical_delta"].to_numpy(float)
    limit = float(np.max(np.abs(canonical)))
    fig, axis = plt.subplots(figsize=(9, 9))
    image = axis.imshow(canonical, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(7), [f"P{k}" for k in EXPECTED_POSITIONS])
    axis.set_yticks(range(20), EXPECTED_SEEDS)
    axis.set_xlabel("Ordinal transition position")
    axis.set_ylabel("Seed")
    axis.set_title("Whole-profile canonicalization: Δ = RT(Slow) − RT(Fast)")
    colorbar = fig.colorbar(image, ax=axis, shrink=.82)
    colorbar.set_label("Normalized RT difference")
    fig.tight_layout()
    fig.savefig(output / "figure_canonical_slow_minus_fast_heatmap.png", dpi=220)
    fig.savefig(output / "figure_canonical_slow_minus_fast_heatmap.pdf")
    plt.close(fig)

    summary = results["profile_summary"]
    colors = {"Fast": "#0072B2", "Slow": "#D55E00", "Mixed": "#009E73"}
    fig, axis = plt.subplots(figsize=(10, 6))
    for label in ("Fast", "Slow", "Mixed"):
        one = summary[summary.canonical_label.eq(label)]
        axis.plot(one.transition_position, one.mean_RT, marker="o", label=label,
                  color=colors[label])
        axis.fill_between(one.transition_position, one.CI95_lower, one.CI95_upper,
                          color=colors[label], alpha=.14)
    axis.set_xlabel("Ordinal transition position")
    axis.set_ylabel("Frozen normalized RT")
    axis.set_title("Frozen High-scale profiles after seed-level label canonicalization")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output / "figure_canonical_fast_slow_mixed_profiles.png", dpi=220)
    fig.savefig(output / "figure_canonical_fast_slow_mixed_profiles.pdf")
    plt.close(fig)


def _write_report(output: Path, results: dict[str, pd.DataFrame | dict]):
    comparison = results["comparison"].set_index("labeling")
    original = comparison.loc["Original B-A"]
    canonical = comparison.loc["Canonical Slow-Fast"]
    signs = results["sign_agreement"]
    canonical_signs = signs[signs.labeling.eq("Canonical Slow-Fast")]
    profiles = results["profile_summary"].pivot(
        index="transition_position", columns="canonical_label", values="mean_RT"
    ).reset_index()
    labels = results["labels"]
    report = f"""# Frozen High A/B whole-profile label canonicalization

## Method

For each of the 20 High-monotonic Stage 2 confirmatory seeds, Fast and Slow were
defined solely from that seed's mean frozen A versus B RT across all seven
transitions. That one seed-level label assignment was then applied to the entire
seven-position profile. No individual transition was relabeled. Mixed was read
only after labels were fixed and was used only for the descriptive mean profile.

No checkpoint was loaded, no model was retrained, and no probe was rerun.

Fast was A for {(labels.Fast_original_context == 'A').sum()} seeds and B for
{(labels.Fast_original_context == 'B').sum()} seeds.

## Original versus canonicalized profiles

```text
{results['comparison'].to_string(index=False)}
```

Canonicalization increased the grand-mean/mean-individual magnitude ratio from
{original.grand_mean_over_mean_individual:.3f} to
{canonical.grand_mean_over_mean_individual:.3f}. Mean pairwise correlation rose
from {original.mean_pairwise_correlation:.3f} to
{canonical.mean_pairwise_correlation:.3f}, and median correlation rose from
{original.median_pairwise_correlation:.3f} to
{canonical.median_pairwise_correlation:.3f}.

Canonicalized majority-sign agreement by transition:

```text
{canonical_signs[['transition_position', 'mean_delta', 'positive_seed_count', 'negative_seed_count', 'majority_sign_fraction']].to_string(index=False)}
```

Mean Fast, Slow, and saved Mixed profiles:

```text
{profiles[['transition_position', 'Fast', 'Slow', 'Mixed']].to_string(index=False)}
```

## Interpretation

**Global A/B label switching explains some cancellation, but substantial
transition-pattern idiosyncrasy remains.** After canonicalization, the grand
mean retains only {100 * canonical.grand_mean_over_mean_individual:.1f}% of the
typical individual magnitude, mean/median pairwise correlations remain near
zero ({canonical.mean_pairwise_correlation:.3f}/
{canonical.median_pairwise_correlation:.3f}), and
{100 * canonical.negative_pair_fraction:.1f}% of seed pairs remain negatively
correlated. Position-wise majority-sign agreement is only
{canonical_signs.majority_sign_fraction.min():.0%}--
{canonical_signs.majority_sign_fraction.max():.0%}. Thus the original
inconsistency is not largely reducible to a global exchange of A and B labels.

Because Fast/Slow is defined from the seven-position mean, `Slow - Fast` is
positive on average within every seed but need not be positive at each
individual transition.
"""
    (output / "FROZEN_HIGH_AB_LABEL_CANONICALIZATION_SUMMARY.md").write_text(
        report, encoding="utf-8"
    )


def run_label_canonicalization_posthoc(
    probe_path: str | Path, output_dir: str | Path, *, quiet: bool = False,
) -> dict[str, pd.DataFrame | dict]:
    source = Path(probe_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = _sha256(source)
    _progress(f"Reading saved frozen probes: {source}", quiet)
    frozen_ab = load_frozen_high_profiles(source)
    frozen_mixed = _load_saved_mixed(source)
    results = canonicalize_whole_profiles(frozen_ab, frozen_mixed)
    outputs = {
        "canonical_seed_labels": results["labels"],
        "canonical_slow_minus_fast_by_seed_transition": results["wide"],
        "canonical_seed_profile_magnitude": results["magnitudes"],
        "canonical_fast_slow_profiles_long": results["canonical_long"],
        "canonical_mean_transition_profiles": results["profile_summary"],
        "canonicalization_sign_agreement": results["sign_agreement"],
        "canonicalization_original_comparison": results["comparison"],
        "canonical_seed_correlation_matrix": results["correlations"],
    }
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    _plot(output, results)
    _write_report(output, results)
    after = _sha256(source)
    if before != after:
        raise AssertionError("Frozen probe source changed during canonicalization.")
    audit = {
        "analysis_version": "frozen-high-ab-label-canonicalization-1.0.0",
        "source": str(source), "source_sha256_before": before,
        "source_sha256_after": after, "source_unchanged": True,
        "AB_rows_used_for_labeling": len(frozen_ab),
        "Mixed_rows_used_for_labeling": 0,
        "Mixed_rows_used_descriptively_after_labeling": len(frozen_mixed),
        "Novel_rows_used": 0, "human_rows_used": 0,
        "checkpoint_loading": False, "model_retraining": False,
        "probes_rerun": False, "optimizer_steps": 0, "parameter_updates": 0,
        "label_assignment_level": "one whole-profile A/B swap per seed",
        "individual_transition_relabeling": False,
    }
    (output / "canonicalization_source_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _progress("Complete; whole-profile labels fixed for all 20 seeds", quiet)
    return results
