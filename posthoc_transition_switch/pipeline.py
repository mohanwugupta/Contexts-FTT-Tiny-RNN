"""Read-only orchestration for transition x time-since-switch analyses."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from .analysis import (
    primary_statistics,
    profile_separation,
    switch_cost_by_transition,
    transition_profiles,
)
from .config import PosthocSwitchConfig
from .data import load_frozen_probes, load_transition_history


REQUIRED_OUTPUTS = (
    "posthoc_switch_transition_profiles.csv",
    "posthoc_switch_cost_by_transition.csv",
    "posthoc_AB_profile_separation.csv",
    "figure_posthoc_transition_profiles.png",
    "figure_posthoc_switch_costs.png",
    "figure_posthoc_block_recovery.png",
    "POSTHOC_SWITCH_TRANSITION_REPORT.md",
)


class MissingHistoricalTransitionData(RuntimeError):
    """Raised when final checkpoints cannot reconstruct the missing history."""


def _progress(config: PosthocSwitchConfig, message: str):
    if config.show_progress:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] POSTHOC-SWITCH | {message}", flush=True)


def _resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_existing(paths: dict[str, Path]) -> dict[str, str | None]:
    return {
        label: _sha256(path) if path.exists() and path.is_file() else None
        for label, path in paths.items()
    }


def _audit_payload(
    *, status: str, sources: dict[str, Path], before: dict[str, str | None],
    after: dict[str, str | None], reason: str | None,
) -> dict:
    return {
        "analysis_version": "posthoc-transition-switch-1.0.0",
        "status": status,
        "reason": reason,
        "sources": {key: str(value) for key, value in sources.items()},
        "source_hashes_before": before,
        "source_hashes_after": after,
        "source_hashes_unchanged": before == after,
        "human_rows_used": 0,
        "model_retraining": False,
        "checkpoint_loading": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "probes_rerun": False,
        "hyperparameters_changed": False,
        "model_selection_performed": False,
    }


def _write_json(path: Path, payload: dict):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _blocked_report(output: Path, transition_path: Path, reason: str):
    report = f"""# Posthoc transition x time-since-switch report

## Status

**Blocked safely: the historical transition-level RT stream is unavailable.**

Expected source: `{transition_path}`

{reason}

The saved final checkpoints encode final parameters only. A zero-step forward
pass can evaluate the final network state, but it cannot reconstruct the RT
emitted after each past training sequence. Reconstructing this table would
require replaying training, which violates the PRD's no-retraining constraint.

No checkpoint was loaded, no optimizer was constructed, and no parameter was
updated. Future ordinal-layout runs now persist this already-computed stream as
`stage2_layout_transition_training.csv`.
"""
    (output / "POSTHOC_SWITCH_TRANSITION_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def _validate_ordinal_registration(path: Path, config: PosthocSwitchConfig):
    if not path.exists():
        raise FileNotFoundError(f"Frozen ordinal configuration not found: {path}")
    registered = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    checks = {
        "n_blocks": config.n_blocks,
        "block_length": config.block_length,
        "confirmatory_seeds": list(config.expected_seeds),
        "human_rows_used": 0,
        "layout_or_context_noise_sigma": 0.0,
    }
    for key, expected in checks.items():
        if registered.get(key) != expected:
            mismatches.append(
                f"{key}: expected {expected!r}, found {registered.get(key)!r}"
            )
    if registered.get("blocked_by") not in (None, ""):
        mismatches.append(f"blocked_by={registered.get('blocked_by')!r}")
    if mismatches:
        raise ValueError(
            "Frozen ordinal run does not match the registered posthoc source: "
            + "; ".join(mismatches)
        )


def _condition_seed_mean(costs: pd.DataFrame) -> pd.Series:
    return costs.groupby(["condition", "seed"]).switch_cost.mean()


def _brief(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6g}")


def _write_report(
    output: Path, config: PosthocSwitchConfig, history: pd.DataFrame,
    costs: pd.DataFrame, separation: pd.DataFrame, statistics: pd.DataFrame,
    frozen: pd.DataFrame,
):
    seed_cost = _condition_seed_mean(costs)
    cost_summary = seed_cost.groupby("condition").agg(
        seeds="size", mean="mean", SD="std",
    ).reset_index()
    sep_summary = separation.groupby("condition", as_index=False).agg(
        seeds=("seed", "nunique"), D_switch=("D_switch", "mean"),
        D_early=("D_early", "mean"), D_settled=("D_settled", "mean"),
        D_frozen=("D_frozen", "mean"),
        D_switch_minus_settled=("D_switch_minus_settled", "mean"),
        fraction_D_switch_larger=("switch_separation_larger", "mean"),
    )
    high_test = statistics[
        statistics.comparison.eq("High D_switch minus D_settled vs 0")
    ].iloc[0]
    reliable = bool(
        high_test.mean_difference > 0 and high_test.CI95_lower > 0
    )
    interpretation = (
        "The High-scale A/B profile was reliably more separated immediately "
        "after a switch than after settling. Time-since-switch dynamics therefore "
        "provide the predicted explanation for the strong training Cross/Split "
        "structure despite closer settled item-level profiles."
        if reliable else
        "High-scale A/B profile separation was not reliably larger immediately "
        "after a switch than after settling. The training-versus-frozen discrepancy "
        "is not explained by time-since-switch dynamics in this posthoc test."
    )
    frozen_text = (
        "Saved Stage 3 A/B probe rows were read directly; probes were not rerun."
        if not frozen.empty else
        "No saved frozen A/B probe table was available; D_frozen and Figure 4 were omitted."
    )
    report = f"""# Posthoc transition x time-since-switch report

## 1. Data and source

This read-only analysis used {len(history):,} saved transition observations from
the 20-seed Stage 2 confirmatory High- and Low-monotonic run. Blocks 2--{config.n_blocks}
were analyzed because block 1 has no preceding context switch. Each seed
therefore contributes eight A-switch and eight B-switch blocks. Switch is
sequence 1, early is sequences 1--2, and settled is sequences 8--10.

No human data were loaded. No checkpoint was loaded, no optimizer was
constructed, and no model parameter was changed.

## 2. Switch cost

Switch cost is `RT_switch - RT_settled`, averaged over contexts and the seven
ordinal transition positions for each seed.

```text
{_brief(cost_summary)}
```

```text
{_brief(statistics.iloc[:3])}
```

## 3. A/B profile separation

`D_state` is the Euclidean distance between the seven-position A and B RT
profiles at that state.

```text
{_brief(sep_summary)}
```

```text
{_brief(statistics.iloc[3:])}
```

## 4. Frozen comparison

{frozen_text}

## 5. Interpretation

**{interpretation}**

All tests are explicitly posthoc. Estimates, seed-level values, 95% confidence
intervals, and paired two-sided t tests are retained in the CSV outputs.
"""
    (output / "POSTHOC_SWITCH_TRANSITION_REPORT.md").write_text(
        report, encoding="utf-8"
    )


def run_posthoc_switch_analysis(
    output_dir: str | Path, *, config: PosthocSwitchConfig,
) -> dict[str, pd.DataFrame]:
    """Run the registered analysis without loading or updating any model."""
    config.validate()
    repo = Path(__file__).resolve().parents[1]
    output = _resolve(repo, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "transition_history": _resolve(repo, config.transition_source),
        "frozen_probes": _resolve(repo, config.frozen_probe_source),
        "ordinal_config": _resolve(repo, config.ordinal_config_source),
    }
    before = _hash_existing(sources)
    transition_path = sources["transition_history"]
    if not transition_path.exists():
        reason = (
            "The ordinal-layout run saved sequence-level training summaries and "
            "final checkpoints, but not the transition-by-sequence history required "
            "to recover time since switch."
        )
        _blocked_report(output, transition_path, reason)
        after = _hash_existing(sources)
        _write_json(
            output / "posthoc_source_audit.json",
            _audit_payload(
                status="blocked_missing_historical_transition_data",
                sources=sources, before=before, after=after, reason=reason,
            ),
        )
        _progress(config, "Blocked: historical transition stream was not saved")
        raise MissingHistoricalTransitionData(
            f"Required historical transition data not found: {transition_path}"
        )

    _progress(config, f"Validating frozen run: {sources['ordinal_config']}")
    _validate_ordinal_registration(sources["ordinal_config"], config)
    _progress(config, f"Loading historical transitions: {transition_path}")
    history = load_transition_history(transition_path, config)
    frozen = load_frozen_probes(sources["frozen_probes"], config)
    _progress(config, f"Validated {len(history):,} transition rows")

    profiles, recovery = transition_profiles(history, config)
    costs = switch_cost_by_transition(profiles)
    separation = profile_separation(profiles, frozen)
    statistics = primary_statistics(costs, separation)

    tables = {
        "posthoc_switch_transition_profiles": profiles,
        "posthoc_switch_cost_by_transition": costs,
        "posthoc_AB_profile_separation": separation,
        "posthoc_block_recovery": recovery,
        "posthoc_primary_statistics": statistics,
    }
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)

    if config.make_plots:
        _progress(config, "Rendering transition, switch-cost, and recovery figures")
        from .plotting import make_figures
        make_figures(output, profiles, costs, recovery, frozen)

    _write_report(
        output, config, history, costs, separation, statistics, frozen
    )
    after = _hash_existing(sources)
    if before != after:
        raise AssertionError("A frozen source changed during the posthoc analysis.")
    audit = _audit_payload(
        status="complete", sources=sources, before=before, after=after, reason=None,
    )
    audit.update({
        "config": asdict(config),
        "transition_rows_analyzed": len(history),
        "seed_level_profile_rows": len(profiles),
        "seed_level_separation_rows": len(separation),
        "frozen_probe_rows_read": len(frozen),
    })
    _write_json(output / "posthoc_source_audit.json", audit)
    expected = [
        name for name in REQUIRED_OUTPUTS
        if config.make_plots or not name.startswith("figure_")
    ]
    missing = [name for name in expected if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Missing posthoc switch outputs: {missing}")
    _progress(config, "Complete; zero optimizer steps and zero parameter updates")
    return tables
