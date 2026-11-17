"""Canonical end-to-end PRD2 Combined-context analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from . import __version__
from .epoch_data import normalize_epoch_data
from .epoch_fit import (
    counterbalance_generalization,
    fit_epoch_model_ladder,
    temporal_prediction_tests,
)
from .epoch_gate import distill_epoch_dynamics, fit_epoch_gate_models
from .epoch_metrics import (
    epoch_behavioral_summary,
    epoch_metric_rows,
    post_mixed_consequences,
)
from .epoch_plotting import make_epoch_figures
from .epoch_recovery import run_epoch_model_recovery
from .epoch_skill_states import fit_epoch_counterfactual_states
from .pipeline import _write_csv


@dataclass(frozen=True)
class EpochAnalysisConfig:
    seed: int = 42
    train_fraction: float = 0.8
    neural_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(42, 62)))
    gate_epochs: int = 300
    gate_learning_rate: float = 0.02
    recovery_simulations: int = 20
    recovery_noise_sd: float = 0.05
    recovery_regimes: tuple[str, ...] = ("easy", "human", "neighborhood")
    make_plots: bool = True
    show_progress: bool = True
    model_version: str = __version__


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_recovery_schedule(data: pd.DataFrame) -> pd.DataFrame:
    participant_ids = (
        data.groupby("counterbalance_history", sort=False).participant_id.first().tolist()
    )
    return data[data.participant_id.isin(participant_ids)].copy().reset_index(drop=True)


def _empirical_parameters(fits: pd.DataFrame) -> dict[str, dict[str, float]]:
    rows = fits[fits.fit_scope.eq("combined")]
    return {
        str(row.model): json.loads(row.parameters)
        for _, row in rows.drop_duplicates("model").iterrows()
    }


def run_epoch_analysis(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: EpochAnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run preprocessing, M0--M6, neural ablations, and recovery v2."""
    config = config or EpochAnalysisConfig()
    _set_seed(config.seed)
    def report(message: str) -> None:
        if config.show_progress:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{timestamp}] PRD2 | {message}", flush=True)

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    report(f"Starting analysis; input={source}, output={output}")
    report("Loading source CSV")
    raw = pd.read_csv(source)
    report(f"Loaded {len(raw):,} raw rows")
    report("Preprocessing and validating the 17 x 9 epoch schedule")
    clean = normalize_epoch_data(raw, dataset="Combined", cohort_only=True)
    report(
        f"Preprocessing complete: {clean.participant_id.nunique()} participants, "
        f"{len(clean):,} analyzed sequence rows"
    )
    report("Fitting restart-free, probe-free A/B counterfactual trajectories")
    counterfactuals, skill_fits = fit_epoch_counterfactual_states(
        clean, cross_talk=0.0, progress=report
    )
    report("A/B counterfactual fitting complete")
    report("Computing epoch behavioral summaries and post-Mixed consequences")
    summary = epoch_behavioral_summary(counterfactuals)
    mixed_1 = epoch_metric_rows(counterfactuals, "Mixed_1")
    mixed_2 = epoch_metric_rows(counterfactuals, "Mixed_2")
    mixed_test = epoch_metric_rows(counterfactuals, "Mixed_Test")
    aftereffects = post_mixed_consequences(counterfactuals)

    report("Fitting and evaluating explicit M0-M6 model ladder")
    fits, predictions, comparison, splits = fit_epoch_model_ladder(
        counterfactuals,
        seed=config.seed,
        train_fraction=config.train_fraction,
        progress=report,
    )
    report("Explicit model ladder complete")
    report("Running prospective temporal prediction tests")
    temporal = temporal_prediction_tests(counterfactuals, splits, progress=report)
    report("Running A/B-mirrored cross-counterbalance generalization")
    cb_generalization = counterbalance_generalization(counterfactuals, progress=report)

    if config.neural_seeds:
        report(
            f"Fitting neural discovery models: {len(config.neural_seeds)} seed(s), "
            f"up to {config.gate_epochs} epochs each"
        )
        gate_fits, gate_states, gate_resets, _ = fit_epoch_gate_models(
            counterfactuals,
            splits,
            seeds=config.neural_seeds,
            epochs=config.gate_epochs,
            learning_rate=config.gate_learning_rate,
            progress=report,
        )
        report("Distilling one-dimensional neural dynamics")
        distilled = distill_epoch_dynamics(counterfactuals, gate_states, splits)
    else:
        gate_fits, gate_states, gate_resets, distilled = (
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
    model_fits = pd.concat(
        [fits.assign(model_family="explicit"), gate_fits.assign(model_family="neural_discovery")],
        ignore_index=True,
        sort=False,
    )

    recovery_schedule = _representative_recovery_schedule(counterfactuals)
    recovery_jobs = len(config.recovery_regimes) * 6 * config.recovery_simulations
    report(
        f"Starting model recovery: {recovery_jobs} generated datasets; "
        "six candidate fits per dataset"
    )
    recovery, recovery_confusion, recovery_human = run_epoch_model_recovery(
        recovery_schedule,
        n_simulations=config.recovery_simulations,
        noise_sd=config.recovery_noise_sd,
        seed=config.seed,
        regimes=config.recovery_regimes,
        empirical_parameters=_empirical_parameters(fits),
        progress=report,
    )
    report("Model recovery complete")

    exports = {
        "clean_probe_data_v2.csv": clean,
        "counterfactual_AB_states_v2.csv": counterfactuals,
        "epoch_behavioral_summary.csv": summary,
        "mixed1_metrics.csv": mixed_1,
        "mixed2_metrics.csv": mixed_2,
        "mixed_test_metrics.csv": mixed_test,
        "post_mixed_aftereffects.csv": aftereffects,
        "model_fits_v2.csv": model_fits,
        "model_predictions_v2.csv": predictions,
        "model_comparison_by_epoch.csv": comparison,
        "temporal_prediction_tests.csv": temporal,
        "counterbalance_generalization.csv": cb_generalization,
        "gate_states_v2.csv": gate_states,
        "gate_reset_interventions.csv": gate_resets,
        "distilled_dynamics.csv": distilled,
        "model_recovery_results_v2.csv": recovery,
        "model_recovery_confusion_v2.csv": recovery_confusion,
        "model_recovery_human_parameter_region.csv": recovery_human,
        # Supporting audit outputs do not replace any required PRD2 file.
        "participant_splits_v2.csv": splits,
        "counterfactual_fit_parameters_v2.csv": skill_fits,
    }
    report(f"Writing {len(exports)} CSV outputs")
    for filename, frame in exports.items():
        report(f"Writing {filename} ({len(frame):,} rows)")
        _write_csv(frame, output / filename)

    final_recovery = (
        config.recovery_simulations >= 20
        and {"easy", "human", "neighborhood"}.issubset(config.recovery_regimes)
    )
    run_record = {
        "analysis_version": "PRD2",
        "analysis_tier": "final_config" if len(config.neural_seeds) >= 20 and final_recovery else "verification_only",
        "config": asdict(config),
        "source": {"path": str(source.resolve()), "sha256": _sha256(source)},
        "preprocessing": {
            "sequence_rule": "raw sequence 0 removed; original 2-10 retained as analysis 1-9",
            "correct_sequence_rule": ">=8 correct finite positive keypress RTs; RT uses correct keypresses only",
            "invalid_blocks": clean.attrs.get("invalid_blocks", []),
            "excluded_nonmodal_participants": clean.attrs.get("excluded_nonmodal_participants", []),
        },
        "outputs": {filename: int(len(frame)) for filename, frame in exports.items()},
        "scientific_safeguards": [
            "Foundational A/B trajectories use A/B observations only and have no restart parameter.",
            "Population mechanism parameters use training participants only; primary calibration excludes Post-Mixed Return and Mixed Test.",
            "Temporal Tests B/C do not refit the final Mixed Test or post-Mixed A/B target.",
            "Counterbalance/history is explicit; no unqualified previous-context claim is made.",
            "Midpoint deviation D on log RT is primary; mixture ratios are secondary and separation-masked.",
            "Recovery uses 17 blocks x 9 observations and both empirical counterbalances.",
        ],
    }
    with (output / "run_config_v2.json").open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2)
    report("Wrote run_config_v2.json")

    if config.make_plots:
        report("Rendering eight required figures")
        make_epoch_figures(
            output,
            counterfactuals=counterfactuals,
            predictions=predictions,
            comparison=comparison,
            aftereffects=aftereffects,
            gate_states=gate_states,
            gate_resets=gate_resets,
            recovery_confusion=recovery_confusion,
        )
        report("Figure rendering complete")
    report("Analysis complete")
    return {
        "clean": clean,
        "counterfactuals": counterfactuals,
        "behavioral_summary": summary,
        "model_fits": model_fits,
        "predictions": predictions,
        "comparison": comparison,
        "temporal_tests": temporal,
        "counterbalance_generalization": cb_generalization,
        "gate_states": gate_states,
        "gate_resets": gate_resets,
        "distilled": distilled,
        "recovery": recovery,
        "recovery_confusion": recovery_confusion,
        "splits": splits,
    }
