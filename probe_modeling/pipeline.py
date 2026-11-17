"""End-to-end, reproducible probe-analysis orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import tempfile
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch

from . import __version__
from .data import EmptyDatasetError, load_probe_data
from .fit import fit_model_ladder, grouped_participant_split
from .gate_models import fit_gate_ladder
from .metrics import compute_behavioral_probe_metrics, compute_post_probe_aftereffects
from .plotting import make_required_figures
from .recovery import recovery_identifiability, run_model_recovery
from .skill_states import fit_counterfactual_states


@dataclass(frozen=True)
class ProbeAnalysisConfig:
    seed: int = 42
    train_fraction: float = 0.8
    neural_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(42, 62)))
    gate_epochs: int = 300
    gate_learning_rate: float = 0.02
    recovery_simulations: int = 20
    recovery_noise_sd: float = 0.05
    recovery_threshold: float = 0.8
    make_plots: bool = True
    strict_empty_datasets: bool = False
    model_version: str = __version__


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    # Cloud-synced workspaces can expose a partially written CSV for minutes.
    # Render locally, bulk-copy to a sibling temporary file, then atomically
    # replace the destination.
    with tempfile.NamedTemporaryFile(prefix="probe-modeling-", suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    sibling = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        shutil.copyfile(temporary, sibling)
        sibling.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _confusion_to_long(
    confusion: pd.DataFrame,
    dataset: str,
    n_simulations: int,
) -> pd.DataFrame:
    return (
        confusion.rename_axis(index="generating_mechanism", columns="selected_model")
        .stack(future_stack=True)
        .rename("proportion")
        .reset_index()
        .assign(dataset=dataset, n_simulations=int(n_simulations))
    )


def _neural_comparison(gate_fits: pd.DataFrame) -> pd.DataFrame:
    if gate_fits.empty:
        return pd.DataFrame()
    rows = []
    for _, row in gate_fits.iterrows():
        rows.append(
            {
                "dataset": row.dataset,
                "model": f"{row.model}_h{int(row.hidden_dim)}",
                "split": "test",
                "region": "all",
                "bic": np.nan,
                "n": row.get("test_n", np.nan),
                "rmse": row.get("test_rmse", np.nan),
                "mae": row.get("test_mae", np.nan),
                "nll": row.get("test_nll", np.nan),
                "correlation": row.get("test_correlation", np.nan),
                "model_family": "neural_ceiling",
                "seed": row.seed,
                "hidden_dim": row.hidden_dim,
            }
        )
    return pd.DataFrame(rows)


def run_probe_analysis(
    input_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    config: Optional[ProbeAnalysisConfig] = None,
) -> dict[str, pd.DataFrame]:
    """Run all explicit, recovery, neural, intervention, and export stages."""
    config = config or ProbeAnalysisConfig()
    set_global_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = []
    source_status = []
    for dataset, source in input_paths.items():
        source = Path(source)
        status = {
            "dataset": dataset,
            "path": str(source.resolve()),
            "sha256": _file_sha256(source) if source.exists() else None,
        }
        try:
            frame = load_probe_data(source, dataset=dataset, cohort_only=True)
            excluded = frame.attrs.get("excluded_nonmodal_participants", [])
            normalized.append(frame)
            status.update(
                {
                    "status": "loaded",
                    "rows": int(len(frame)),
                    "participants": int(frame.participant_id.nunique()),
                    "excluded_nonmodal_participants": excluded,
                }
            )
        except EmptyDatasetError as error:
            status.update({"status": "skipped_empty", "reason": str(error)})
            if config.strict_empty_datasets:
                raise
        source_status.append(status)
    if not normalized:
        raise EmptyDatasetError("None of the supplied datasets contained analyzable observations.")
    clean = pd.concat(normalized, ignore_index=True)

    counterfactuals, skill_fits = fit_counterfactual_states(clean, cross_talk=0.0)
    behavioral = compute_behavioral_probe_metrics(counterfactuals)
    aftereffects = compute_post_probe_aftereffects(counterfactuals)
    splits = grouped_participant_split(
        counterfactuals,
        seed=config.seed,
        train_fraction=config.train_fraction,
    )
    explicit_fits, predictions, explicit_comparison, splits = fit_model_ladder(
        counterfactuals,
        seed=config.seed,
        splits=splits,
    )
    explicit_fits["model_family"] = "explicit"
    explicit_comparison["model_family"] = "explicit"

    if config.neural_seeds:
        gate_fits, gate_states, cue_interventions, state_interventions = fit_gate_ladder(
            counterfactuals,
            splits,
            seeds=config.neural_seeds,
            epochs=config.gate_epochs,
            learning_rate=config.gate_learning_rate,
            model_version=config.model_version,
        )
    else:
        gate_fits = pd.DataFrame()
        gate_states = pd.DataFrame()
        cue_interventions = pd.DataFrame()
        state_interventions = pd.DataFrame()
    if not gate_fits.empty:
        gate_fits["model_family"] = "neural_ceiling"
    model_fits = pd.concat([explicit_fits, gate_fits], ignore_index=True, sort=False)
    comparison = pd.concat(
        [explicit_comparison, _neural_comparison(gate_fits)], ignore_index=True, sort=False
    )

    recovery_frames = []
    confusion_frames = []
    identifiability_frames = []
    for dataset, schedule in counterfactuals.groupby("dataset", sort=False):
        results, confusion = run_model_recovery(
            schedule,
            n_simulations=config.recovery_simulations,
            noise_sd=config.recovery_noise_sd,
            seed=config.seed,
        )
        results["dataset"] = dataset
        recovery_frames.append(results)
        confusion_frames.append(
            _confusion_to_long(confusion, dataset, config.recovery_simulations)
        )
        identified = recovery_identifiability(results, threshold=config.recovery_threshold)
        identified["dataset"] = dataset
        identifiability_frames.append(identified)
    recovery = pd.concat(recovery_frames, ignore_index=True)
    recovery_confusion = pd.concat(confusion_frames, ignore_index=True)
    identifiability = pd.concat(identifiability_frames, ignore_index=True)

    exports = {
        "clean_probe_data.csv": clean,
        "counterfactual_AB_states.csv": counterfactuals,
        "behavioral_probe_metrics.csv": behavioral,
        "model_fits.csv": model_fits,
        "model_predictions.csv": predictions,
        "model_comparison.csv": comparison,
        "gate_states.csv": gate_states,
        "cue_intervention_results.csv": cue_interventions,
        "state_intervention_results.csv": state_interventions,
        "post_probe_aftereffects.csv": aftereffects,
        "model_recovery_results.csv": recovery,
        "model_recovery_confusion.csv": recovery_confusion,
        "participant_splits.csv": splits,
        "counterfactual_fit_parameters.csv": skill_fits,
        "model_recovery_identifiability.csv": identifiability,
    }
    for filename, frame in exports.items():
        _write_csv(frame, output_dir / filename)

    run_record = {
        "analysis_tier": (
            "final_config"
            if len(config.neural_seeds) >= 20 and config.recovery_simulations >= 20
            else "verification_only"
        ),
        "config": asdict(config),
        "sources": source_status,
        "outputs": {name: int(len(frame)) for name, frame in exports.items()},
        "notes": [
            "A/B counterfactual fitting excluded every probe observation.",
            "Neural parameters were fixed while recurrent state evolved through each participant timeline.",
            "Mechanistic interpretation is gated by model_recovery_identifiability.csv.",
            "Mixture indices are masked when counterfactual A/B RTs differ by less than 2%.",
        ],
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2)

    if config.make_plots:
        make_required_figures(
            output_dir,
            counterfactuals=counterfactuals,
            behavioral_metrics=behavioral,
            predictions=predictions,
            aftereffects=aftereffects,
            gate_states=gate_states,
            cue_interventions=cue_interventions,
            recovery_confusion=recovery_confusion,
            comparison=comparison,
        )
    return {
        "clean": clean,
        "counterfactuals": counterfactuals,
        "behavioral": behavioral,
        "model_fits": model_fits,
        "predictions": predictions,
        "comparison": comparison,
        "gate_states": gate_states,
        "cue_interventions": cue_interventions,
        "state_interventions": state_interventions,
        "aftereffects": aftereffects,
        "recovery": recovery,
        "recovery_confusion": recovery_confusion,
        "identifiability": identifiability,
        "splits": splits,
    }
