"""Standalone PRD3 orchestration; PRD2 code remains a frozen dependency."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from probe_modeling.data import EmptyDatasetError

from . import __version__
from .context_features import context_feature_table
from .context_geometry import context_geometry
from .cross_dataset import cross_dataset_transfer
from .data import load_retrieval_data
from .diagnostics import participant_ab_separation_test, structural_summary
from .distillation import distill_retrieval_rule
from .explicit_models import predict_retrieval_model
from .fitting import fit_joint_retrieval_ladder
from .plotting import make_retrieval_figures
from .recovery import run_retrieval_recovery
from .skill_states import fit_frozen_ab_states
from .tiny_gate import fit_tiny_gate_ladder


@dataclass(frozen=True)
class RetrievalAnalysisConfig:
    seed: int = 42
    train_fraction: float = 0.8
    gate_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(42, 47)))
    gate_epochs: int = 300
    gate_learning_rate: float = 0.02
    recovery_simulations: int = 20
    recovery_noise_sd: float = 0.08
    recovery_regimes: tuple[str, ...] = ("easy", "human", "neighborhood")
    make_plots: bool = True
    show_progress: bool = True
    allow_partial: bool = False
    require_no_context_probe: bool = True
    model_version: str = __version__


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix="retrieval-", suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    sibling = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        shutil.copyfile(temporary, sibling)
        sibling.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _representative_schedule(data: pd.DataFrame) -> pd.DataFrame:
    selection = (
        data.groupby(["dataset", "counterbalance"], dropna=False, sort=False)
        .participant_id.first().reset_index()
    )
    keys = set(zip(selection.dataset, selection.participant_id))
    return data[
        [(dataset, participant) in keys for dataset, participant in zip(data.dataset, data.participant_id)]
    ].reset_index(drop=True)


def _fit_parameter_map(fits: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        row.model: json.loads(row.parameters)
        for row in fits.drop_duplicates("model").itertuples()
    }


def _label_dictionary(data: pd.DataFrame) -> pd.DataFrame:
    """Describe every literal export label and its analysis meaning."""
    rows = data.copy()
    rows["participant_block"] = (
        rows.participant_id.astype(str) + "::" + rows.block_id.astype(str)
    )
    return (
        rows.groupby(
            ["dataset", "raw_context", "retrieval_context", "is_probe", "probe_phase"],
            dropna=False, sort=False,
        )
        .agg(
            n_participants=("participant_id", "nunique"),
            n_blocks=("participant_block", "nunique"),
            n_analysis_sequences=("participant_id", "size"),
        )
        .reset_index()
    )


def _cohort_summary(data: pd.DataFrame) -> pd.DataFrame:
    """One row per design cohort/counterbalance after exclusions."""
    participants = data[
        [
            "dataset", "participant_id", "counterbalance", "probe_exposure_cohort",
            "probe_training_context", "is_probe_control", "probe_entry_context",
        ]
    ].drop_duplicates()
    return (
        participants.groupby(
            [
                "dataset", "counterbalance", "probe_exposure_cohort",
                "probe_training_context", "is_probe_control", "probe_entry_context",
            ],
            dropna=False, sort=False,
        )
        .agg(n_participants=("participant_id", "nunique"))
        .reset_index()
    )


def _structural_diagnostics(states: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("continuous_similarity", "similarity_history"):
        model_prediction = predictions[predictions.model.eq(model)]
        training = model_prediction[model_prediction.split.eq("train")]
        for context in ("Both", "Novel", "Default"):
            context_states = states[states.retrieval_context.eq(context)]
            if context_states.empty:
                continue
            q = float(training.loc[training.retrieval_context.eq(context), "q_A"].mean())
            if not np.isfinite(q):
                q = float(model_prediction.loc[model_prediction.retrieval_context.eq(context), "q_A"].mean())
            diagnostic = participant_ab_separation_test(context_states, q)
            diagnostic["model"] = model
            rows.append(diagnostic)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _continuous_categorical_table(fits: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    selected = ["continuous_similarity", "categorical_similarity"]
    rows = []
    for model in selected:
        fit = fits[fits.model.eq(model)].iloc[0]
        prediction = predictions[
            predictions.model.eq(model)
            & predictions.split.eq("test")
            & ~predictions.retrieval_context.isin(["A", "B"])
        ]
        residual = prediction.log_rt - prediction.predicted_log_rt
        rows.append(
            {
                "model": model, "readout_family": "categorical" if "categorical" in model else "continuous",
                "train_aic": fit.aic, "train_bic": fit.bic,
                "heldout_probe_n": int(len(prediction)),
                "heldout_probe_log_likelihood": float(prediction.observation_log_likelihood.sum()),
                "heldout_probe_rmse": float(np.sqrt(np.mean(residual**2))) if len(residual) else np.nan,
                "mechanistic_claim": (
                    "stochastic expected responsibility" if "categorical" in model
                    else "simultaneous continuous interpolation"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_retrieval_analysis(
    combined_path: str | Path,
    novel_path: str | Path,
    output_dir: str | Path,
    *,
    config: RetrievalAnalysisConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or RetrievalAnalysisConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        if config.show_progress:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] RETRIEVAL | {message}", flush=True)

    sources = {"Combined": Path(combined_path), "Novel": Path(novel_path)}
    source_status, cleaned = [], []
    report("Starting PRD3 similarity-weighted retrieval analysis")
    for dataset, path in sources.items():
        report(f"Loading {dataset}: {path}")
        status = {"dataset": dataset, "path": str(path.resolve()), "sha256": _sha256(path)}
        try:
            frame = load_retrieval_data(path, dataset)
            cleaned.append(frame)
            status.update(
                status="loaded", rows=int(len(frame)), participants=int(frame.participant_id.nunique()),
                probe_contexts=sorted(frame.loc[frame.is_probe, "retrieval_context"].unique().tolist()),
                exposure_cohorts=sorted(frame.probe_exposure_cohort.unique().tolist()),
                excluded_nonmodal_participants=frame.attrs.get("excluded_nonmodal_participants", []),
            )
            report(f"{dataset} ready: {status['participants']} participants, {status['rows']:,} rows")
        except (EmptyDatasetError, pd.errors.EmptyDataError) as error:
            status.update(status="pending_empty", reason=str(error))
            report(f"{dataset} pending: {error}")
        source_status.append(status)
    loaded_datasets = {frame.dataset.iloc[0] for frame in cleaned}
    if "Combined" not in loaded_datasets:
        raise EmptyDatasetError("Combined data are required for PRD3.")
    if "Novel" not in loaded_datasets and not config.allow_partial:
        raise EmptyDatasetError(
            "Novel/third-context data are still empty. Add observations or use allow_partial=True for development only."
        )
    if "Novel" in loaded_datasets and config.require_no_context_probe:
        novel = next(frame for frame in cleaned if frame.dataset.iloc[0] == "Novel")
        missing = {"Novel", "Default"}.difference(set(novel.retrieval_context))
        if missing:
            raise ValueError(f"Third-context data must include Novel and No Context probes; missing {sorted(missing)}")

    state_frames, skill_fit_frames = [], []
    for frame in cleaned:
        dataset = frame.dataset.iloc[0]
        report(f"Estimating frozen A/B experts for {dataset}")
        states, skill_fits = fit_frozen_ab_states(frame, progress=report)
        state_frames.append(states)
        skill_fit_frames.append(skill_fits.assign(dataset=dataset))
    clean = pd.concat(cleaned, ignore_index=True)
    states = pd.concat(state_frames, ignore_index=True)
    skill_fits = pd.concat(skill_fit_frames, ignore_index=True)

    report("Jointly fitting shared retrieval mechanisms")
    fits, predictions, comparison, splits = fit_joint_retrieval_ladder(
        states, seed=config.seed, train_fraction=config.train_fraction, progress=report
    )
    parameter_map = _fit_parameter_map(fits)
    similarity_params = parameter_map["continuous_similarity"]
    evidence = predict_retrieval_model(states, "continuous_similarity", similarity_params)
    evidence = evidence[
        [
            "dataset", "participant_id", "counterbalance", "probe_exposure_cohort",
            "probe_training_context", "is_probe_control", "probe_entry_context", "raw_context",
            "probe_phase", "block_id", "global_sequence", "retrieval_context", "d_A", "d_B",
            "relative_evidence_R", "evidence_A", "evidence_B", "absolute_evidence_E", "q_A",
            "log_rt", "predicted_log_rt",
        ]
    ]
    responsibilities = predictions[
        [
            "dataset", "participant_id", "counterbalance", "probe_exposure_cohort",
            "probe_training_context", "is_probe_control", "probe_entry_context", "raw_context",
            "probe_phase", "block_id", "global_sequence", "retrieval_context", "model", "split",
            "q_A", "q_B", "retrieval_logit", "history_state", "absolute_evidence_E",
            "default_weight", "categorical_posterior_A",
        ]
    ]
    continuous_categorical = _continuous_categorical_table(fits, predictions)
    structural = _structural_diagnostics(states, predictions)
    structural_summaries = structural_summary(structural) if not structural.empty else pd.DataFrame()
    if not structural.empty:
        structural = structural.merge(
            structural_summaries, on=["dataset", "retrieval_context"], how="left", suffixes=("", "_summary")
        )

    transfer_models = ("continuous_similarity", "similarity_history", "similarity_evidence", "categorical_similarity")
    if {"Combined", "Novel"}.issubset(loaded_datasets):
        report("Running Combined -> Novel zero-shot transfer")
        combined_to_novel = pd.concat(
            [cross_dataset_transfer(states, "Combined", "Novel", model) for model in transfer_models],
            ignore_index=True,
        )
        report("Running Novel -> Combined zero-shot transfer")
        novel_to_combined = pd.concat(
            [cross_dataset_transfer(states, "Novel", "Combined", model) for model in transfer_models],
            ignore_index=True,
        )
    else:
        combined_to_novel = pd.DataFrame()
        novel_to_combined = pd.DataFrame()

    if config.gate_seeds:
        report("Fitting shared logistic/1D/2D tiny retrieval gates")
        gate_fits, gate_states = fit_tiny_gate_ladder(
            states, splits, seeds=config.gate_seeds, epochs=config.gate_epochs,
            learning_rate=config.gate_learning_rate, progress=report,
        )
        one_dimensional = gate_states[gate_states.model.eq("recurrent_gate_1d")]
        distilled = distill_retrieval_rule(one_dimensional) if not one_dimensional.empty else pd.DataFrame()
    else:
        gate_fits, gate_states, distilled = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if config.recovery_simulations > 0 and {"Combined", "Novel"}.issubset(loaded_datasets):
        report("Running multi-regime retrieval model recovery")
        recovery, confusion = run_retrieval_recovery(
            _representative_schedule(states),
            regimes=config.recovery_regimes,
            n_simulations=config.recovery_simulations,
            noise_sd=config.recovery_noise_sd,
            seed=config.seed,
            empirical_parameters=parameter_map,
            progress=report,
        )
    else:
        recovery, confusion = pd.DataFrame(), pd.DataFrame()

    features = context_feature_table()
    geometry = context_geometry(beta=1.0)
    label_dictionary = _label_dictionary(clean)
    cohort_summary = _cohort_summary(clean)
    exports = {
        "context_feature_table.csv": features,
        "context_geometry.csv": geometry,
        "retrieval_label_dictionary.csv": label_dictionary,
        "retrieval_cohort_summary.csv": cohort_summary,
        "retrieval_model_fits.csv": fits,
        "retrieval_predictions.csv": predictions,
        "retrieval_responsibilities.csv": responsibilities,
        "relative_absolute_evidence.csv": evidence,
        "continuous_vs_categorical.csv": continuous_categorical,
        "participant_AB_separation_test.csv": structural,
        "combined_to_novel_transfer.csv": combined_to_novel,
        "novel_to_combined_transfer.csv": novel_to_combined,
        "tiny_gate_fits.csv": gate_fits,
        "tiny_gate_states.csv": gate_states,
        "distilled_retrieval_rule.csv": distilled,
        "retrieval_model_recovery.csv": recovery,
        "retrieval_recovery_confusion.csv": confusion,
        # Supporting audit artifacts.
        "retrieval_clean_data.csv": clean,
        "retrieval_AB_states.csv": states,
        "retrieval_AB_fit_parameters.csv": skill_fits,
        "retrieval_participant_splits.csv": splits,
        "retrieval_model_comparison.csv": comparison,
    }
    report(f"Writing {len(exports)} retrieval artifacts")
    for filename, frame in exports.items():
        _write_csv(frame, output / filename)
    run_record = {
        "analysis": "PRD3 similarity-weighted retrieval",
        "config": asdict(config),
        "sources": source_status,
        "joint_claim_available": {"Combined", "Novel"}.issubset(loaded_datasets),
        "no_context_target": "Default=[0,0,0]",
        "actual_export_label_contract": {
            "Maze Training / Maze Test": "A",
            "Melody Training / test melodyA": "B",
            "Mixed Training / Mixed Test": "Both",
            "Space Training / test spaceA": "Novel",
            "No Context Training / No Context Test": "Default",
            "warmup MazeA / warmup melodyA / test warmupA": "excluded",
        },
        "outputs": {filename: int(len(frame)) for filename, frame in exports.items()},
        "frozen_dependencies": {
            "PRD2_preprocessing": True,
            "PRD2_AB_skill_fit": True,
            "original_task_SRN": "unchanged negative control",
        },
        "safeguards": [
            "Context vectors are determined exclusively from task design.",
            "A/B experts and participant intercepts use A/B observations only.",
            "Retrieval models never alter the frozen A/B experts.",
            "Mechanism parameters are shared jointly across Combined and Novel.",
            "Cross-dataset target probe observations fit no retrieval parameter.",
            "Categorical and continuous readouts are compared on individual-sequence likelihood.",
        ],
    }
    with (output / "run_config_retrieval.json").open("w", encoding="utf-8") as handle:
        json.dump(run_record, handle, indent=2)
    if config.make_plots:
        report("Rendering eight PRD3 figures")
        make_retrieval_figures(
            output, features=features, geometry=geometry, predictions=predictions,
            structural=structural, combined_to_novel=combined_to_novel,
            novel_to_combined=novel_to_combined, evidence=evidence,
            gate_states=gate_states, distilled=distilled,
        )
    report("PRD3 retrieval analysis complete")
    return {
        "clean": clean, "states": states, "fits": fits, "predictions": predictions,
        "responsibilities": responsibilities, "evidence": evidence,
        "continuous_categorical": continuous_categorical, "structural": structural,
        "combined_to_novel": combined_to_novel, "novel_to_combined": novel_to_combined,
        "gate_fits": gate_fits, "gate_states": gate_states, "distilled": distilled,
        "recovery": recovery, "confusion": confusion, "splits": splits,
    }
