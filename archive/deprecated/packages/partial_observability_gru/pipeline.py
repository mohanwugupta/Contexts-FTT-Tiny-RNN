"""Factorial partial-observability GRU experiment and artifact contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .analysis.bayes_risk import mean_squared_risk, sampling_squared_risk
from .analysis.dimensionality import dimensionality_summary
from .analysis.distillation import distill_dynamics
from .analysis.hidden_probes import fit_hidden_probes
from .analysis.human_probes import evaluate_human_style_probes
from .analysis.interventions import belief_intervention_table, skill_intervention_table
from .environment.episode import EpisodeConfig, generate_batch
from .models import OBJECTIVES, build_model
from .plotting import make_figures
from .training.trainer import TrainingConfig, evaluate_model, train_model


@dataclass(frozen=True)
class PartialObservabilityConfig:
    uncertainty_levels: tuple[str, ...] = ("perfect", "low", "moderate", "high")
    skill_separations: tuple[str, ...] = ("small", "medium", "large")
    objectives: tuple[str, ...] = OBJECTIVES
    hidden_dims: tuple[int, ...] = (1, 2, 3, 4, 8, 16)
    seeds: tuple[int, ...] = (0,)
    train_episodes: int = 128
    eval_episodes: int = 64
    episode_length: int = 40
    context_dim: int = 2
    epochs: int = 60
    batches_per_epoch: int = 8
    batch_size: int = 32
    learning_rate: float = 0.003
    make_plots: bool = True
    show_progress: bool = True
    human_reference_path: str | None = None
    version: str = __version__

    def validate(self):
        unknown = set(self.objectives).difference(OBJECTIVES)
        if unknown:
            raise ValueError(f"Unknown objectives: {sorted(unknown)}")
        invalid_dims = set(self.hidden_dims).difference({1, 2, 3, 4, 8, 16})
        if invalid_dims:
            raise ValueError(f"Hidden dimensions must use the PRD ladder; got {sorted(invalid_dims)}")
        if min(self.seeds, default=0) < 0:
            raise ValueError("Seeds must be nonnegative.")


def smoke_config() -> PartialObservabilityConfig:
    return PartialObservabilityConfig(
        uncertainty_levels=("perfect", "moderate"), skill_separations=("medium",),
        hidden_dims=(3,), seeds=(0,), epochs=8, batches_per_epoch=3,
        batch_size=12, eval_episodes=12, episode_length=24,
    )


def pilot_config() -> PartialObservabilityConfig:
    return PartialObservabilityConfig(seeds=tuple(range(5)), epochs=50, batches_per_epoch=5)


def full_config() -> PartialObservabilityConfig:
    return PartialObservabilityConfig(seeds=tuple(range(20)), epochs=100, batches_per_epoch=8)


def _write_csv(frame: pd.DataFrame, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _metadata(condition: dict) -> dict:
    return {
        "condition_id": condition["condition_id"], "uncertainty": condition["uncertainty"],
        "skill_separation": condition["skill_separation"], "objective": condition["objective"],
        "hidden_dim": condition["hidden_dim"], "seed": condition["seed"],
    }


def _attach(frame: pd.DataFrame, condition: dict) -> pd.DataFrame:
    result = frame.copy()
    for key, value in reversed(list(_metadata(condition).items())):
        result.insert(0, key, value)
    return result


def _human_reference(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source)
    if not {"retrieval_context", "log_rt"}.issubset(data):
        raise ValueError("Human reference must contain retrieval_context and log_rt.")
    return (
        data[data.retrieval_context.isin(["Both", "Novel", "Default"])]
        .groupby("retrieval_context").log_rt.mean().rename("human_mean_log_rt").reset_index()
        .rename(columns={"retrieval_context": "probe"})
    )


def run_partial_observability_analysis(
    output_dir: str | Path,
    *,
    config: PartialObservabilityConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or smoke_config()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def report(message: str):
        if config.show_progress:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] PARTIAL-GRU | {message}", flush=True)

    condition_rows, histories, performances, ground_truth = [], [], [], []
    alignments, probe_rows, intervention_rows, skill_rows = [], [], [], []
    hidden_rows, distilled_rows, ambiguity_rows, sampling_rows = [], [], [], []
    combinations = [
        (uncertainty, separation, objective, hidden_dim, int(seed))
        for uncertainty in config.uncertainty_levels
        for separation in config.skill_separations
        for objective in config.objectives
        for hidden_dim in config.hidden_dims
        for seed in config.seeds
    ]
    report(f"Starting {len(combinations)} factorial training conditions")
    for number, (uncertainty, separation, objective, hidden_dim, seed) in enumerate(combinations, 1):
        condition = {
            "condition_id": f"U-{uncertainty}_S-{separation}_O-{objective}_H-{hidden_dim}_seed-{seed}",
            "uncertainty": uncertainty, "skill_separation": separation,
            "objective": objective, "hidden_dim": hidden_dim, "seed": seed,
        }
        report(f"Condition {number}/{len(combinations)}: {condition['condition_id']}")
        episode_config = EpisodeConfig(
            length=config.episode_length, context_dim=config.context_dim,
            uncertainty=uncertainty, skill_separation=separation,
        )
        training_config = TrainingConfig(
            epochs=config.epochs, batches_per_epoch=config.batches_per_epoch,
            batch_size=min(config.batch_size, config.train_episodes),
            learning_rate=config.learning_rate,
        )
        model, history = train_model(
            objective, hidden_dim, episode_config, training_config, seed=seed,
            progress=lambda message, prefix=condition["condition_id"]: report(f"{prefix} | {message}"),
        )
        evaluation = generate_batch(config.eval_episodes, episode_config, seed=seed + 900_001)
        evaluated, metrics, hidden = evaluate_model(model, objective, evaluation, seed=seed + 77)
        metadata = _metadata(condition)
        condition_rows.append(
            {
                **metadata, "context_noise_sd": episode_config.context_noise_sd,
                "skill_separation_value": episode_config.skill_separation_value,
                "training_episodes_per_epoch": training_config.batch_size * training_config.batches_per_epoch,
                "epochs": training_config.epochs, "exact_ood_probes_in_training": False,
            }
        )
        histories.append(_attach(history, condition))
        performances.append(pd.DataFrame([{**metadata, "final_train_loss": history.loss.iloc[-1], **metrics}]))
        evaluated_with_meta = _attach(evaluated, condition)
        ground_truth.append(evaluated_with_meta)
        alignments.append(pd.DataFrame([{**metadata, **{key: metrics[key] for key in (
            "posterior_mean_r2", "belief_r2", "posterior_mean_mae", "belief_mae"
        )}}]))

        hidden_flat = hidden.reshape(-1, hidden.shape[-1])
        targets = {
            "S_A": evaluated.skill_a.to_numpy(), "S_B": evaluated.skill_b.to_numpy(),
            "q_star": evaluated.q_star.to_numpy(),
        }
        decoded = fit_hidden_probes(
            hidden_flat, targets, evaluated.episode_id.to_numpy(), seed=seed + 31
        )
        decoded["network_state"] = "trained"
        hidden_rows.append(_attach(decoded, condition))
        torch.manual_seed(seed)
        untrained_model = build_model(objective, episode_config.input_dim, hidden_dim)
        with torch.no_grad():
            untrained_hidden = untrained_model(evaluation.inputs)["hidden"].numpy().reshape(-1, hidden_dim)
        untrained_decoded = fit_hidden_probes(
            untrained_hidden, targets, evaluated.episode_id.to_numpy(), seed=seed + 31
        )
        untrained_decoded["network_state"] = "untrained"
        hidden_rows.append(_attach(untrained_decoded, condition))
        sample_count = min(128, len(hidden_flat))
        subset = np.linspace(0, len(hidden_flat) - 1, sample_count, dtype=int)
        belief = belief_intervention_table(
            model, objective, hidden_flat[subset], evaluated.q_star.to_numpy()[subset],
            evaluated.skill_a.to_numpy()[subset], evaluated.skill_b.to_numpy()[subset],
        )
        intervention_rows.append(_attach(belief, condition))
        skills = skill_intervention_table(
            model, objective, hidden_flat[subset], evaluated.skill_a.to_numpy()[subset],
            evaluated.skill_b.to_numpy()[subset], evaluated.q_star.to_numpy()[subset],
        )
        skill_rows.append(_attach(skills, condition))
        distilled_rows.append(_attach(distill_dynamics(hidden, evaluated), condition))
        probes = _attach(evaluate_human_style_probes(model, objective, evaluation), condition)
        probe_rows.append(probes)
        ambiguity_rows.append(
            probes[probes.probe.isin(["Midpoint", "Both", "Novel", "Default"])].copy()
        )
        normative_values = zip(
            evaluated.skill_a.to_numpy(float), evaluated.skill_b.to_numpy(float),
            evaluated.q_star.to_numpy(float),
        )
        normative_pairs = [
            (mean_squared_risk(a, b, q), sampling_squared_risk(a, b, q))
            for a, b, q in normative_values
        ]
        normative_mean = float(np.mean([pair[0] for pair in normative_pairs]))
        normative_sample = float(np.mean([pair[1] for pair in normative_pairs]))
        sampling_rows.append(
            {
                **metadata, "normative_mean_risk": normative_mean,
                "normative_sampling_risk": normative_sample,
                "network_continuous_mse": metrics["rmse"] ** 2,
                "network_sampled_mse": metrics["sampled_rmse"] ** 2 if np.isfinite(metrics["sampled_rmse"]) else np.nan,
            }
        )

    frames = {
        "conditions": pd.DataFrame(condition_rows),
        "training_history": pd.concat(histories, ignore_index=True),
        "performance": pd.concat(performances, ignore_index=True),
        "bayesian_ground_truth": pd.concat(ground_truth, ignore_index=True),
        "posterior_alignment": pd.concat(alignments, ignore_index=True),
        "hidden_probes": pd.concat(hidden_rows, ignore_index=True),
        "belief_interventions": pd.concat(intervention_rows, ignore_index=True),
        "skill_interventions": pd.concat(skill_rows, ignore_index=True),
        "human_probes": pd.concat(probe_rows, ignore_index=True),
        "ambiguity": pd.concat(ambiguity_rows, ignore_index=True),
        "distilled": pd.concat(distilled_rows, ignore_index=True),
        "continuous_sampling": pd.DataFrame(sampling_rows),
    }
    human = _human_reference(config.human_reference_path)
    if not human.empty:
        frames["human_probes"] = frames["human_probes"].merge(human, on="probe", how="left")
    frames["dimensionality"] = dimensionality_summary(frames["performance"], frames["hidden_probes"])
    frames["objective_comparison"] = (
        frames["performance"].groupby(["uncertainty", "skill_separation", "objective"], sort=False)
        .agg(
            n_networks=("condition_id", "nunique"), rmse=("rmse", "mean"),
            posterior_mean_r2=("posterior_mean_r2", "mean"), belief_r2=("belief_r2", "mean"),
            sampled_rmse=("sampled_rmse", "mean"), objective_loss=("objective_loss", "mean"),
            context_accuracy=("context_accuracy", "mean"),
            mean_predicted_sigma=("mean_predicted_sigma", "mean"),
            ambiguous_predicted_sigma=("ambiguous_predicted_sigma", "mean"),
        ).reset_index()
    )
    exports = {
        "training_conditions.csv": frames["conditions"],
        "network_performance.csv": frames["performance"],
        "ambiguity_behavior.csv": frames["ambiguity"],
        "bayesian_ground_truth.csv": frames["bayesian_ground_truth"],
        "posterior_alignment.csv": frames["posterior_alignment"],
        "dimensionality_comparison.csv": frames["dimensionality"],
        "hidden_state_probes.csv": frames["hidden_probes"],
        "belief_interventions.csv": frames["belief_interventions"],
        "skill_interventions.csv": frames["skill_interventions"],
        "objective_comparison.csv": frames["objective_comparison"],
        "continuous_vs_sampling.csv": frames["continuous_sampling"],
        "human_style_probe_predictions.csv": frames["human_probes"],
        "distilled_dynamics.csv": frames["distilled"],
        "training_history.csv": frames["training_history"],
    }
    report(f"Writing {len(exports)} CSV artifacts")
    for filename, frame in exports.items():
        _write_csv(frame, output / filename)
    record = {
        "analysis": "Emergence of graded skill retrieval under partial observability",
        "config": asdict(config), "n_conditions": len(combinations),
        "target_leakage_safeguard": "inputs[t] contain y[t-1], never y[t]",
        "training_probe_contract": "Only noisy A/B observations; exact Both/Novel/Default probes absent",
        "human_data_role": "external validation only; never a training target",
        "architecture_safeguard": "Primary GRU receives no true context, skills, posterior, experts, or mixture weights",
        "outputs": {name: int(len(frame)) for name, frame in exports.items()},
    }
    (output / "run_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if config.make_plots:
        report("Rendering eight partial-observability figures")
        make_figures(
            output, performance=frames["performance"], hidden_probes=frames["hidden_probes"],
            belief_interventions=frames["belief_interventions"],
            continuous_sampling=frames["continuous_sampling"], human_probes=frames["human_probes"],
        )
    report("Partial-observability GRU analysis complete")
    return frames
