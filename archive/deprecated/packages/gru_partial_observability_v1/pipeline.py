"""Two-condition v1 proof-of-concept analysis and exact artifact contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analysis.ambiguity_tests import DEFAULT_Q_LEVELS, evaluate_ambiguity, generate_ambiguity_batch
from .analysis.bayes_risk import risk_table
from .analysis.hidden_probes import fit_hidden_probes
from .analysis.interventions import belief_intervention
from .analysis.posterior_mean import compare_conditions
from .environment.contexts import bayes_accuracy
from .environment.episodes import V1EpisodeConfig, generate_batch
from .plotting import make_v1_figures
from .training.trainer import TrainingConfig, evaluate_predictor, train_predictor


@dataclass(frozen=True)
class V1AnalysisConfig:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    hidden_dim: int = 4
    perfect_noise_sd: float = 0.0
    uncertain_noise_sd: float = 1.1
    context_persistence: float = 0.85
    outcome_noise_sd: float = 0.02
    train_episode_length: int = 60
    epochs: int = 65
    batches_per_epoch: int = 6
    batch_size: int = 64
    learning_rate: float = 0.006
    eval_episodes: int = 96
    ambiguity_q_levels: tuple[float, ...] = DEFAULT_Q_LEVELS
    ambiguity_episodes_per_q: int = 24
    ambiguity_history_length: int = 30
    intervention_decode_threshold: float = 0.2
    make_plots: bool = True
    show_progress: bool = True
    version: str = __version__

    def validate(self):
        if self.hidden_dim != 4:
            raise ValueError("V1 primary hidden_dim is fixed at 4.")
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if self.uncertain_noise_sd <= 0:
            raise ValueError("Uncertain condition requires positive context noise.")
        if not all(0 < q < 1 for q in self.ambiguity_q_levels):
            raise ValueError("Diagnostic q levels must lie strictly inside (0,1).")


def smoke_config() -> V1AnalysisConfig:
    return V1AnalysisConfig(
        seeds=(0,), train_episode_length=16, epochs=4, batches_per_epoch=2,
        batch_size=10, eval_episodes=8, ambiguity_episodes_per_q=3,
        ambiguity_history_length=12,
    )


def mvp_config() -> V1AnalysisConfig:
    return V1AnalysisConfig()


def final_config() -> V1AnalysisConfig:
    return replace(V1AnalysisConfig(), seeds=tuple(range(20)), epochs=100)


def _write_csv(frame: pd.DataFrame, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _attach(frame: pd.DataFrame, condition: str, seed: int, diagnostic_set_id: str | None = None):
    result = frame.copy()
    result.insert(0, "seed", int(seed))
    result.insert(0, "condition", condition)
    if diagnostic_set_id is not None:
        result.insert(2, "diagnostic_set_id", diagnostic_set_id)
    return result


def _proof_status(seed_summary: pd.DataFrame, comparison: pd.DataFrame, n_seeds: int) -> tuple[str, dict]:
    uncertain = seed_summary[seed_summary.condition.eq("Uncertain")]
    perfect = seed_summary[seed_summary.condition.eq("Perfect")]
    basic_learning = bool(
        not perfect.empty
        and (perfect.model_mse < perfect.global_mean_mse).mean() >= 0.8
        and (perfect.model_mse < perfect.previous_outcome_mse).mean() >= 0.5
    )
    positive_alignment = bool(not uncertain.empty and uncertain.posterior_mean_r2.mean() > 0)
    uncertainty_advantage = bool(
        not comparison.empty and comparison.uncertain_minus_perfect.mean() > 0
    )
    belief_decodable = bool(
        not uncertain.empty and uncertain.q_heldout_r2.mean() >= 0.2
    )
    sufficient_mvp_seeds = n_seeds >= 5
    checks = {
        "basic_clear_context_learning": basic_learning,
        "uncertain_posterior_mean_alignment_positive": positive_alignment,
        "uncertain_exceeds_perfect_on_identical_diagnostics": uncertainty_advantage,
        "q_star_decodable_in_uncertain_networks": belief_decodable,
        "at_least_five_seeds": sufficient_mvp_seeds,
    }
    if all(checks.values()):
        status = "Strong support: v1 proof of concept succeeded; proceed to v2 replication/extension."
    elif positive_alignment and uncertainty_advantage:
        status = "Partial support: behavioral averaging emerged, but the full representational/MVP criterion was not met."
    else:
        status = "Not yet supported: do not proceed to v2 without diagnosing the failed v1 criteria."
    return status, checks


def _write_report(path: Path, status: str, checks: dict, frames: dict):
    comparison = frames["comparison"]
    uncertain_advantage = (
        float(comparison.uncertain_minus_perfect.mean()) if not comparison.empty else np.nan
    )
    uncertain = frames["alignment"][frames["alignment"].condition.eq("Uncertain")]
    perfect = frames["alignment"][frames["alignment"].condition.eq("Perfect")]
    lines = [
        "# V1 proof-of-concept report", "", f"**Decision:** {status}", "", "## Prespecified checks", "",
    ]
    lines.extend([f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}" for name, passed in checks.items()])
    lines.extend(
        [
            "", "## Primary estimates", "",
            f"- Mean posterior-mean R², Perfect: {perfect.posterior_mean_r2.mean():.4f}",
            f"- Mean posterior-mean R², Uncertain: {uncertain.posterior_mean_r2.mean():.4f}",
            f"- Mean Uncertain − Perfect difference: {uncertain_advantage:.4f}",
            "", "This report is generated from synthetic data only. No human RT, Both, or Novel data were used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_v1_analysis(output_dir: str | Path, *, config: V1AnalysisConfig | None = None):
    config = config or smoke_config()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def report(message: str):
        if config.show_progress:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] V1-GRU | {message}", flush=True)

    conditions = (("Perfect", config.perfect_noise_sd), ("Uncertain", config.uncertain_noise_sd))
    condition_rows, performance_rows, ambiguity_frames = [], [], []
    ground_truth_frames, alignment_rows, responsibility_frames = [], [], []
    probe_frames, intervention_frames, histories, evaluation_frames = [], [], [], []
    total = len(config.seeds) * len(conditions)
    counter = 0
    report(f"Starting v1 Perfect-versus-Uncertain experiment ({total} networks)")
    for seed in config.seeds:
        diagnostic_set_id = f"ambiguity-seed-{seed}"
        diagnostic = generate_ambiguity_batch(
            config.ambiguity_q_levels,
            episodes_per_level=config.ambiguity_episodes_per_q,
            history_length=config.ambiguity_history_length,
            diagnostic_noise_sd=config.uncertain_noise_sd,
            persistence=config.context_persistence,
            outcome_noise_sd=config.outcome_noise_sd,
            seed=50_000 + int(seed),
        )
        ground_truth_added = False
        for condition, context_noise in conditions:
            counter += 1
            report(f"Network {counter}/{total}: {condition}, seed {seed}")
            episode_config = V1EpisodeConfig(
                length=config.train_episode_length, context_noise_sd=context_noise,
                context_persistence=config.context_persistence,
                outcome_noise_sd=config.outcome_noise_sd,
            )
            training_config = TrainingConfig(
                hidden_dim=config.hidden_dim, epochs=config.epochs,
                batches_per_epoch=config.batches_per_epoch, batch_size=config.batch_size,
                learning_rate=config.learning_rate,
            )
            model, history = train_predictor(
                episode_config, training_config, seed=int(seed),
                progress=lambda message, label=f"{condition} seed {seed}": report(f"{label} | {message}"),
            )
            training_mean = float(history.training_target_mean.iloc[-1])
            histories.append(_attach(history, condition, seed))
            evaluation = generate_batch(
                config.eval_episodes, episode_config, seed=100_000 + int(seed)
            )
            evaluated, metrics, hidden = evaluate_predictor(model, evaluation, training_mean)
            evaluated = _attach(evaluated, condition, seed)
            evaluation_frames.append(evaluated)
            condition_rows.append(
                {
                    "condition": condition, "seed": int(seed), "context_noise_sd": context_noise,
                    "bayes_context_accuracy": bayes_accuracy(context_noise),
                    "hidden_dim": config.hidden_dim, "epochs": config.epochs,
                    "training_target_mean": training_mean,
                    "inputs": "x_t|y_t_minus_1|previous_outcome_mask|episode_start",
                    "auxiliary_losses": False, "human_data_used": False,
                }
            )
            performance_rows.append({"condition": condition, "seed": int(seed), **metrics})

            ambiguity, alignment, ambiguity_hidden = evaluate_ambiguity(model, diagnostic)
            ambiguity = _attach(ambiguity, condition, seed, diagnostic_set_id)
            ambiguity_frames.append(ambiguity)
            responsibility_frames.append(
                ambiguity[[
                    "condition", "seed", "diagnostic_set_id", "episode_id", "q_star",
                    "q_implied_unclipped", "skill_a", "skill_b", "predicted_y", "posterior_mean",
                ]].copy()
            )
            alignment_rows.append({
                "condition": condition, "seed": int(seed), "diagnostic_set_id": diagnostic_set_id,
                **alignment,
            })
            if not ground_truth_added:
                ground_truth_frames.append(
                    ambiguity[[
                        "seed", "diagnostic_set_id", "episode_id", "q_star",
                        "diagnostic_q_requested", "diagnostic_x", "skill_a", "skill_b",
                        "posterior_mean",
                    ]].drop_duplicates().copy()
                )
                ground_truth_added = True

            hidden_flat = hidden.reshape(-1, config.hidden_dim)
            regular_targets = {
                "S_A": evaluated.skill_a.to_numpy(float),
                "S_B": evaluated.skill_b.to_numpy(float),
                "q_star": evaluated.q_star.to_numpy(float),
            }
            probes = fit_hidden_probes(
                hidden_flat, regular_targets, evaluated.episode_id.to_numpy(), seed=int(seed) + 73
            )
            probes = _attach(probes, condition, seed)
            probe_frames.append(probes)
            q_r2 = float(probes.loc[probes.target.eq("q_star"), "heldout_r2"].iloc[0])
            intervention = belief_intervention(
                model, ambiguity_hidden, ambiguity.q_star.to_numpy(float),
                ambiguity.skill_a.to_numpy(float), ambiguity.skill_b.to_numpy(float),
                heldout_q_r2=q_r2, minimum_decode_r2=config.intervention_decode_threshold,
            )
            intervention_frames.append(_attach(intervention, condition, seed))

    frames = {
        "conditions": pd.DataFrame(condition_rows),
        "performance": pd.DataFrame(performance_rows),
        "ambiguity": pd.concat(ambiguity_frames, ignore_index=True),
        "ground_truth": pd.concat(ground_truth_frames, ignore_index=True),
        "alignment": pd.DataFrame(alignment_rows),
        "responsibility": pd.concat(responsibility_frames, ignore_index=True),
        "probes": pd.concat(probe_frames, ignore_index=True),
        "interventions": pd.concat(intervention_frames, ignore_index=True),
        "training_history": pd.concat(histories, ignore_index=True),
        "evaluation": pd.concat(evaluation_frames, ignore_index=True),
        "risk": risk_table(config.ambiguity_q_levels),
    }
    frames["comparison"] = compare_conditions(frames["alignment"])
    q_probe = (
        frames["probes"][frames["probes"].target.eq("q_star")]
        [["condition", "seed", "heldout_r2"]]
        .rename(columns={"heldout_r2": "q_heldout_r2"})
    )
    frames["seed_summary"] = (
        frames["performance"].merge(frames["alignment"], on=["condition", "seed"])
        .merge(q_probe, on=["condition", "seed"])
    )
    status, checks = _proof_status(frames["seed_summary"], frames["comparison"], len(config.seeds))
    exports = {
        "training_conditions.csv": frames["conditions"],
        "network_performance.csv": frames["performance"],
        "ambiguity_behavior.csv": frames["ambiguity"],
        "bayesian_ground_truth.csv": frames["ground_truth"],
        "posterior_mean_alignment.csv": frames["alignment"],
        "implied_responsibility.csv": frames["responsibility"],
        "perfect_vs_uncertain.csv": frames["comparison"],
        "continuous_vs_sampling.csv": frames["risk"],
        "hidden_state_probes.csv": frames["probes"],
        "belief_interventions.csv": frames["interventions"],
        "seed_summary.csv": frames["seed_summary"],
    }
    report(f"Writing {len(exports)} required v1 CSV artifacts")
    for filename, frame in exports.items():
        _write_csv(frame, output / filename)
    run_record = {
        "analysis": "PRD v1 emergent averaging under partial observability",
        "config": asdict(config), "proof_of_concept_decision": status,
        "prespecified_checks": checks,
        "gru_inputs": ["x_t", "y_t_minus_1", "previous_outcome_mask", "episode_start"],
        "forbidden_inputs": ["z_t", "S_A", "S_B", "P(A)", "current_y_t"],
        "objective": "deterministic MSE only", "auxiliary_losses": False,
        "human_data_used": False, "same_ambiguity_sets_across_conditions": True,
        "outputs": {name: int(len(frame)) for name, frame in exports.items()},
    }
    (output / "run_config_v1.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    _write_report(output / "proof_of_concept_report.md", status, checks, frames)
    if config.make_plots:
        report("Rendering seven v1 figures")
        example = frames["evaluation"][
            frames["evaluation"].condition.eq("Uncertain")
            & frames["evaluation"].seed.eq(config.seeds[0])
        ]
        make_v1_figures(
            output, example_episode=example, ambiguity=frames["ambiguity"],
            comparison=frames["comparison"], risk=frames["risk"],
            probes=frames["probes"], interventions=frames["interventions"],
        )
    report(f"V1 analysis complete | {status}")
    frames["status"] = pd.DataFrame([{"decision": status, **checks}])
    return frames
