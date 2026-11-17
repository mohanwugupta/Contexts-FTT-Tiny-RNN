"""Staged Perfect-versus-Uncertain Elman-SRN proof-of-concept pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analysis.ambiguity_tests import (
    DEFAULT_Q_LEVELS,
    evaluate_ambiguity,
    generate_ambiguity_batch,
)
from .analysis.bayes_risk import risk_table
from .analysis.dimensionality import run_architecture_grid
from .analysis.hidden_probes import fit_hidden_probes
from .analysis.interventions import belief_intervention
from .analysis.posterior_alignment import compare_conditions
from .environment.contexts import bayes_accuracy
from .environment.episodes import EpisodeConfig, generate_batch
from .training.trainer import TrainingConfig, evaluate_predictor, train_predictor


@dataclass(frozen=True)
class AnalysisConfig:
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
    behavioral_alignment_threshold: float = 0.5
    baseline_win_fraction: float = 0.6
    run_dimensionality: bool = True
    dimensionality_hidden_dims: tuple[int, ...] = (1, 2, 3, 4, 6, 12)
    dimensionality_seeds: tuple[int, ...] | None = None
    run_gru_control: bool = False
    control_seeds: tuple[int, ...] | None = None
    force_staged_analyses: bool = False
    make_plots: bool = True
    show_progress: bool = True
    profile: str = "mvp"
    mvp_gate_record: str | None = None
    version: str = __version__

    def validate(self):
        if self.hidden_dim != 4:
            raise ValueError("The primary v1 Elman SRN is prespecified at hidden_dim=4.")
        if not self.seeds:
            raise ValueError("At least one primary seed is required.")
        if self.uncertain_noise_sd <= 0:
            raise ValueError("Uncertain context must have positive cue noise.")
        if not 0.5 <= self.context_persistence <= 1.0:
            raise ValueError("context_persistence must be in [0.5, 1].")
        if not all(0.0 < q < 1.0 for q in self.ambiguity_q_levels):
            raise ValueError("Ambiguity q levels must lie strictly in (0,1).")
        if not self.dimensionality_hidden_dims or min(self.dimensionality_hidden_dims) < 1:
            raise ValueError("Dimensionality values must be positive.")
        if self.profile == "final":
            if not self.mvp_gate_record:
                raise ValueError(
                    "The final profile requires mvp_gate_record pointing to the successful "
                    "MVP run_config_v1.json."
                )
            record_path = Path(self.mvp_gate_record)
            if not record_path.is_file():
                raise ValueError(f"MVP gate record does not exist: {record_path}")
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not read MVP gate record: {record_path}") from error
            if not bool(record.get("behavioral_stage_gate_passed")):
                raise ValueError(
                    "The supplied MVP record did not pass the behavioral stage gate; "
                    "the 20-seed final run is intentionally blocked."
                )


def smoke_config() -> AnalysisConfig:
    return AnalysisConfig(
        seeds=(0,),
        train_episode_length=14,
        epochs=3,
        batches_per_epoch=2,
        batch_size=10,
        eval_episodes=8,
        ambiguity_episodes_per_q=2,
        ambiguity_history_length=10,
        run_dimensionality=False,
        profile="smoke",
    )


def mvp_config() -> AnalysisConfig:
    return AnalysisConfig()


def final_config() -> AnalysisConfig:
    return replace(
        AnalysisConfig(), seeds=tuple(range(20)), dimensionality_seeds=tuple(range(20)),
        epochs=100, profile="final",
    )


def _write_csv(frame: pd.DataFrame, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_text(text: str, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _attach(frame: pd.DataFrame, condition: str, seed: int, **columns):
    result = frame.copy()
    for name, value in reversed(list(columns.items())):
        result.insert(0, name, value)
    result.insert(0, "seed", int(seed))
    result.insert(0, "condition", condition)
    return result


def _not_run(stage: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame([{"status": "not_run", "stage": stage, "reason": reason}])


def _behavior_gate(performance: pd.DataFrame, alignment: pd.DataFrame, comparison: pd.DataFrame,
                   config: AnalysisConfig) -> tuple[bool, dict[str, bool]]:
    uncertain_performance = performance[performance.condition.eq("Uncertain")]
    uncertain_alignment = alignment[alignment.condition.eq("Uncertain")]
    beats_baselines = np.asarray([
        (uncertain_performance.model_mse < uncertain_performance.global_mean_mse).mean(),
        (uncertain_performance.model_mse < uncertain_performance.previous_outcome_mse).mean(),
        (uncertain_performance.model_mse < uncertain_performance.context_only_mse).mean(),
    ])
    checks = {
        "uncertain_srn_beats_simple_baselines": bool(
            len(uncertain_performance) and np.all(beats_baselines >= config.baseline_win_fraction)
        ),
        "uncertain_posterior_mean_alignment": bool(
            len(uncertain_alignment)
            and uncertain_alignment.posterior_mean_r2.mean() >= config.behavioral_alignment_threshold
        ),
        "uncertain_exceeds_perfect_on_identical_diagnostics": bool(
            len(comparison) and comparison.uncertain_minus_perfect.mean() > 0.0
        ),
    }
    return all(checks.values()), checks


def _decision(
    core_success: bool,
    behavior_checks: dict[str, bool],
    probes: pd.DataFrame,
    comparison: pd.DataFrame,
    control: pd.DataFrame,
    n_seeds: int,
) -> tuple[str, str, dict[str, bool]]:
    completed_probes = probes[probes.status.eq("completed")] if "status" in probes else probes
    uncertain_probes = completed_probes[completed_probes.condition.eq("Uncertain")] \
        if "condition" in completed_probes else pd.DataFrame()
    target_r2 = uncertain_probes.groupby("target").heldout_r2.mean() \
        if not uncertain_probes.empty else pd.Series(dtype=float)
    representation_success = bool(
        {"S_A", "S_B", "q_star"}.issubset(target_r2.index)
        and target_r2[["S_A", "S_B", "q_star"]].min() >= 0.2
    )
    checks = {
        **behavior_checks,
        "S_A_S_B_q_star_decodable": representation_success,
        "at_least_five_seeds": n_seeds >= 5,
    }
    if core_success and representation_success and n_seeds >= 5:
        return (
            "A",
            "Strong support: the SRN proof of concept succeeded; proceeding to v2 is supported.",
            checks,
        )
    if core_success:
        return (
            "B",
            "Behavioral averaging emerged, but the full latent-representation criterion was not met.",
            checks,
        )
    if not comparison.empty:
        perfect = comparison.posterior_mean_alignment_perfect.mean()
        uncertain = comparison.posterior_mean_alignment_uncertain.mean()
        if perfect >= 0.5 and abs(uncertain - perfect) <= 0.05:
            return (
                "C",
                "Perfect and Uncertain SRNs averaged similarly; generic interpolation remains plausible.",
                checks,
            )
    completed_control = control[control.status.eq("completed")] if "status" in control else control
    if not completed_control.empty:
        uncertain = completed_control[completed_control.condition.eq("Uncertain")]
        by_architecture = uncertain.groupby("architecture").posterior_mean_r2.mean()
        if by_architecture.get("GRU", -np.inf) >= 0.5 and by_architecture.get("SRN", np.inf) < 0.5:
            return (
                "D",
                "The uncertainty-trained SRN failed while the matched GRU succeeded.",
                checks,
            )
    return (
        "E",
        "The SRN proof of concept is not yet supported; diagnose before proceeding to v2.",
        checks,
    )


def _write_report(path: Path, result_code: str, decision: str, checks: dict, frames: dict):
    alignment = frames["posterior_mean_alignment"]
    comparison = frames["perfect_vs_uncertain"]
    perfect = alignment[alignment.condition.eq("Perfect")]
    uncertain = alignment[alignment.condition.eq("Uncertain")]
    difference = float(comparison.uncertain_minus_perfect.mean()) if not comparison.empty else np.nan
    lines = [
        "# SRN v1 proof-of-concept report",
        "",
        f"**Prespecified result: {result_code}. {decision}**",
        "",
        "## Decision checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] {name.replace('_', ' ')}"
        for name, passed in checks.items()
    )
    lines.extend([
        "",
        "## Primary estimates",
        "",
        f"- Mean posterior-mean R², Perfect: {perfect.posterior_mean_r2.mean():.4f}",
        f"- Mean posterior-mean R², Uncertain: {uncertain.posterior_mean_r2.mean():.4f}",
        f"- Mean Uncertain − Perfect R²: {difference:.4f}",
        "",
        "This synthetic proof of concept used a plain Elman SRN with deterministic MSE training. "
        "No human Mixed/Novel outcomes, explicit posterior, expert heads, or mixture loss entered training.",
    ])
    _write_text("\n".join(lines) + "\n", path)


def run_analysis(output_dir: str | Path, *, config: AnalysisConfig | None = None):
    config = config or smoke_config()
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    def report(message: str):
        if config.show_progress:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(f"[{stamp}] SRN-V1 | {message}", flush=True)

    conditions = (("Perfect", config.perfect_noise_sd),
                  ("Uncertain", config.uncertain_noise_sd))
    condition_rows = []
    network_rows = []
    baseline_rows = []
    ambiguity_frames = []
    ground_truth_frames = []
    alignment_rows = []
    responsibility_frames = []
    evaluation_frames = []
    bundles = []
    total = len(config.seeds) * len(conditions)
    report(f"Starting primary Elman-SRN comparison ({total} networks)")
    counter = 0
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
        for condition_index, (condition, context_noise) in enumerate(conditions):
            counter += 1
            report(f"Primary network {counter}/{total}: {condition}, seed {seed}")
            episode_config = EpisodeConfig(
                length=config.train_episode_length,
                context_noise_sd=context_noise,
                context_persistence=config.context_persistence,
                outcome_noise_sd=config.outcome_noise_sd,
            )
            training_config = TrainingConfig(
                hidden_dim=config.hidden_dim,
                epochs=config.epochs,
                batches_per_epoch=config.batches_per_epoch,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
            )
            model, training_result = train_predictor(
                episode_config,
                training_config,
                seed=int(seed),
                architecture="srn",
                progress=lambda message, label=f"{condition} seed {seed}": report(
                    f"{label} | {message}"
                ),
            )
            evaluation = generate_batch(
                config.eval_episodes, episode_config, seed=100_000 + int(seed)
            )
            evaluated, metrics, hidden = evaluate_predictor(
                model, evaluation, training_result.baselines
            )
            evaluation_frames.append(_attach(evaluated, condition, seed))
            condition_rows.append({
                "condition": condition,
                "seed": int(seed),
                "architecture": "Elman SRN",
                "context_noise_sd": context_noise,
                "bayes_context_accuracy": bayes_accuracy(context_noise),
                "context_persistence": config.context_persistence,
                "outcome_noise_sd": config.outcome_noise_sd,
                "hidden_dim": config.hidden_dim,
                "epochs": config.epochs,
                "training_target_mean": training_result.baselines.training_mean,
                "inputs": "x_t|y_t_minus_1|previous_outcome_mask|episode_start",
                "objective": "deterministic_mse",
                "auxiliary_losses": False,
                "human_data_used": False,
            })
            network_rows.append({
                "condition": condition,
                "seed": int(seed),
                "architecture": "SRN",
                "hidden_dim": config.hidden_dim,
                "mse": metrics["model_mse"],
                "rmse": metrics["model_rmse"],
                "mae": metrics["model_mae"],
                **metrics,
            })
            for baseline, label in (
                ("global_mean", "B0_global_mean"),
                ("previous_outcome", "B1_previous_outcome"),
                ("context_only", "B2_context_only_regression"),
                ("oracle_bayes", "B3_oracle_bayesian"),
            ):
                baseline_rows.append({
                    "condition": condition,
                    "seed": int(seed),
                    "baseline": label,
                    "mse": metrics[f"{baseline}_mse"],
                    "rmse": metrics[f"{baseline}_rmse"],
                    "mae": metrics[f"{baseline}_mae"],
                    "fit_on_training_data": baseline != "oracle_bayes",
                    "oracle": baseline == "oracle_bayes",
                })
            ambiguity, alignment, ambiguity_hidden = evaluate_ambiguity(model, diagnostic)
            ambiguity = _attach(
                ambiguity, condition, seed, diagnostic_set_id=diagnostic_set_id
            )
            ambiguity_frames.append(ambiguity)
            responsibility_frames.append(ambiguity[[
                "condition", "seed", "diagnostic_set_id", "episode_id",
                "matched_state_history_id", "q_star",
                "q_implied_unclipped", "skill_a", "skill_b", "predicted_y", "posterior_mean",
            ]].copy())
            alignment_rows.append({
                "condition": condition,
                "seed": int(seed),
                "diagnostic_set_id": diagnostic_set_id,
                **alignment,
            })
            if condition_index == 0:
                ground_truth_frames.append(ambiguity[[
                    "seed", "diagnostic_set_id", "episode_id", "matched_state_history_id", "q_star",
                    "diagnostic_q_requested", "diagnostic_x", "skill_a", "skill_b",
                    "posterior_mean",
                ]].drop_duplicates())
            bundles.append({
                "condition": condition,
                "seed": int(seed),
                "model": model,
                "evaluated": evaluated,
                "hidden": hidden,
                "ambiguity": ambiguity,
                "ambiguity_hidden": ambiguity_hidden,
            })

    frames = {
        "training_conditions": pd.DataFrame(condition_rows),
        "network_performance": pd.DataFrame(network_rows),
        "baseline_performance": pd.DataFrame(baseline_rows),
        "ambiguity_behavior": pd.concat(ambiguity_frames, ignore_index=True),
        "bayesian_ground_truth": pd.concat(ground_truth_frames, ignore_index=True),
        "posterior_mean_alignment": pd.DataFrame(alignment_rows),
        "implied_responsibility": pd.concat(responsibility_frames, ignore_index=True),
        "continuous_vs_sampling": risk_table(config.ambiguity_q_levels),
    }
    frames["perfect_vs_uncertain"] = compare_conditions(frames["posterior_mean_alignment"])
    core_success, behavior_checks = _behavior_gate(
        frames["network_performance"], frames["posterior_mean_alignment"],
        frames["perfect_vs_uncertain"], config,
    )
    staged_allowed = core_success or config.force_staged_analyses
    report(f"Behavioral stage gate: {'passed' if core_success else 'not passed'}")

    if staged_allowed:
        report("Running episode-held-out hidden-state probes")
        probe_frames = []
        intervention_frames = []
        for bundle in bundles:
            evaluated = bundle["evaluated"]
            hidden = bundle["hidden"]
            probes = fit_hidden_probes(
                hidden.reshape(-1, config.hidden_dim),
                {
                    "S_A": evaluated.skill_a.to_numpy(float),
                    "S_B": evaluated.skill_b.to_numpy(float),
                    "q_star": evaluated.q_star.to_numpy(float),
                },
                evaluated.episode_id.to_numpy(),
                seed=bundle["seed"] + 73,
            )
            probes = _attach(probes, bundle["condition"], bundle["seed"])
            probe_frames.append(probes)
            q_r2 = float(probes.loc[probes.target.eq("q_star"), "heldout_r2"].iloc[0])
            ambiguity = bundle["ambiguity"]
            intervention = belief_intervention(
                bundle["model"],
                bundle["ambiguity_hidden"],
                ambiguity.q_star.to_numpy(float),
                ambiguity.skill_a.to_numpy(float),
                ambiguity.skill_b.to_numpy(float),
                heldout_q_r2=q_r2,
                minimum_decode_r2=config.intervention_decode_threshold,
            )
            intervention_frames.append(
                _attach(intervention, bundle["condition"], bundle["seed"])
            )
        frames["hidden_state_probes"] = pd.concat(probe_frames, ignore_index=True)
        frames["belief_interventions"] = pd.concat(intervention_frames, ignore_index=True)
    else:
        reason = "behavioral averaging gate was not met"
        frames["hidden_state_probes"] = _not_run("hidden_state_probes", reason)
        frames["belief_interventions"] = _not_run("belief_interventions", reason)

    dimension_requested = config.run_dimensionality
    if dimension_requested and staged_allowed:
        report("Running staged Elman-SRN hidden-dimension ladder")
        frames["dimensionality_comparison"] = run_architecture_grid(
            architectures=("srn",),
            hidden_dims=config.dimensionality_hidden_dims,
            seeds=config.dimensionality_seeds or config.seeds,
            conditions=conditions,
            train_episode_length=config.train_episode_length,
            context_persistence=config.context_persistence,
            outcome_noise_sd=config.outcome_noise_sd,
            epochs=config.epochs,
            batches_per_epoch=config.batches_per_epoch,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            eval_episodes=config.eval_episodes,
            ambiguity_q_levels=config.ambiguity_q_levels,
            ambiguity_episodes_per_q=config.ambiguity_episodes_per_q,
            ambiguity_history_length=config.ambiguity_history_length,
            diagnostic_noise_sd=config.uncertain_noise_sd,
            progress=report,
        )
    else:
        reason = (
            "disabled by configuration" if not dimension_requested
            else "behavioral averaging gate was not met"
        )
        frames["dimensionality_comparison"] = _not_run("dimensionality_comparison", reason)

    control_requested = config.run_gru_control
    if control_requested and staged_allowed:
        report("Running optional matched SRN-versus-GRU robustness control")
        frames["gru_robustness_control"] = run_architecture_grid(
            architectures=("srn", "gru"),
            hidden_dims=(config.hidden_dim,),
            seeds=config.control_seeds or config.seeds,
            conditions=conditions,
            train_episode_length=config.train_episode_length,
            context_persistence=config.context_persistence,
            outcome_noise_sd=config.outcome_noise_sd,
            epochs=config.epochs,
            batches_per_epoch=config.batches_per_epoch,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            eval_episodes=config.eval_episodes,
            ambiguity_q_levels=config.ambiguity_q_levels,
            ambiguity_episodes_per_q=config.ambiguity_episodes_per_q,
            ambiguity_history_length=config.ambiguity_history_length,
            diagnostic_noise_sd=config.uncertain_noise_sd,
            progress=report,
        )
    else:
        reason = (
            "optional control disabled by configuration" if not control_requested
            else "behavioral averaging gate was not met"
        )
        frames["gru_robustness_control"] = _not_run("gru_robustness_control", reason)

    completed_probes = frames["hidden_state_probes"]
    if "condition" in completed_probes and "target" in completed_probes:
        q_probe = (
            completed_probes[completed_probes.target.eq("q_star")]
            [["condition", "seed", "heldout_r2"]]
            .rename(columns={"heldout_r2": "q_heldout_r2"})
        )
    else:
        q_probe = pd.DataFrame(columns=["condition", "seed", "q_heldout_r2"])
    frames["seed_summary"] = (
        frames["network_performance"]
        .merge(frames["posterior_mean_alignment"], on=["condition", "seed"])
        .merge(q_probe, on=["condition", "seed"], how="left")
    )
    result_code, decision, decision_checks = _decision(
        core_success,
        behavior_checks,
        frames["hidden_state_probes"],
        frames["perfect_vs_uncertain"],
        frames["gru_robustness_control"],
        len(config.seeds),
    )

    export_order = {
        "training_conditions.csv": "training_conditions",
        "network_performance.csv": "network_performance",
        "baseline_performance.csv": "baseline_performance",
        "ambiguity_behavior.csv": "ambiguity_behavior",
        "bayesian_ground_truth.csv": "bayesian_ground_truth",
        "posterior_mean_alignment.csv": "posterior_mean_alignment",
        "implied_responsibility.csv": "implied_responsibility",
        "perfect_vs_uncertain.csv": "perfect_vs_uncertain",
        "continuous_vs_sampling.csv": "continuous_vs_sampling",
        "hidden_state_probes.csv": "hidden_state_probes",
        "dimensionality_comparison.csv": "dimensionality_comparison",
        "belief_interventions.csv": "belief_interventions",
        "gru_robustness_control.csv": "gru_robustness_control",
        "seed_summary.csv": "seed_summary",
    }
    report(f"Writing {len(export_order)} required CSV artifacts")
    for filename, key in export_order.items():
        _write_csv(frames[key], output / filename)

    run_record = {
        "analysis": "PRD v1 Elman SRN emergent averaging under partial observability",
        "version": __version__,
        "profile": config.profile,
        "config": asdict(config),
        "primary_architecture": "Elman SRN",
        "inputs": ["x_t", "y_t_minus_1", "previous_outcome_mask", "episode_start"],
        "forbidden_training_inputs": ["z_t", "S_A", "S_B", "q_star", "current_y_t"],
        "objective": "deterministic MSE only",
        "auxiliary_losses": False,
        "explicit_mixture_computation": False,
        "human_data_used": False,
        "same_ambiguity_sets_across_conditions": True,
        "behavioral_stage_gate_passed": core_success,
        "prespecified_result": result_code,
        "proof_of_concept_decision": decision,
        "decision_checks": decision_checks,
        "staged_status": {
            "hidden_state_probes": frames["hidden_state_probes"].status.iloc[0],
            "dimensionality_comparison": frames["dimensionality_comparison"].status.iloc[0],
            "belief_interventions": frames["belief_interventions"].status.iloc[0],
            "gru_robustness_control": frames["gru_robustness_control"].status.iloc[0],
        },
        "outputs": {filename: int(len(frames[key])) for filename, key in export_order.items()},
    }
    _write_text(json.dumps(run_record, indent=2), output / "run_config_v1.json")
    _write_report(output / "proof_of_concept_report.md", result_code, decision, decision_checks, frames)
    if config.make_plots:
        report("Rendering nine required figures")
        # Import plotting only at the plotting stage. This keeps CLI startup
        # immediate and ensures a first-time Matplotlib font-cache build cannot
        # make the training command appear silent.
        from .plotting import make_figures

        evaluations = pd.concat(evaluation_frames, ignore_index=True)
        example = evaluations[
            evaluations.condition.eq("Uncertain")
            & evaluations.seed.eq(config.seeds[0])
        ]
        make_figures(
            output,
            example_episode=example,
            ambiguity=frames["ambiguity_behavior"],
            comparison=frames["perfect_vs_uncertain"],
            risk=frames["continuous_vs_sampling"],
            probes=frames["hidden_state_probes"],
            dimensionality=frames["dimensionality_comparison"],
            gru_control=frames["gru_robustness_control"],
        )
    report(f"Analysis complete | Result {result_code}: {decision}")
    frames["status"] = pd.DataFrame([{
        "result": result_code, "decision": decision, "behavioral_stage_gate_passed": core_success,
    }])
    return frames
