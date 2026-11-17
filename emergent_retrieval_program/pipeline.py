"""Hard-gated, A/B-only execution of the retrieval/context-inference program."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .analysis import (
    bootstrap_statistics, evaluate_gates, hidden_corridor,
    hidden_differentiation, interpolation_metrics, latent_context,
    probe_metrics, profile_fits, state_space_tables, success_audit,
    sufficiency_table, training_summary,
)
from .config import EMERGENT_VARIANTS, MODEL_CONDITIONS, RetrievalProgramConfig
from .controls import build_control, train_baseline
from .diagnostics import ablation_results, history_effects
from .modeling import train_emergent_model
from .plotting import make_all_figures


REQUIRED_OUTPUTS = (
    "retrieval_program_config.json", "model_variant_registry.json",
    "retrieval_training_summary.csv", "retrieval_state_space.csv",
    "retrieval_hidden_differentiation.csv", "retrieval_rt_grounding.csv",
    "retrieval_probe_metrics.csv", "retrieval_interpolation_metrics.csv",
    "retrieval_profile_fit.csv", "retrieval_history_effects.csv",
    "retrieval_hidden_corridor.csv", "retrieval_latent_context.csv",
    "retrieval_latent_dynamics.csv", "retrieval_ablation_results.csv",
    "EMERGENT_RETRIEVAL_PROGRAM_REPORT.md",
)


def _progress(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] RETRIEVAL-INFERENCE | {message}", flush=True)


def _frames(runs: list):
    training = pd.concat([run.training for run in runs], ignore_index=True, sort=False)
    probes = pd.concat([run.probes for run in runs], ignore_index=True, sort=False)
    audits = pd.DataFrame([run.audit for run in runs])
    dynamics = pd.concat([run.latent_dynamics for run in runs if len(run.latent_dynamics)],
                         ignore_index=True, sort=False) if any(len(run.latent_dynamics) for run in runs) else pd.DataFrame()
    return training, probes, audits, dynamics


def _analyze_gates(runs: list, config: RetrievalProgramConfig):
    training, probes, audits, _ = _frames(runs)
    summary = training_summary(training)
    hidden = hidden_differentiation(probes)
    fits, classes, predictions = state_space_tables(training)
    gates = evaluate_gates(summary, hidden, classes, audits, config.gate_fraction)
    return summary, hidden, fits, classes, predictions, gates


def _control_succeeds(runs: list, config: RetrievalProgramConfig) -> tuple[bool, str]:
    _, _, _, _, _, gates = _analyze_gates(runs, config)
    probes = pd.concat([run.probes for run in runs], ignore_index=True, sort=False)
    metrics = probe_metrics(probes)
    indices = interpolation_metrics(probes)
    means = metrics[(metrics.row_type.eq("seed_mean")) &
                    metrics.condition.eq("High_monotonic") &
                    metrics.probe_context.eq("Mixed")]
    nll = means.groupby("model_variant").NLL_deficit_over_worse_endpoint.mean()
    fractions = indices[(indices.row_type.eq("seed_mean")) &
                        indices.condition.eq("High_monotonic") &
                        indices.probe_context.eq("Mixed")].groupby("model_variant").between_endpoint_fraction.mean()
    passed_gate = bool(gates[gates.model_variant.eq("C1")].gate_passed.iloc[0])
    effective = bool(nll.get("C1", np.inf) < nll.get("V0", np.inf) and
                     fractions.get("C1", -np.inf) > fractions.get("V0", -np.inf))
    reason = ("C1 passed the frozen familiar gates and improved both prespecified Mixed "
              "benchmarks." if passed_gate and effective else
              "C1 did not jointly pass familiar gates and improve Mixed NLL deficit plus between-endpoint fraction.")
    return passed_gate and effective, reason


def _reproduce_v0(runs: list, config: RetrievalProgramConfig) -> pd.DataFrame:
    columns = ["condition", "seed", "context", "transition_position",
               "correct_token_nll", "correct_token_probability", "normalized_RT"]
    if config.smoke:
        return pd.DataFrame([{"status": "skipped_in_smoke", "within_tolerance": True}])
    source = config.repo_root / config.payload["baseline_reference_dir"] / "geometry_probe_transition_level.csv"
    if not source.exists():
        raise FileNotFoundError(f"Frozen V0 reference missing: {source}")
    expected = pd.read_csv(source)
    expected = expected[(expected.geometry_condition.eq("G2_shared_x")) &
                        expected.seed.isin(config.screen_seeds) &
                        expected.condition.isin(MODEL_CONDITIONS)]
    observed = pd.concat([run.probes for run in runs], ignore_index=True)
    observed = observed[observed.context.ne("Neutral")]
    joined = observed.merge(expected[columns], on=columns[:4], suffixes=("_observed", "_reference"), validate="one_to_one")
    error_columns = []
    for metric in columns[4:]:
        name = f"{metric}_absolute_error"; error_columns.append(name)
        joined[name] = (joined[f"{metric}_observed"] - joined[f"{metric}_reference"]).abs()
    tolerance = float(config.payload["baseline_reproduction_tolerance"])
    joined["within_tolerance"] = joined[error_columns].max(axis=1).le(tolerance)
    if len(joined) != len(observed) or not joined.within_tolerance.all():
        raise AssertionError("V0 does not reproduce the saved G2 baseline.")
    return joined[columns[:4] + error_columns + ["within_tolerance"]]


def _write_report(output: Path, config: RetrievalProgramConfig, stage_status: str,
                  c1_reason: str, gates: pd.DataFrame, success: pd.DataFrame,
                  sufficiency: pd.DataFrame, audits: pd.DataFrame) -> None:
    passing = gates[gates.gate_passed].model_variant.tolist() if len(gates) else []
    candidates = success[success.strong_candidate].model_variant.tolist() if len(success) else []
    firewall = bool(len(audits) and audits.num_train_mixed.eq(0).all() and audits.num_train_novel.eq(0).all())
    lines = [
        "# Emergent Retrieval and Context-Inference Program",
        "", "## Outcome", "",
        f"- Pipeline status: **{stage_status}**",
        f"- C1 checkpoint: {c1_reason}",
        f"- Familiar-gate passing variants: {', '.join(passing) or 'none'}",
        f"- Strong emergent candidates under S1–S5: {', '.join(candidates) or 'none'}",
        f"- Probe firewall (zero Mixed/Novel training batches): **{firewall}**",
        "", "## Interpretation guardrails", "",
        "Mixed, Novel, interpolation, corridor, and human outcomes were never used for training, hyperparameter selection, or confirmatory seed extension. C1/C2 impose output averaging; E1–E4 do not. All primary emergent models use a two-dimensional context state and the primary probe policy resets that state.",
        "", "## Gate audit", "", gates.to_markdown(index=False) if len(gates) else "No gate rows.",
        "", "## Sufficiency versus emergence", "", sufficiency.to_markdown(index=False) if len(sufficiency) else "No probe comparison rows.",
        "", "## Success audit", "", success.to_markdown(index=False) if len(success) else "No emergent models reached confirmatory evaluation.",
        "", "## Reproducibility", "",
        f"- Analysis version: `{config.payload['analysis_version']}`",
        f"- Config SHA-256: `{config.config_sha256}`",
        f"- Variant registry SHA-256: `{config.registry_sha256}`",
        f"- Coordinate registry SHA-256: `{audits.coordinate_registry_sha256.iloc[0] if 'coordinate_registry_sha256' in audits and len(audits) else 'recorded in config'}`",
        "- RT is the frozen monotonic execution mapping `1 - speed`.",
        "- Statistical contrasts are paired-seed bootstrap summaries (the paired design absorbs the seed random intercept).",
    ]
    (output / "EMERGENT_RETRIEVAL_PROGRAM_REPORT.md").write_text("\n".join(lines) + "\n")


def run_retrieval_program(config: RetrievalProgramConfig) -> dict[str, pd.DataFrame]:
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, output / "retrieval_program_config.json")
    shutil.copyfile(config.registry_path, output / "model_variant_registry.json")
    registry = load_coordinate_registry(config.coordinate_registry, config.layout_assignment)
    _progress(f"start; output={output}; smoke={config.smoke}")
    lookup = {}; screen_runs = []

    # Stage 0: exact G2 baseline.
    for condition in MODEL_CONDITIONS:
        for seed in config.screen_seeds:
            run = train_baseline(config, registry, condition, seed)
            run.audit["coordinate_registry_sha256"] = registry.source_sha256
            lookup[("V0", condition, seed)] = run; screen_runs.append(run)
            _progress(f"Stage 0 V0 {condition} seed {seed}")
    reproduction = _reproduce_v0(screen_runs, config)

    # Stage 1: explicit sufficiency control, derived post hoc from identical V0 weights.
    for condition in MODEL_CONDITIONS:
        for seed in config.screen_seeds:
            run = build_control(config, registry, lookup[("V0", condition, seed)], "C1")
            run.audit["coordinate_registry_sha256"] = registry.source_sha256
            lookup[("C1", condition, seed)] = run; screen_runs.append(run)
            _progress(f"Stage 1 C1 {condition} seed {seed}")
    c1_passed, c1_reason = _control_succeeds(screen_runs, config)
    _progress(f"Stage 1 checkpoint: {c1_passed}; {c1_reason}")

    screen_gate_rows = []
    if c1_passed:
        # Stages 2--5: primary emergent models, never selected on probe outcomes.
        for stage, variant in enumerate(EMERGENT_VARIANTS, 2):
            for condition in MODEL_CONDITIONS:
                for seed in config.screen_seeds:
                    run = train_emergent_model(config, registry, variant, condition, seed)
                    run.audit["coordinate_registry_sha256"] = registry.source_sha256
                    lookup[(variant, condition, seed)] = run; screen_runs.append(run)
                    _progress(f"Stage {stage} {variant} {condition} seed {seed}")
        # Stage 6: temporal explicit control.
        for condition in MODEL_CONDITIONS:
            for seed in config.screen_seeds:
                run = build_control(config, registry, lookup[("V0", condition, seed)], "C2")
                run.audit["coordinate_registry_sha256"] = registry.source_sha256
                lookup[("C2", condition, seed)] = run; screen_runs.append(run)
                _progress(f"Stage 6 C2 {condition} seed {seed}")

    _, _, _, _, _, screen_gates = _analyze_gates(screen_runs, config)
    screen_gates.insert(1, "cohort", "screen")
    gate_pass = set(screen_gates[screen_gates.gate_passed].model_variant)

    # Stage 7: controls always extend after a successful C1 checkpoint; emergent
    # variants extend solely on their familiar A/B screen gates.
    if c1_passed:
        extend = {"V0", "C1", "C2"} | (gate_pass & set(EMERGENT_VARIANTS))
        extra_seeds = [seed for seed in config.confirmatory_seeds if seed not in config.screen_seeds]
        for condition in MODEL_CONDITIONS:
            for seed in extra_seeds:
                if "V0" in extend:
                    base = train_baseline(config, registry, condition, seed)
                    base.audit["coordinate_registry_sha256"] = registry.source_sha256
                    lookup[("V0", condition, seed)] = base
                for variant in ("C1", "C2"):
                    if variant in extend:
                        run = build_control(config, registry, lookup[("V0", condition, seed)], variant)
                        run.audit["coordinate_registry_sha256"] = registry.source_sha256
                        lookup[(variant, condition, seed)] = run
                for variant in EMERGENT_VARIANTS:
                    if variant in extend:
                        run = train_emergent_model(config, registry, variant, condition, seed)
                        run.audit["coordinate_registry_sha256"] = registry.source_sha256
                        lookup[(variant, condition, seed)] = run
                _progress(f"Stage 7 confirmatory {condition} seed {seed}; variants={sorted(extend)}")

    final_runs = list(lookup.values())
    training, probes, audits, dynamics = _frames(final_runs)
    summary = training_summary(training)
    hidden = hidden_differentiation(probes)
    fits, classes, predictions = state_space_tables(training)
    confirm_gates = evaluate_gates(summary, hidden, classes, audits, config.gate_fraction)
    confirm_gates.insert(1, "cohort", "available_final")
    gates = pd.concat([screen_gates, confirm_gates], ignore_index=True, sort=False)
    metrics = probe_metrics(probes); interpolation = interpolation_metrics(probes)
    profiles = profile_fits(probes); corridor = hidden_corridor(probes)
    latent = latent_context(probes)
    history, history_dynamics = history_effects(final_runs, registry, int(config.payload["history_sequences"]))
    if len(history_dynamics):
        dynamics = pd.concat([dynamics, history_dynamics], ignore_index=True, sort=False)
    current_gates = confirm_gates
    sufficiency = sufficiency_table(metrics, interpolation, current_gates, config.variant_payload)
    success = success_audit(
        metrics, interpolation, corridor, current_gates,
        float(config.payload["success"]["novel_maximum_NLL_worsening"]),
    )
    successful = success[success.strong_candidate].model_variant.tolist() if len(success) else []
    ablations = ablation_results(successful, lookup, config, registry)
    statistics = bootstrap_statistics(metrics, interpolation, corridor, config.bootstrap_draws,
                                      int(config.payload["bootstrap"]["seed"]))

    tables = {
        "retrieval_training_sequence_level.csv": training,
        "retrieval_training_audit.csv": audits,
        "retrieval_training_summary.csv": summary,
        "retrieval_state_space.csv": fits,
        "retrieval_state_space_classification.csv": classes,
        "retrieval_state_space_predictions.csv": predictions,
        "retrieval_hidden_differentiation.csv": hidden,
        "retrieval_rt_grounding.csv": summary[["model_variant", "condition", "seed", "confidence_to_RT_slope", "confidence_to_RT_slope_negative"]],
        "retrieval_gate_summary.csv": gates,
        "retrieval_probe_transition_level.csv": probes,
        "retrieval_probe_metrics.csv": metrics,
        "retrieval_interpolation_metrics.csv": interpolation,
        "retrieval_profile_fit.csv": profiles,
        "retrieval_history_effects.csv": history,
        "retrieval_hidden_corridor.csv": corridor,
        "retrieval_latent_context.csv": latent,
        "retrieval_latent_dynamics.csv": dynamics,
        "retrieval_ablation_results.csv": ablations,
        "retrieval_sufficiency_emergence.csv": sufficiency,
        "retrieval_success_audit.csv": success,
        "retrieval_confirmatory_statistics.csv": statistics,
        "retrieval_v0_reproduction.csv": reproduction,
    }
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    make_all_figures(output, config.variant_payload, summary, hidden, confirm_gates,
                     metrics, interpolation, corridor, latent, history, sufficiency)
    stage_status = "completed" if c1_passed else "stopped_after_C1_prespecified_checkpoint"
    _write_report(output, config, stage_status, c1_reason, confirm_gates, success, sufficiency, audits)
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Required outputs missing: {missing}")
    manifest = {
        "analysis_version": config.payload["analysis_version"],
        "config_sha256": config.config_sha256, "variant_registry_sha256": config.registry_sha256,
        "coordinate_registry_sha256": registry.source_sha256,
        "smoke": config.smoke, "stage_status": stage_status,
        "c1_checkpoint_passed": c1_passed,
        "required_outputs": list(REQUIRED_OUTPUTS),
    }
    (output / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _progress(f"finished; status={stage_status}")
    return tables
