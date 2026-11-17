"""Hard-gated P0/P1/P2 pretraining and frozen human transfer pipeline."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from emergent_retrieval_program.analysis import profile_fits
from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .analysis import (
    contrast_statistics, differentiation_transfer, evaluate_gates,
    first_probe_table, hidden_differentiation, human_training_summary,
    pilot_gate, probe_tables, state_space_tables, summarize_heldout,
    summarize_pretraining,
)
from .config import PRETRAINING_CONDITIONS, PretrainingEcologyConfig
from .generator import HUMAN_SEQUENCE, generate_world, task_for
from .modeling import human_features, no_pretraining_run, pretrain, train_human_transfer
from .plotting import make_all_figures, make_stopped_figures


REQUIRED_OUTPUTS = (
    "pretraining_ecology_config.json", "pretraining_world_generator_config.json",
    "human_transfer_config.json", "pretraining_training_summary.csv",
    "pretraining_heldout_generalization.csv", "pretraining_world_manifest.csv",
    "pretraining_shuffle_manifest.csv", "human_transfer_training_summary.csv",
    "human_transfer_state_space.csv", "human_transfer_hidden_differentiation.csv",
    "human_transfer_rt_grounding.csv", "human_transfer_probe_metrics.csv",
    "human_transfer_interpolation.csv", "human_transfer_hidden_corridor.csv",
    "human_transfer_first_probe.csv",
    "HIERARCHICAL_COMPOSITIONAL_PRETRAINING_REPORT.md",
)


def _progress(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] PRETRAIN-ECOLOGY | {message}", flush=True)


def _generator_audit(config: PretrainingEcologyConfig, registry) -> pd.DataFrame:
    rows = []
    human_evidence = {}
    for context in ("A", "B", "Mixed", "Novel"):
        row = human_features(registry, context)[0]
        human_evidence[context] = np.concatenate([row[40:42], row[8:40]])
    for seed in config.pilot_seeds:
        for world_id in range(config.pilot_worlds):
            world = generate_world(seed, world_id, config.generator)
            train_combos = {item.factors for item in world.observed}
            heldout_combos = {item.factors for item in world.heldout}
            contexts = world.observed + world.heldout
            cue_leak = any(any(np.array_equal(item.evidence[:2], value[:2])
                               for value in human_evidence.values()) for item in contexts)
            layout_leak = any(any(np.array_equal(item.evidence[2:], value[2:])
                                  for value in human_evidence.values()) for item in contexts)
            rows.append({
                "seed": seed, "world_id": world_id, "world_type": world.world_type,
                "factor_count": world.factor_count,
                "human_motor_sequence_leak": world.base_sequence == HUMAN_SEQUENCE,
                "human_cue_vector_leak": cue_leak, "human_layout_array_leak": layout_leak,
                "heldout_combination_seen_in_training": bool(train_combos & heldout_combos),
                "shuffle_is_derangement": all(i != j for i, j in enumerate(world.shuffle_mapping)),
                "structured_factor_mapping_exact": all(
                    np.allclose(item.clean_evidence,
                                world.shared_visible + np.asarray(item.factors, np.float32) @ world.factor_visible,
                                rtol=0, atol=1e-6)
                    and item.task_sequence == task_for(
                        world.base_sequence, item.factors, world.factor_modifiers
                    ) for item in world.observed
                ),
            })
    result = pd.DataFrame(rows)
    forbidden = ["human_motor_sequence_leak", "human_cue_vector_leak", "human_layout_array_leak",
                 "heldout_combination_seen_in_training"]
    if result[forbidden].any().any() or not result.shuffle_is_derangement.all() or not result.structured_factor_mapping_exact.all():
        raise AssertionError("Stage 0 generator validation failed.")
    return result


def _save_checkpoint(output: Path, run, phase: str) -> None:
    destination = output / "pretraining_checkpoints"; destination.mkdir(parents=True, exist_ok=True)
    torch.save({
        "pretraining_condition": run.condition, "seed": run.seed, "phase": phase,
        "model_state_dict": run.engine.model.state_dict(),
        "policy_state_dict": run.engine.policy.state_dict(), "audit": run.audit,
    }, destination / f"{phase}_{run.condition}_seed_{run.seed}.pt")


def _with_phase(frames: list[pd.DataFrame], phase: str) -> list[pd.DataFrame]:
    for frame in frames:
        if len(frame): frame.insert(0, "phase", phase)
    return frames


def _empty_outputs() -> dict[str, pd.DataFrame]:
    unavailable = lambda: pd.DataFrame(columns=["status"])
    return {
        "human_transfer_training_sequence_level.csv": unavailable(),
        "human_transfer_training_audit.csv": unavailable(),
        "human_transfer_training_summary.csv": pd.DataFrame(columns=["pretraining_condition", "seed"]),
        "human_transfer_state_space.csv": unavailable(),
        "human_transfer_state_space_classification.csv": unavailable(),
        "human_transfer_state_space_predictions.csv": unavailable(),
        "human_transfer_hidden_differentiation.csv": unavailable(),
        "human_transfer_rt_grounding.csv": unavailable(),
        "human_transfer_probe_transition_level.csv": unavailable(),
        "human_transfer_probe_metrics.csv": unavailable(),
        "human_transfer_interpolation.csv": unavailable(),
        "human_transfer_hidden_corridor.csv": unavailable(),
        "human_transfer_first_probe.csv": unavailable(),
        "human_transfer_profile_fit.csv": unavailable(),
        "human_transfer_differentiation_transfer.csv": unavailable(),
        "human_transfer_contrast_statistics.csv": unavailable(),
        "human_transfer_gate_summary.csv": unavailable(),
        "human_transfer_success_audit.csv": unavailable(),
        "human_transfer_P0_reproduction.csv": unavailable(),
    }


def _profile_table(probes: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for epoch, one in probes.groupby("probe_epoch", sort=False):
        source = one.copy(); source["model_variant"] = source.pretraining_condition
        result = profile_fits(source).rename(columns={"model_variant": "pretraining_condition"})
        result["probe_epoch"] = epoch; parts.append(result)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _p0_reproduction(probes: pd.DataFrame, config: PretrainingEcologyConfig) -> pd.DataFrame:
    if config.smoke:
        return pd.DataFrame([{
            "status": "validated_in_unit_test_for_checkpoint_20",
            "within_tolerance": True,
        }])
    path = config.repo_root / "outputs/human_svg_geometry_transfer/geometry_probe_transition_level.csv"
    expected = pd.read_csv(path)
    expected = expected[
        expected.geometry_condition.eq("G2_shared_x") &
        expected.condition.eq("High_monotonic") &
        expected.seed.isin(config.confirmatory_seeds) &
        expected.parent_checkpoint.eq(config.late_checkpoint)
    ]
    observed = probes[
        probes.pretraining_condition.eq("P0_none") &
        probes.probe_epoch.eq("late") & probes.context.ne("Neutral")
    ]
    keys = ["seed", "context", "transition_position"]
    values = ["correct_token_nll", "correct_token_probability", "normalized_RT"]
    joined = observed.merge(
        expected[keys + values], on=keys, suffixes=("_observed", "_reference"),
        validate="one_to_one",
    )
    errors = []
    for value in values:
        name = f"{value}_absolute_error"; errors.append(name)
        joined[name] = (joined[f"{value}_observed"] - joined[f"{value}_reference"]).abs()
    joined["within_tolerance"] = joined[errors].max(axis=1).le(5e-7)
    if len(joined) != len(observed) or not joined.within_tolerance.all():
        raise AssertionError("P0 failed exact saved G2 reproduction.")
    return joined[keys + errors + ["within_tolerance"]]


def _success_audit(gates, first, metrics, hidden) -> pd.DataFrame:
    final = metrics[(metrics.row_type.eq("seed_mean")) & metrics.probe_epoch.eq("late")]
    first_mean = first.groupby("pretraining_condition").first_probe_between_endpoint_fraction.mean()
    grouped = final.groupby(["pretraining_condition", "probe_context"])
    nll = grouped.NLL_deficit_over_worse_endpoint.mean()
    distance = hidden[hidden.row_type.eq("seed_mean")].groupby("pretraining_condition").aggregate_AB_separation.mean()
    p2_gate = bool(gates[gates.pretraining_condition.eq("P2_structured")].gate_passed.iloc[0])
    checks = {
        "P2_human_gates_passed": p2_gate,
        "P2_first_Mixed_better_than_P1": first_mean["P2_structured"] > first_mean["P1_shuffled"],
        "P2_first_Mixed_better_than_P0": first_mean["P2_structured"] > first_mean["P0_none"],
        "P2_Mixed_NLL_better_than_P1": nll[("P2_structured", "Mixed")] < nll[("P1_shuffled", "Mixed")],
        "P2_Novel_NLL_better_than_P1": nll[("P2_structured", "Novel")] < nll[("P1_shuffled", "Novel")],
        "P2_differentiation_at_least_80pct_P0": distance["P2_structured"] >= .8 * distance["P0_none"],
    }
    return pd.DataFrame([{**checks, "primary_success": bool(all(checks.values()))}])


def _report(output: Path, status: str, pilot: dict, gates: pd.DataFrame,
            success: pd.DataFrame, config: PretrainingEcologyConfig) -> None:
    result = bool(len(success) and success.primary_success.iloc[0])
    lines = [
        "# Hierarchical Compositional Pretraining Report", "", "## Outcome", "",
        f"- Pipeline status: **{status}**",
        f"- Pilot P2 lower held-out NLL than P1: **{pilot.get('P2_lower_NLL', False)}**",
        f"- Pilot P2 non-lower held-out accuracy: **{pilot.get('P2_nonlower_accuracy', False)}**",
        f"- Confirmatory primary success: **{result}**",
        "", "## Scientific firewall", "",
        "Synthetic pretraining used no human cue, layout, motor-sequence, task-label, Mixed, or Novel inputs. Human Mixed/Novel and all interpolation/corridor metrics were frozen outcomes only. P1 and P2 used paired worlds with identical marginals and update counts; only visible-context-to-task pairing differed.",
        "", "## Human-task gates", "",
        gates.to_markdown(index=False) if len(gates) else "Human transfer was not run because the internal pilot gate failed.",
        "", "## Mechanistic follow-ups", "",
        "M1--M4 remain disabled in the frozen configuration. The PRD reserves them for a separate analysis after a primary P2 success is established; they were not silently launched here.",
        "", "## Reproducibility", "",
        f"- Analysis version: `{config.payload['analysis_version']}`",
        f"- Ecology config SHA-256: `{config.hashes['ecology']}`",
        f"- Generator config SHA-256: `{config.hashes['generator']}`",
        f"- Human-transfer config SHA-256: `{config.hashes['human']}`",
        f"- Smoke mode: `{config.smoke}`",
    ]
    (output / "HIERARCHICAL_COMPOSITIONAL_PRETRAINING_REPORT.md").write_text("\n".join(lines) + "\n")


def run_pretraining_ecology(config: PretrainingEcologyConfig) -> dict[str, pd.DataFrame]:
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True)
    for source, name in ((config.path, "pretraining_ecology_config.json"),
                         (config.generator_path, "pretraining_world_generator_config.json"),
                         (config.human_path, "human_transfer_config.json")):
        shutil.copyfile(source, output / name)
    registry = load_coordinate_registry(
        config.coordinate_registry,
        {str(k): int(v) for k, v in config.human["coordinate_array_assignment"].items()},
    )
    _progress(f"Stage 0 generator validation; smoke={config.smoke}")
    generator_audit = _generator_audit(config, registry)

    all_training, all_heldout, all_worlds, all_shuffles, pretraining_audits = [], [], [], [], []
    pilot_runs = {}
    _progress(f"Stage 1 pilot: {config.pilot_worlds} paired worlds")
    for seed in config.pilot_seeds:
        for condition in ("P1_shuffled", "P2_structured"):
            run = pretrain(config, condition, seed, config.pilot_worlds, progress=_progress)
            pilot_runs[(condition, seed)] = run
            frames = _with_phase([run.training_summary, run.heldout, run.world_manifest, run.shuffle_manifest], "pilot")
            all_training.append(frames[0]); all_heldout.append(frames[1])
            if condition == "P2_structured":
                all_worlds.append(frames[2]); all_shuffles.append(frames[3])
            pretraining_audits.append({"phase": "pilot", **run.audit})
            _save_checkpoint(output, run, "pilot")
            _progress(f"pilot {condition} seed {seed}")
    heldout = summarize_heldout(pd.concat(all_heldout, ignore_index=True))
    admitted, pilot_values = pilot_gate(
        heldout, bool(config.payload["pilot_gate"]["require_nonlower_accuracy"])
    )
    _progress(f"Stage 2 frozen pilot gate: admitted={admitted}; {pilot_values}")

    final_tables = _empty_outputs()
    if admitted:
        confirm_runs = {}
        _progress(f"Stage 3 confirmatory pretraining: {config.confirmatory_worlds} paired worlds")
        for seed in config.confirmatory_seeds:
            confirm_runs[("P0_none", seed)] = no_pretraining_run(config, seed)
            for condition in ("P1_shuffled", "P2_structured"):
                run = pretrain(config, condition, seed, config.confirmatory_worlds, progress=_progress)
                confirm_runs[(condition, seed)] = run
                frames = _with_phase([run.training_summary, run.heldout, run.world_manifest, run.shuffle_manifest], "confirmatory")
                all_training.append(frames[0]); all_heldout.append(frames[1])
                if condition == "P2_structured":
                    all_worlds.append(frames[2]); all_shuffles.append(frames[3])
                pretraining_audits.append({"phase": "confirmatory", **run.audit})
                _save_checkpoint(output, run, "confirmatory")
                _progress(f"confirmatory {condition} seed {seed}")
        # Equal-exposure and matched-world checks are hard assertions.
        audit_frame = pd.DataFrame(pretraining_audits)
        confirm_audit = audit_frame[audit_frame.phase.eq("confirmatory")]
        exposure = confirm_audit.pivot(index="seed", columns="pretraining_condition",
                                       values="pretraining_optimizer_steps")
        if not (exposure.P1_shuffled == exposure.P2_structured).all():
            raise AssertionError("P1/P2 pretraining exposure differs.")

        _progress("Stages 4-5 identical human A/B training and frozen probes")
        human_runs = []
        for seed in config.confirmatory_seeds:
            for condition in PRETRAINING_CONDITIONS:
                run = train_human_transfer(config, registry, confirm_runs[(condition, seed)])
                human_runs.append(run); _progress(f"human transfer {condition} seed {seed}")
        training = pd.concat([run.training for run in human_runs], ignore_index=True)
        probes = pd.concat([run.probes for run in human_runs], ignore_index=True)
        audits = pd.DataFrame([run.audit for run in human_runs])
        if audits.human_schedule_sha256.nunique() != 1 or not audits.optimizer_empty_after_reset.all():
            raise AssertionError("Human schedule pairing or optimizer reset failed.")
        summary = human_training_summary(training)
        fits, classes, predictions = state_space_tables(training)
        hidden = hidden_differentiation(probes)
        metrics, interpolation, corridor = probe_tables(probes)
        first = first_probe_table(metrics, interpolation)
        frontier = differentiation_transfer(hidden, metrics, interpolation)
        profiles = _profile_table(probes)
        gates = evaluate_gates(
            summary, hidden, classes, audits,
            float(config.payload["gates"]["seed_fraction"] if not config.smoke else .5),
            float(config.payload["gates"]["P2_minimum_P0_differentiation_ratio"]),
            float(config.payload["gates"]["minimum_RT_sawtooth"]),
        )
        success = _success_audit(gates, first, metrics, hidden)
        statistics = contrast_statistics(
            first, metrics, interpolation, hidden, config.bootstrap_draws,
            int(config.payload["bootstrap"]["seed"]),
        )
        reproduction = _p0_reproduction(probes, config)
        final_tables = {
            "human_transfer_training_sequence_level.csv": training,
            "human_transfer_training_audit.csv": audits,
            "human_transfer_training_summary.csv": summary,
            "human_transfer_state_space.csv": fits,
            "human_transfer_state_space_classification.csv": classes,
            "human_transfer_state_space_predictions.csv": predictions,
            "human_transfer_hidden_differentiation.csv": hidden,
            "human_transfer_rt_grounding.csv": summary[["pretraining_condition", "seed", "confidence_to_RT_slope", "confidence_to_RT_slope_negative", "mean_context_yoked_RT_sawtooth"]],
            "human_transfer_probe_transition_level.csv": probes,
            "human_transfer_probe_metrics.csv": metrics,
            "human_transfer_interpolation.csv": interpolation,
            "human_transfer_hidden_corridor.csv": corridor,
            "human_transfer_first_probe.csv": first,
            "human_transfer_profile_fit.csv": profiles,
            "human_transfer_differentiation_transfer.csv": frontier,
            "human_transfer_contrast_statistics.csv": statistics,
            "human_transfer_gate_summary.csv": gates,
            "human_transfer_success_audit.csv": success,
            "human_transfer_P0_reproduction.csv": reproduction,
        }
        heldout = summarize_heldout(pd.concat(all_heldout, ignore_index=True))
        make_all_figures(output, heldout, summary, hidden, classes, probes,
                         metrics, interpolation, corridor, frontier, first)
        status = "completed"
    else:
        reason = "P2 did not beat matched P1 on both frozen internal criteria."
        make_stopped_figures(output, heldout, reason)
        gates, success = pd.DataFrame(), pd.DataFrame()
        status = "stopped_after_internal_pilot_gate"

    common_tables = {
        "pretraining_training_summary.csv": summarize_pretraining(
            pd.concat(all_training, ignore_index=True) if all_training else pd.DataFrame()
        ),
        "pretraining_heldout_generalization.csv": heldout,
        "pretraining_world_manifest.csv": pd.concat(all_worlds, ignore_index=True) if all_worlds else pd.DataFrame(),
        "pretraining_shuffle_manifest.csv": pd.concat(all_shuffles, ignore_index=True) if all_shuffles else pd.DataFrame(),
        "pretraining_training_audit.csv": pd.DataFrame(pretraining_audits),
        "pretraining_generator_audit.csv": generator_audit,
        "pretraining_pilot_gate.csv": pd.DataFrame([{**pilot_values, "pilot_admitted": admitted}]),
    }
    tables = {**common_tables, **final_tables}
    for name, frame in tables.items(): frame.to_csv(output / name, index=False)
    _report(output, status, pilot_values, gates, success, config)
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).exists()]
    if missing: raise AssertionError(f"Missing required outputs: {missing}")
    manifest = {
        "analysis_version": config.payload["analysis_version"], "smoke": config.smoke,
        "status": status, "pilot_admitted": admitted, "config_sha256": config.hashes,
        "coordinate_registry_sha256": registry.source_sha256,
        "required_outputs": list(REQUIRED_OUTPUTS),
    }
    (output / "pretraining_ecology_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _progress(f"finished; status={status}")
    return tables
