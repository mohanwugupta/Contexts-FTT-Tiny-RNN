"""Sequential E0--E3 gates followed by selected-ecology human transfer."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

import pandas as pd
import torch

from emergent_retrieval_program.analysis import profile_fits
from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .analysis import (
    decomposition_summary, heldout_summary, human_gates, human_success,
    human_tables, stage_gate, training_summary,
)
from .config import PRETRAINING_CONDITIONS, STAGES, CurriculumConfig
from .human import ablation_states, train_human
from .modeling import train_curriculum_stage
from .plotting import human_figures, make_internal_figures


PRETRAINING_OUTPUTS = (
    "curriculum_pretraining_config.json", "ecology_stage_registry.json",
    "curriculum_world_manifest.csv", "curriculum_training_summary.csv",
    "curriculum_heldout_generalization.csv",
    "curriculum_modified_position_accuracy.csv", "curriculum_stage_gates.csv",
    "STRUCTURAL_PRIOR_CURRICULUM_REPORT.md",
)
HUMAN_OUTPUTS = (
    "curriculum_human_training.csv", "curriculum_human_state_space.csv",
    "curriculum_human_probe_metrics.csv", "curriculum_human_first_probe.csv",
    "curriculum_human_interpolation.csv", "curriculum_human_hidden_corridor.csv",
)


def _progress(message):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] PRIOR-CURRICULUM | {message}", flush=True)


def _phase_frame(frame, phase):
    value = frame.copy(); value.insert(1, "phase", phase); return value


def _checkpoint(output: Path, run):
    target = output / "curriculum_checkpoints"; target.mkdir(parents=True, exist_ok=True)
    torch.save({
        "stage": run.stage, "phase": run.phase,
        "pretraining_condition": run.condition, "seed": run.seed,
        "model_state_dict": run.engine.model.state_dict(),
        "policy_state_dict": run.engine.policy.state_dict(), "audit": run.audit,
    }, target / f"{run.stage}_{run.phase}_{run.condition}_seed_{run.seed}.pt")


def _human_profile(probes):
    parts = []
    for epoch, one in probes.groupby("probe_epoch"):
        source = one.copy(); source["model_variant"] = source.pretraining_condition
        frame = profile_fits(source).rename(columns={"model_variant": "pretraining_condition"})
        frame["probe_epoch"] = epoch; parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _report(output, config, gates, selected_stage, human_gate, success):
    attempted = ", ".join(gates.stage.drop_duplicates()) if len(gates) else "none"
    passing = gates[(gates.phase.eq("confirmatory")) & gates.gate_passed].stage.tolist() if len(gates) else []
    human_result = bool(len(success) and success.human_transfer_success.iloc[0])
    if selected_stage is None:
        interpretation = "No ecology passed the full pilot plus confirmatory gate; the human task was not run."
    else:
        interpretation = f"{selected_stage} was the most abstract passing ecology and was selected before human transfer."
    lines = [
        "# Structural Prior Curriculum Report", "", "## Outcome", "",
        f"- Attempted ecology stages: {attempted}",
        f"- Confirmatory-passing stages: {', '.join(passing) or 'none'}",
        f"- Selected transfer ecology: **{selected_stage or 'none'}**",
        f"- Human-transfer success criterion: **{human_result}**",
        f"- Interpretation: {interpretation}", "", "## Sequential stage gates", "",
        gates.to_markdown(index=False) if len(gates) else "No stages attempted.",
        "", "## Human familiar gates", "",
        human_gate.to_markdown(index=False) if len(human_gate) else "Not reached; no human outputs were created.",
        "", "## Firewalls", "",
        "The [1,1] combination received zero optimizer updates in every world. No human A/B, Mixed, Novel, corridor, or interpolation calculation occurred before an ecology was selected. Ecology progression and checkpoint selection used internal held-out outcomes only.",
        "", "## Success-contingent ablations", "",
        "A1--A4 run only when H2 passes the registered first-probe human-transfer criterion.",
        "", "## Reproducibility", "",
        f"- Analysis version: `{config.payload['analysis_version']}`",
        f"- Config SHA-256: `{config.config_sha256}`",
        f"- Stage registry SHA-256: `{config.registry_sha256}`",
        f"- Smoke mode: `{config.smoke}`",
    ]
    (output / "STRUCTURAL_PRIOR_CURRICULUM_REPORT.md").write_text("\n".join(lines) + "\n")


def run_structural_prior_curriculum(config: CurriculumConfig):
    output = config.output_dir; output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.path, output / "curriculum_pretraining_config.json")
    shutil.copyfile(config.registry_path, output / "ecology_stage_registry.json")
    registry = load_coordinate_registry(
        config.coordinate_registry,
        {str(k): int(v) for k, v in config.payload["human_transfer"]["coordinate_array_assignment"].items()},
    )
    training_parts, heldout_parts, decomposition_parts = [], [], []
    world_parts, audits, gate_rows = [], [], []
    selected_stage = None; selected_runs = {}
    _progress(f"start; smoke={config.smoke}; output={output}")
    for stage in STAGES:
        _progress(f"{stage} pilot begins")
        for seed in config.pilot_seeds:
            for condition in PRETRAINING_CONDITIONS:
                run = train_curriculum_stage(
                    config, stage, "pilot", condition, seed, config.pilot_worlds,
                    progress=_progress,
                )
                training_parts.append(run.training); heldout_parts.append(run.heldout)
                decomposition_parts.append(run.decomposition); audits.append(run.audit)
                if condition == "Structured": world_parts.append(_phase_frame(run.worlds, "pilot"))
                _checkpoint(output, run)
        current_heldout = heldout_summary(pd.concat(heldout_parts, ignore_index=True))
        pilot = stage_gate(current_heldout, stage, "pilot", config.gate_fraction)
        gate_rows.append(pilot); _progress(f"{stage} pilot gate={pilot['gate_passed']}")
        if not pilot["gate_passed"]:
            break
        _progress(f"{stage} confirmatory begins")
        stage_confirm = {}
        for seed in config.confirmatory_seeds:
            for condition in PRETRAINING_CONDITIONS:
                run = train_curriculum_stage(
                    config, stage, "confirmatory", condition, seed,
                    config.confirmatory_worlds, progress=_progress,
                )
                stage_confirm[(condition, seed)] = run
                training_parts.append(run.training); heldout_parts.append(run.heldout)
                decomposition_parts.append(run.decomposition); audits.append(run.audit)
                if condition == "Structured": world_parts.append(_phase_frame(run.worlds, "confirmatory"))
                _checkpoint(output, run)
        current_heldout = heldout_summary(pd.concat(heldout_parts, ignore_index=True))
        confirm = stage_gate(current_heldout, stage, "confirmatory", config.gate_fraction)
        gate_rows.append(confirm); _progress(f"{stage} confirmatory gate={confirm['gate_passed']}")
        if not confirm["gate_passed"]:
            break
        selected_stage = stage; selected_runs = stage_confirm

    raw_training = pd.concat(training_parts, ignore_index=True) if training_parts else pd.DataFrame()
    raw_heldout = pd.concat(heldout_parts, ignore_index=True) if heldout_parts else pd.DataFrame()
    raw_decomposition = pd.concat(decomposition_parts, ignore_index=True) if decomposition_parts else pd.DataFrame()
    training = training_summary(raw_training); heldout = heldout_summary(raw_heldout)
    decomposition = decomposition_summary(raw_decomposition); gates = pd.DataFrame(gate_rows)
    pretraining_tables = {
        "curriculum_world_manifest.csv": pd.concat(world_parts, ignore_index=True) if world_parts else pd.DataFrame(),
        "curriculum_training_summary.csv": training,
        "curriculum_heldout_generalization.csv": heldout,
        "curriculum_modified_position_accuracy.csv": decomposition,
        "curriculum_stage_gates.csv": gates,
        "curriculum_training_audit.csv": pd.DataFrame(audits),
    }
    for name, frame in pretraining_tables.items(): frame.to_csv(output / name, index=False)
    make_internal_figures(output, heldout, gates, decomposition)

    human_tables_out = {}; human_gate = pd.DataFrame(); success = pd.DataFrame()
    if selected_stage is not None:
        _progress(f"human transfer begins from selected stage {selected_stage}")
        human_runs = []
        snapshots = {}
        for seed in config.confirmatory_seeds:
            for condition, group in (("Shuffled", "H1_shuffled"), ("Structured", "H2_structured")):
                source = selected_runs[(condition, seed)]
                snapshots[(group, seed)] = (
                    deepcopy(source.engine.model.state_dict()),
                    deepcopy(source.engine.policy.state_dict()),
                )
            human_runs.append(train_human(config, registry, "H0_random", seed, selected_stage))
            for group in ("H1_shuffled", "H2_structured"):
                model, policy = snapshots[(group, seed)]
                human_runs.append(train_human(
                    config, registry, group, seed, selected_stage,
                    model_state=model, policy_state=policy,
                ))
                _progress(f"human {group} seed {seed}")
        human_training = pd.concat([run.training for run in human_runs], ignore_index=True)
        human_probes = pd.concat([run.probes for run in human_runs], ignore_index=True)
        human_audits = pd.DataFrame([run.audit for run in human_runs])
        if human_audits.human_schedule_sha256.nunique() != 1 or not human_audits.optimizer_empty_after_reset.all():
            raise AssertionError("Human optimizer reset or paired schedule failed.")
        (summary, fits, classes, predictions, hidden, metrics, interpolation,
         corridor, first, frontier) = human_tables(human_training, human_probes)
        human_gate = human_gates(summary, hidden, classes, human_audits, config.gate_fraction)
        success = human_success(human_gate, first, metrics, hidden)
        profile = _human_profile(human_probes)
        human_tables_out = {
            "curriculum_human_training.csv": human_training,
            "curriculum_human_training_summary.csv": summary,
            "curriculum_human_training_audit.csv": human_audits,
            "curriculum_human_state_space.csv": fits,
            "curriculum_human_state_space_classification.csv": classes,
            "curriculum_human_state_space_predictions.csv": predictions,
            "curriculum_human_hidden_differentiation.csv": hidden,
            "curriculum_human_probe_transition_level.csv": human_probes,
            "curriculum_human_probe_metrics.csv": metrics,
            "curriculum_human_first_probe.csv": first,
            "curriculum_human_interpolation.csv": interpolation,
            "curriculum_human_hidden_corridor.csv": corridor,
            "curriculum_human_profile_fit.csv": profile,
            "curriculum_human_differentiation_transfer.csv": frontier,
            "curriculum_human_gate_summary.csv": human_gate,
            "curriculum_human_success_audit.csv": success,
        }
        human_figures(output, summary, classes, hidden, human_probes, metrics,
                      interpolation, corridor, frontier)
        if bool(success.human_transfer_success.iloc[0]):
            _progress("human success gate passed; running A1--A4")
            ablation_runs = []
            for seed in config.confirmatory_seeds:
                model, policy = snapshots[("H2_structured", seed)]
                for label, (branch_model, branch_policy) in ablation_states(
                    config, seed, model, policy
                ).items():
                    ablation_runs.append(train_human(
                        config, registry, f"H2_{label}", seed, selected_stage,
                        model_state=branch_model, policy_state=branch_policy,
                        ablation=label,
                    ))
            ablation_probes = pd.concat([run.probes for run in ablation_runs], ignore_index=True)
            _, _, _, _, _, ablation_metrics, ablation_interpolation, ablation_corridor, ablation_first, _ = human_tables(
                pd.concat([run.training for run in ablation_runs], ignore_index=True), ablation_probes
            )
            human_tables_out["curriculum_human_ablation_results.csv"] = pd.concat([
                ablation_metrics.assign(result_family="probe_metrics"),
                ablation_interpolation.assign(result_family="interpolation"),
                ablation_corridor.assign(result_family="hidden_corridor"),
                ablation_first.assign(result_family="first_probe"),
            ], ignore_index=True, sort=False)
        for name, frame in human_tables_out.items(): frame.to_csv(output / name, index=False)

    _report(output, config, gates, selected_stage, human_gate, success)
    missing = [name for name in PRETRAINING_OUTPUTS if not (output / name).exists()]
    if selected_stage is not None:
        missing += [name for name in HUMAN_OUTPUTS if not (output / name).exists()]
    if missing: raise AssertionError(f"Required outputs missing: {missing}")
    manifest = {
        "analysis_version": config.payload["analysis_version"], "smoke": config.smoke,
        "attempted_stages": gates.stage.drop_duplicates().tolist(),
        "confirmatory_passing_stages": gates[(gates.phase.eq("confirmatory")) & gates.gate_passed].stage.tolist(),
        "selected_stage": selected_stage, "human_transfer_reached": selected_stage is not None,
        "human_ablations_run": "curriculum_human_ablation_results.csv" in human_tables_out,
        "config_sha256": config.config_sha256, "stage_registry_sha256": config.registry_sha256,
        "coordinate_registry_sha256": registry.source_sha256,
    }
    (output / "structural_prior_curriculum_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _progress(f"finished; selected_stage={selected_stage}")
    return {**pretraining_tables, **human_tables_out,
            "curriculum_stage_gates.csv": gates}
