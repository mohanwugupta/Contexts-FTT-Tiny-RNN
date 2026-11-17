"""Staged, resumable orchestration for the human/SRN transfer program."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from .analysis import cluster_bootstrap
from .config import TransferProgramConfig, load_config
from .experiments import build_reference_cache, load_snapshot, run_e1, run_e2, run_e3, run_e5
from .human import bootstrap_first_later_human, build_human_event_audit, fit_local_familiar_references
from .human_task import analyze_human_task_events, assignment_cells, build_task_configs
from .modeling import SRNEngine, run_acquisition
from .plotting import (
    figure_differentiation, figure_e1, figure_e2, figure_e3, figure_e4, figure_e5,
    figure_h0, figure_synthesis,
)
from .power import estimate_human_nuisance, simulate_power_planning


VERSION = "transfer-experiment-program-1.0.0"
ORDER = ("audit", "human_audit", "e1", "e2", "e3", "e4", "e5", "report")


def progress(message: str):
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] TRANSFER | {message}", flush=True)


def _json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "git_metadata_unavailable"


def _config_for_run(path: str | Path, smoke: bool) -> TransferProgramConfig:
    config = load_config(path, smoke=smoke)
    if smoke:
        config = replace(config, output_dir=Path(str(config.output_dir) + "_smoke"))
    return config


def source_manifest(config: TransferProgramConfig) -> pd.DataFrame:
    rows = []
    for key in config.payload["sources"]:
        path = config.source(key)
        row = {"source_key": key, "resolved_path": str(path), "exists": path.exists(),
               "source_type": "directory" if path.is_dir() else "file"}
        if path.is_file():
            row.update({"bytes": path.stat().st_size, "sha256": _sha(path)})
        elif path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.is_file())
            row.update({"file_count": len(files),
                        "directory_listing_sha256": hashlib.sha256(
                            "\n".join(str(item.relative_to(path)) for item in files).encode()
                        ).hexdigest()})
        rows.append(row)
    return pd.DataFrame(rows)


def dry_run(config: TransferProgramConfig) -> dict:
    start = time.perf_counter()
    run_acquisition(
        config.conditions[0], config.additional_seeds[0], max_sequences=20,
        evaluation_times=(0, 20), cohort="dry_run", experiment="dry_run_timing",
    )
    seconds_per_20 = time.perf_counter() - start
    n_original, n_additional, n_condition = len(config.original_seeds), len(config.additional_seeds), len(config.conditions)
    p2 = config.payload["e2"]
    counts = {
        "audit_reference_training_updates": n_original * n_condition * 170,
        "e1_training_updates": n_additional * n_condition * 680,
        "e2_core_practice_sequences": n_additional * n_condition * len(p2["checkpoints"]) * len(p2["probes"]) * len(p2["primers"]) * 4 * 3 * int(p2["repetitions"]),
        "e3_factorial_branches": n_original * n_condition * len(config.payload["e3"]["checkpoints"]) * 2 * 2 * 4,
        "e5_pretraining_updates": n_additional * n_condition * 2 * 640,
        "e5_target_updates": n_additional * n_condition * 3 * 170,
        "power_simulation_draws_per_cell": config.simulation_draws,
    }
    total_updates = counts["audit_reference_training_updates"] + counts["e1_training_updates"] + counts["e5_pretraining_updates"] + counts["e5_target_updates"]
    result = {
        "version": VERSION, "smoke": config.smoke, "sources": source_manifest(config).to_dict("records"),
        "original_seeds": config.original_seeds, "additional_seeds": config.additional_seeds,
        "conditions": config.conditions, "reference_checkpoints": config.payload["reference"]["reference_checkpoints"],
        "training_caps": config.payload["reference"]["training_caps"], "counts": counts,
        "timing_sample_seconds_for_20_updates": seconds_per_20,
        "rough_training_only_minutes": total_updates / 20 * seconds_per_20 / 60,
        "note": "Estimate excludes CSV, browser, hashing, frozen branches, plotting, and human collection.",
    }
    print(json.dumps(result, indent=2, default=str), flush=True)
    return result


def _status_path(config: TransferProgramConfig) -> Path:
    return config.output_dir / "branch_status.csv"


def update_status(config: TransferProgramConfig, record: dict[str, Any]):
    path = _status_path(config)
    if path.exists():
        frame = pd.read_csv(path)
        frame = frame[~frame.work_package.eq(record["work_package"])]
    else:
        frame = pd.DataFrame()
    frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
    order = {name: index for index, name in enumerate(("Stage 0", "H0", "E1", "E2", "E3", "E4", "E5", "Synthesis"))}
    frame["_order"] = frame.work_package.map(order).fillna(999)
    frame.sort_values("_order").drop(columns="_order").to_csv(path, index=False)


def _mark(config: TransferProgramConfig, stage: str, start: float, status: str = "completed"):
    _json(config.output_dir / ".stage_status" / f"{stage}.json", {
        "stage": stage, "status": status, "version": VERSION, "config_hash": config.config_hash,
        "smoke": config.smoke, "completed_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
    })


def _already_done(config: TransferProgramConfig, stage: str) -> bool:
    path = config.output_dir / ".stage_status" / f"{stage}.json"
    if not path.exists(): return False
    payload = json.loads(path.read_text())
    return payload.get("status") == "completed" and payload.get("config_hash") == config.config_hash


def stage_audit(config: TransferProgramConfig):
    out = config.output_dir / "audit"; out.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest(config); manifest.to_csv(out / "source_manifest.csv", index=False)
    missing = manifest[~manifest.exists]
    if len(missing): raise FileNotFoundError(f"Required sources missing: {missing.source_key.tolist()}")
    _json(out / "resolved_config.json", {**config.payload, "resolved_repo_root": str(config.repo_root),
                                         "resolved_output_dir": str(config.output_dir),
                                         "config_hash": config.config_hash, "smoke": config.smoke})
    progress("Stage 0: replaying original frozen parents")
    original = build_reference_cache(config, original=True)
    progress("Stage 0: preparing independent additional-seed parents")
    build_reference_cache(config, original=False)
    isolation_rows = []
    cache = out / "checkpoint_cache"
    for condition in config.conditions:
        for seed in config.original_seeds:
            parent = load_snapshot(cache, "original_reference", condition, seed, 20)
            control = SRNEngine.from_snapshot(parent); diagnostic = SRNEngine.from_snapshot(parent)
            before = diagnostic.snapshot()
            diagnostic.score_sequence(context="Mixed", branch_id="stage0:isolation", update=False,
                                      metadata={"experiment": "stage0", "cohort": "original_reference"})
            after = diagnostic.snapshot()
            for engine in (control, diagnostic):
                engine.score_sequence(context="A", branch_id="stage0:next_training", update=True,
                                      metadata={"experiment": "stage0", "cohort": "original_reference"},
                                      record_transitions=False, audit_hashes=False)
            isolation_rows.append({"condition": condition, "seed": seed,
                                   "frozen_state_unchanged": before.state_sha256 == after.state_sha256,
                                   "frozen_optimizer_unchanged": before.optimizer_sha256 == after.optimizer_sha256,
                                   "frozen_rng_unchanged": before.rng_sha256 == after.rng_sha256,
                                   "subsequent_training_identical": control.snapshot().state_sha256 == diagnostic.snapshot().state_sha256})
    isolation = pd.DataFrame(isolation_rows); isolation.to_csv(out / "evaluation_isolation_audit.csv", index=False)
    if not isolation.select_dtypes(bool).all().all(): raise AssertionError("Frozen branch isolation failed.")
    pd.DataFrame([{"experiment": "stage0_reference", "cohort": "original_reference",
                   "seeds": ",".join(map(str, config.original_seeds)), "config_hash": config.config_hash,
                   "code_revision": _revision(config.repo_root), "initialization": "High=2.0;Low=.02",
                   "training_dose": 170, "input_layout_sequence_ids": "Stage2 A/B; fixed target sequence",
                   "parent_checkpoint": "20,100,170", "parent_hash": "see checkpoint cache"}]).to_csv(out / "run_manifest.csv", index=False)
    update_status(config, {"work_package": "Stage 0", "question": "Does the frozen baseline reproduce?",
                           "implementation_status": "validated", "evidence_status": "completed",
                           "prerequisites": "reference outputs and model source", "outputs": str(out),
                           "failed_checks": "", "scientific_interpretation_limit": "Reference reproduction only; not a mechanism test."})
    return original


def stage_human_audit(config: TransferProgramConfig):
    out = config.output_dir / "h0_human_audit"; out.mkdir(parents=True, exist_ok=True)
    event_frames, sequence_frames, audit_rows = [], [], []
    for dataset, source in (("Combined", config.source("human_mixed")), ("Novel", config.source("human_novel"))):
        progress(f"H0: reconstructing {dataset} chronology before exclusions")
        events, sequences, audit = build_human_event_audit(source, dataset)
        event_frames.append(events); sequence_frames.append(sequences); audit_rows.append(audit)
    events = pd.concat(event_frames, ignore_index=True); sequences = pd.concat(sequence_frames, ignore_index=True)
    progress("H0: selecting A/B-only regularization and validating pseudo-probe references")
    references, validation, selection, reference_audit = fit_local_familiar_references(sequences)
    bootstrap = bootstrap_first_later_human(references, draws=config.bootstrap_draws,
                                            seed=int(config.payload["bootstrap"]["seed"]))
    for name, frame in {"human_events.csv": events, "human_sequences.csv": sequences,
                        "human_local_references.csv": references, "pseudo_event_validation.csv": validation,
                        "regularization_selection.csv": selection, "first_later_bootstrap.csv": bootstrap,
                        "chronology_audit.csv": pd.DataFrame(audit_rows),
                        "local_reference_audit.csv": pd.DataFrame(reference_audit["datasets"])}.items():
        frame.to_csv(out / name, index=False)
    update_status(config, {"work_package": "H0", "question": "Are human/network probes comparable in time and exposure?",
                           "implementation_status": "validated", "evidence_status": "completed",
                           "prerequisites": "existing event-level Mixed/Novel files", "outputs": str(out),
                           "failed_checks": "elapsed/rest timestamps unavailable in legacy data",
                           "scientific_interpretation_limit": "Local A/B values are model-estimated counterfactuals, not same-snapshot observations."})
    return {"events": events, "sequences": sequences, "references": references,
            "validation": validation, "bootstrap": bootstrap}


def stage_e4(config: TransferProgramConfig, human_events: str | Path | None = None):
    out = config.output_dir / "e4_human_task"; out.mkdir(parents=True, exist_ok=True)
    configs = build_task_configs(out / "configs")
    info = pd.DataFrame([
        {"system": "Human full preview", "before_action_visible": "all 8 action labels, cue, layout", "recurrent_input": "not specified", "target_known": "before sequence", "execution_readout_input": "behavioral interval", "learning": "practice permitted"},
        {"system": "Human one target", "before_action_visible": "current action only, cue, layout", "recurrent_input": "not specified", "target_known": "at each action onset", "execution_readout_input": "behavioral interval", "learning": "practice permitted"},
        {"system": "Frozen Stage-2 SRN", "before_action_visible": "previous token, ordinal position, cue, full layout", "recurrent_input": "token + 42 context features", "target_known": "only for loss after prediction", "execution_readout_input": "correct-token confidence", "learning": "none during frozen probe"},
    ])
    info["tautological_target_feeding_implemented"] = False
    info.to_csv(out / "model_information_timing_audit.csv", index=False)
    schedules = []
    for experiment in ("e2", "e3", "e4"):
        payload = json.loads(configs[experiment].read_text())
        for cell_index, schedule in enumerate(payload["schedules"]):
            for row in schedule: schedules.append({"experiment": experiment, "cell_index": cell_index, **row})
    schedule_frame = pd.DataFrame(schedules); schedule_frame.to_csv(out / "generated_schedules.csv", index=False)
    balance = pd.DataFrame([{"experiment": experiment,
                             "assignment_cells_including_layers": len(assignment_cells(experiment)),
                             "core_cells": len({cell["core_cell_index"] for cell in assignment_cells(experiment)}),
                             "cell_count_matches_contract": len({cell["core_cell_index"] for cell in assignment_cells(experiment)}) == {"e2":4,"e3":16,"e4":16}[experiment],
                             "sequences_per_assignment": int((schedule_frame.experiment.eq(experiment) & schedule_frame.cell_index.eq(0)).sum())}
                            for experiment in ("e2","e3","e4")])
    balance.to_csv(out / "schedule_balance_audit.csv", index=False)
    human_sequences = pd.read_csv(config.output_dir / "h0_human_audit" / "human_sequences.csv")
    nuisance = estimate_human_nuisance(human_sequences); _json(out / "power_nuisance.json", nuisance)
    p = config.payload["power"]
    power = simulate_power_planning(n_options=tuple(int(x) for x in p["total_N_options"]),
                                    effects=tuple(float(x) for x in p["standardized_effects"]),
                                    draws=config.simulation_draws, seed=int(p["simulation_seed"]),
                                    residual_sd=nuisance["sequence_log_RT_residual_SD"],
                                    participant_sd=nuisance["participant_intercept_SD"],
                                    participant_missingness=float(p["conservative_participant_missingness"]),
                                    event_error=float(p["conservative_event_error_rate"]))
    power.to_csv(out / "prospective_power_simulation.csv", index=False)
    empirical_status = "awaiting_human_data"
    if human_events is not None:
        source = Path(human_events)
        events = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.DataFrame(json.loads(source.read_text()))
        experiment = str(events.experiment.iloc[0])
        sequences, contrasts, audit = analyze_human_task_events(events, experiment)
        sequences.to_csv(out / f"{experiment}_observed_sequence_results.csv", index=False)
        contrasts.to_csv(out / f"{experiment}_observed_primary_contrasts.csv", index=False)
        _json(out / f"{experiment}_observed_analysis_audit.json", audit)
        empirical_status = "completed"
    task_source = (config.repo_root / "transfer_experiment_program" / "human_task" / "app.js").read_text()
    engineering = {"configs_created": sorted(str(x) for x in configs.values()),
                   "one_target_future_attribute_leak_pattern_absent": "dataset.token" not in task_source and "data-token" not in task_source,
                   "repeat_key_suppression_present": "event.repeat" in task_source,
                   "focus_logging_present": "\"blur\"" in task_source and "\"focus\"" in task_source,
                   "resume_present": "localStorage" in task_source, "synthetic_separated": "synthetic_task_validation" in task_source,
                   "browser_QA_status": "pending_in_app_browser_connection_unavailable",
                   "screenshots_are_live_browser_captures": False,
                   "empirical_status": empirical_status}
    _json(out / "task_engineering_audit.json", engineering)
    update_status(config, {"work_package": "E4", "question": "Does action visibility alter human probe disadvantage?",
                           "implementation_status": "validated_static_browser_QA_pending", "evidence_status": empirical_status,
                           "prerequisites": "new randomized participant observations", "outputs": str(out),
                           "failed_checks": (("" if empirical_status == "completed" else "No new browser-task observations supplied. ")
                                             + "In-app browser screenshot QA unavailable in this execution environment."),
                           "scientific_interpretation_limit": "Task/power/model-information evidence is not a human outcome; no new execution architecture was tested."})
    return {"information": info, "schedules": schedule_frame, "power": power, "engineering": engineering}


def _schema_manifest(config: TransferProgramConfig):
    rows = []
    for path in sorted(config.output_dir.rglob("*.csv")):
        if path.name == "schema_manifest.csv": continue
        try: frame = pd.read_csv(path, nrows=5)
        except Exception: continue
        for column, dtype in frame.dtypes.items():
            rows.append({"table": str(path.relative_to(config.output_dir)), "column": column,
                         "observed_dtype": str(dtype), "description": "See table name and PRD measurement contract."})
    pd.DataFrame(rows).to_csv(config.output_dir / "schema_manifest.csv", index=False)


def _program_manifest(config: TransferProgramConfig) -> pd.DataFrame:
    revision=_revision(config.repo_root)
    rows=[
        {"experiment":"Stage0","cohort":"original_reference","seed":",".join(map(str,config.original_seeds)),"parent_checkpoint":"initialized","parent_hash":"audit/checkpoint_cache","config_code_hash":f"{config.config_hash}:{revision}","initialization":"High=2.0;Low=.02","training_dose":"170 A/B","input_layout_sequence_ids":"repository Stage2/fixed target","manipulation_values":"none"},
        {"experiment":"H0","cohort":"existing_human","seed":"NA","parent_checkpoint":"NA","parent_hash":"source_manifest","config_code_hash":f"{config.config_hash}:{revision}","initialization":"NA","training_dose":"observed chronology","input_layout_sequence_ids":"source labels preserved","manipulation_values":"first/later; local A/B"},
        {"experiment":"E1","cohort":"additional_prespecified","seed":",".join(map(str,config.additional_seeds)),"parent_checkpoint":"0","parent_hash":"audit/checkpoint_cache","config_code_hash":f"{config.config_hash}:{revision}","initialization":"High=2.0;Low=.02","training_dose":"170/340/680 A/B","input_layout_sequence_ids":"repository Stage2/fixed target","manipulation_values":"duration; A/B-only matching"},
        {"experiment":"E2","cohort":"additional_prespecified","seed":",".join(map(str,config.additional_seeds)),"parent_checkpoint":"20,100,170","parent_hash":"audit/checkpoint_cache","config_code_hash":f"{config.config_hash}:{revision}","initialization":"frozen parent","training_dose":"8 repetitions + optional 10-return","input_layout_sequence_ids":"A/B/Mixed/Novel fixed target","manipulation_values":"frozen/learn x reset/carry x primer A/B"},
        {"experiment":"E3","cohort":"original_diagnostic","seed":",".join(map(str,config.original_seeds)),"parent_checkpoint":"20,100,170","parent_hash":"audit/checkpoint_cache","config_code_hash":f"{config.config_hash}:{revision}","initialization":"frozen parent","training_dose":"none","input_layout_sequence_ids":"42-wide factorial adapter/fixed target","manipulation_values":"cue x layout; lambda grid"},
        {"experiment":"E4","cohort":"new_human_planned","seed":"NA","parent_checkpoint":"NA","parent_hash":"NA","config_code_hash":f"{config.config_hash}:{revision}","initialization":"NA","training_dose":"140 acquisition + 8 test + 20 return","input_layout_sequence_ids":"browser A/B/Mixed/Novel","manipulation_values":"full_preview/one_target"},
        {"experiment":"E5","cohort":"additional_prespecified","seed":",".join(map(str,config.additional_seeds)),"parent_checkpoint":"paired initialization","parent_hash":"pretraining_audit","config_code_hash":f"{config.config_hash}:{revision}","initialization":"High=2.0;Low=.02","training_dose":"P0=0;P1/P2=640; target=170, support extension <=680","input_layout_sequence_ids":"8 held-out sequences x 8 layouts","manipulation_values":"P0/P1/P2"},
    ]
    return pd.DataFrame(rows)


def stage_report(config: TransferProgramConfig):
    figures = config.output_dir / "figures"; figures.mkdir(parents=True, exist_ok=True)
    h0 = config.output_dir / "h0_human_audit"; audit = config.output_dir / "audit"
    e1 = config.output_dir / "e1_acquisition"; e2 = config.output_dir / "e2_probe_experience"
    e3 = config.output_dir / "e3_cue_decomposition"; e4 = config.output_dir / "e4_human_task"
    e5 = config.output_dir / "e5_prior_experience"
    required = [h0/"human_sequences.csv", e1/"sequence_results.csv", e2/"primary_outcomes.csv",
                e3/"factorial_effects.csv", e4/"prospective_power_simulation.csv", e5/"paired_contrasts.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError(f"Report prerequisites missing: {missing}")
    human = pd.read_csv(h0/"human_sequences.csv"); refs=pd.read_csv(h0/"human_local_references.csv")
    srn=pd.read_csv(audit/"original_reference_sequence_results.csv")
    e1seq=pd.read_csv(e1/"sequence_results.csv");matches=pd.read_csv(e1/"acquisition_matches.csv")
    e1con=pd.read_csv(e1/"paired_contrasts.csv");split=pd.read_csv(e1/"splitting_trajectories.csv")
    e2out=pd.read_csv(e2/"primary_outcomes.csv");e3eff=pd.read_csv(e3/"factorial_effects.csv")
    e3seq=pd.read_csv(e3/"sequence_results.csv"); info=pd.read_csv(e4/"model_information_timing_audit.csv")
    power=pd.read_csv(e4/"prospective_power_simulation.csv");e5con=pd.read_csv(e5/"paired_contrasts.csv")
    pairing=pd.read_csv(e5/"pretraining_pairing_schedule.csv")
    figure_h0(human,refs,srn,figures);figure_e1(e1seq,matches,figures);figure_differentiation(e1con,split,figures)
    figure_e2(e2out,figures);figure_e3(e3eff,e3seq,figures)
    screenshots=list((e4/"screenshots").glob("*.png")) if (e4/"screenshots").exists() else []
    figure_e4(info,power,screenshots,figures);figure_e5(e5con,pairing,figures)
    synthesis = {"work_package":"Synthesis","question":"Which tested differences explain partial transfer?",
                 "implementation_status":"validated","evidence_status":"completed_computational_human_tasks_await_data",
                 "prerequisites":"completed independent stages","outputs":str(config.output_dir),"failed_checks":"",
                 "scientific_interpretation_limit":"No single manipulation is a unique mechanistic explanation; new human E2-E4 evidence awaits collection."}
    update_status(config,synthesis);status=pd.read_csv(_status_path(config));figure_synthesis(status,figures)
    primary = []
    for value in ("NLL_deficit_over_worse","RT_deficit_over_worse"):
        selected=e1con[e1con.parent_checkpoint.isin([170,340,680])]
        draws,summary=cluster_bootstrap(selected,cluster="seed",value=value,
                                        group=("condition","parent_checkpoint","probe_context"),
                                        draws=config.bootstrap_draws,seed=int(config.payload["bootstrap"]["seed"]))
        draws.assign(metric=value).to_csv(config.output_dir/f"e1_bootstrap_draws_{value}.csv",index=False)
        primary.append(summary.assign(experiment="E1", metric=value))
    e2_interactions=pd.read_csv(e2/"update_x_carry_interactions.csv")
    for value in ("update_x_carry_NLL_deficit_change_first_to_late","update_x_carry_RT_deficit_change_first_to_late"):
        _,summary=cluster_bootstrap(e2_interactions,cluster="seed",value=value,
                                    group=("condition","parent_checkpoint","probe_family","primer"),
                                    draws=config.bootstrap_draws,seed=int(config.payload["bootstrap"]["seed"])+1)
        primary.append(summary.assign(experiment="E2",metric=value))
    for value in ("interaction_NLL","interaction_RT"):
        _,summary=cluster_bootstrap(e3eff,cluster="seed",value=value,
                                    group=("condition","parent_checkpoint","probe_family","anchor"),
                                    draws=config.bootstrap_draws,seed=int(config.payload["bootstrap"]["seed"])+2)
        primary.append(summary.assign(experiment="E3",metric=value))
    e5end=e5con[e5con.parent_checkpoint.eq(170)]
    for value in ("NLL_deficit_over_worse","RT_deficit_over_worse"):
        wide=e5end.pivot_table(index=["condition","seed","probe_context"],columns="arm",values=value,aggfunc="first").dropna(subset=["P1","P2"]).reset_index()
        effect_name=f"P1_minus_P2_{value}";wide[effect_name]=wide.P1-wide.P2
        _,summary=cluster_bootstrap(wide,cluster="seed",value=effect_name,
                                    group=("condition","probe_context"),draws=config.bootstrap_draws,
                                    seed=int(config.payload["bootstrap"]["seed"])+3)
        primary.append(summary.assign(experiment="E5",metric=effect_name))
    uncertainty=pd.concat(primary,ignore_index=True);uncertainty.to_csv(config.output_dir/"primary_uncertainty_summary.csv",index=False)
    report = _render_report(config,status,uncertainty,e2out,e3eff,e5con)
    (config.output_dir/"TRANSFER_EXPERIMENT_PROGRAM_REPORT.md").write_text(report,encoding="utf-8")
    _program_manifest(config).to_csv(config.output_dir/"run_manifest.csv",index=False)
    _schema_manifest(config)
    _json(config.output_dir/"run_status.json",{"status":"completed","version":VERSION,"smoke":config.smoke,
                                               "config_hash":config.config_hash,"human_new_experiments":"awaiting unless supplied",
                                               "code_revision":_revision(config.repo_root)})


def _render_report(config,status,uncertainty,e2out,e3eff,e5con) -> str:
    support = pd.read_csv(config.output_dir/"e1_acquisition"/"acquisition_matches.csv")
    primary_support = support[np.isclose(support.NLL_caliper,.05) & support.selected]
    e2mean=e2out.groupby(["weight_policy","state_policy"])[["NLL_change_first_to_late","RT_change_first_to_late"]].mean()
    e3mean=e3eff.groupby("probe_family")[["interaction_NLL","interaction_RT"]].mean()
    end=e5con[e5con.parent_checkpoint.eq(170)].groupby(["arm","probe_context"])[["NLL_deficit_over_worse","RT_deficit_over_worse"]].mean()
    h0=pd.read_csv(config.output_dir/"h0_human_audit"/"first_later_bootstrap.csv")
    u1=uncertainty[uncertainty.experiment.eq("E1")].dropna(axis=1,how="all")
    u2=uncertainty[uncertainty.experiment.eq("E2")].dropna(axis=1,how="all")
    u3=uncertainty[uncertainty.experiment.eq("E3")].dropna(axis=1,how="all")
    u5=uncertainty[uncertainty.experiment.eq("E5")].dropna(axis=1,how="all")
    chronology=pd.read_csv(config.output_dir/"h0_human_audit"/"chronology_audit.csv")
    strict=pd.read_csv(config.output_dir/"e1_acquisition"/"strict_match_outcomes.csv")
    strict=strict[np.isclose(strict.NLL_caliper,.05)]
    strict_mean=strict.groupby(["high_anchor","probe_context"])[["High_minus_Low_NLL_deficit","High_minus_Low_RT_deficit"]].agg(["mean","count"])
    e2_interaction=pd.read_csv(config.output_dir/"e2_probe_experience"/"update_x_carry_interactions.csv")
    e2_interaction=e2_interaction[(e2_interaction.condition.eq("High_monotonic")) & e2_interaction.parent_checkpoint.eq(170)].groupby("probe_family")[["update_x_carry_NLL_deficit_change_first_to_late","update_x_carry_RT_deficit_change_first_to_late"]].mean()
    e3_high=e3eff[(e3eff.condition.eq("High_monotonic")) & e3eff.parent_checkpoint.eq(170)].groupby("probe_family")[["interaction_NLL","interaction_RT","cue_effect_NLL","layout_effect_NLL"]].mean()
    e5match=pd.read_csv(config.output_dir/"e5_prior_experience"/"acquisition_matches.csv")
    e5_support=e5match.groupby(["condition","seed","candidate_arm"]).support.any()
    return f"""# Human/SRN Transfer Experiment Program Report

Analysis version `{VERSION}`; configuration `{config.config_hash}`. This is {'a two-seed smoke validation and cannot support conclusions' if config.smoke else 'the preregistered 20-seed execution'}.

## Outcome

All five experiment implementations plus H0 are runnable. H0 uses existing empirical keypress data. E1, E2, E3, and E5 use real frozen SRN code and prespecified seeds. E4 and the human components of E2/E3 are engineering-complete at the code/static-test level but remain **awaiting human data**; interactive browser screenshot QA is separately marked pending, and synthetic power rows are never human evidence.

The reference result remains partial transfer, not no transfer. Every computational interpretation must jointly inspect absolute learning, relative probe deficit, and A/B differentiation.

## Evidence status

{status.to_markdown(index=False)}

## H0 — human measurement audit

**Question/intervention/controls.** Chronology and first exposure were reconstructed from the complete source stream before RT/correctness exclusions. Local A/B references were fit only to familiar events, with ridge strength selected by held-out whole A/B blocks; probe outcomes were unavailable to fitting and selection.

**Sample/status.** The audit covered {int(chronology.source_rows.sum()):,} raw events ({int(chronology.target_participants_in_raw.sum())} dataset-namespaced target participants before complete-case filtering). Existing empirical evidence is complete, but legacy elapsed/rest time is unrecoverable and local counterfactual A/B estimates are not simultaneous observations.

{h0.to_markdown(index=False)}

**Result/decision.** The legacy cleaning contract removes original sequence 1 from every block. Consequently the registered first-versus-later inferential contrast is not estimable under that contract; first encounters remain in the audit/local-reference tables as explicitly ineligible descriptive rows and are not renamed. The human first-encounter mismatch therefore remains temporally unresolved; collect the new tasks without changing this exclusion post hoc.

## E1 — acquisition

**Question/intervention/controls.** The additional prespecified cohort continued the unmodified alternating A/B schedule to 340 and 680, with frozen zero-state A/B/Mixed/Novel probes every ten sequences. Matching used labeled A/B NLL, labeled A/B accuracy, and incoming phase only.

**Result.** Strict `.05`-nat matches selected: **{len(primary_support)}**. Unmatched anchors remain in the candidate table. Among valid matches, High-minus-Low relative deficits were:

{strict_mean.to_markdown() if len(strict_mean) else 'No strict matched outcome was estimable.'}

Whole-seed trajectory uncertainty ({config.bootstrap_draws} draws):

{u1.to_markdown(index=False)}

**Remaining explanation/decision.** Valid matched High deficits remain larger in the available matches, so acquisition stage alone is insufficient there; unmatched seeds/anchors remain inconclusive. Duration is not a pure initialization manipulation, and the next decision is to interpret E2/E3/E5 without adding training or selecting favorable matches.

## E2 — state and probe learning

**Question/intervention/controls.** Eight probe repetitions crossed frozen/learn with reset/carry after frozen A/B primers at checkpoints 20/100/170. Parallel A/B-practice controls, probe-adapted same-parameter zero-state endpoints, no-primer references, and isolated A/B return descendants are distinct tables.

**Result.** First-to-late changes (negative means absolute improvement):

{e2mean.to_markdown()}

Whole-seed interaction uncertainty:

{u2.to_markdown(index=False)}

High-scale checkpoint-170 mean update × carry interactions on relative deficit:

{e2_interaction.to_markdown()}

Frozen/reset equality, branch isolation, zero-state adapted endpoints, parallel familiar practice, and separate A/B return descendants are preserved in the branch audit and transition tables. Absolute probe performance can improve while relative deficit worsens; carry and learning are interventions on the SRN, not literal human manipulations. Human history/practice evidence remains awaiting data, so no shared internal process is claimed.

## E3 — cue/layout composition

**Question/intervention/controls.** Original-cohort frozen parents received cue-only, layout-only, combined, and unchanged familiar-anchor inputs at constant physical sequence, targets, dimensionality, parameters, and zero state. Cell 00 and cell 11 reproduce repository encoders exactly.

**Result.** Mean prespecified factorial interactions across scales/checkpoints:

{e3mean.to_markdown()}

Whole-seed factorial uncertainty:

{u3.to_markdown(index=False)}

High-scale checkpoint-170 decomposition:

{e3_high.to_markdown()}

The large negative interactions show non-additivity: cue-only and layout-only costs do not simply sum in the combined input. This localizes sensitivity to composition without proving configural learning. Layout interpolation is labeled an input-space diagnostic and is not instantiated as a psychologically valid human display; the human factorial remains awaiting data.

## E4 — action visibility

**Question/intervention/controls.** The local task crosses between-participant full-preview/one-target visibility with A/B/Mixed/Novel first tests and preceding A/B context. Layout and response counterbalances are layered over the 16 core cells. A/B test arms control the boundary/pause; acquisition and return schedules use shared timing logic.

**Status/decision.** The task implements deterministic assignment, exact event timestamps, focus/abort/resume handling, repeat-key suppression, and a future-label-free one-target DOM. The SRN already receives ordinal, cue, and full-layout information; feeding the scored target directly was not implemented. Power results are prospective simulations, not outcomes. Static/task-contract testing passed, but the in-app browser connection was unavailable in this execution environment, so interactive screenshot QA remains pending. Collecting participants and estimating the visibility interaction are the next empirical steps.

## E5 — prior structural experience

**Question/intervention/controls.** P1 and P2 have identical 640-sequence token streams and sequence/layout/cue marginals; P1 distributes each sequence across layouts while P2 uses a seed-counterbalanced bijection. P0 has no pretraining. Predictive weights carry into identical target acquisition while the execution readout and optimizer buffers reset.

**Result.** Checkpoint-170 arm means:

{end.to_markdown()}

Whole-seed P1-minus-P2 uncertainty:

{u5.to_markdown(index=False)}

A/B-only acquisition support remained absent through 680 for **{int((~e5_support).sum())}** of {len(e5_support)} condition/seed/candidate-arm comparisons. The registered P1-minus-P2 High-scale effect does not show a uniform joint transfer advantage: Mixed NLL deficit is numerically higher for P1, Novel is numerically lower, and both intervals include zero.

The tested structural dose therefore does not solve the joint human target, and unsupported matched comparisons remain inconclusive. No new dose, seed, or architecture is added; any refinement belongs in a separately frozen follow-up.

## Reproduction

```bash
python scripts/run_transfer_experiment_program.py --config configs/transfer_experiment_program.yaml --stage all
python scripts/serve_transfer_human_task.py --port 8765
```

Use `--dry-run` for source/branch/runtime inspection and `--smoke` for a non-inferential two-seed validation. To analyze a real browser export, rerun E4 with `--human-events PATH`.

## Scientific boundary

Negative interventions are informative and do not select a replacement architecture. Human E2–E4 outcomes require new participants. The program does not tune against human probe RTs, add seeds after inspecting outcomes, or turn missing evidence into a null result.
"""


def run_transfer_program(config_path: str | Path, *, stage: str = "all", smoke: bool = False,
                         dry: bool = False, force: bool = False,
                         human_events: str | Path | None = None) -> TransferProgramConfig:
    config = _config_for_run(config_path, smoke)
    torch.set_num_threads(int(config.payload["runtime"]["torch_threads"]))
    if dry:
        dry_run(config); return config
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stages = ORDER if stage == "all" else (stage,)
    if any(item not in ORDER for item in stages): raise ValueError(f"Stage must be one of all/{'/'.join(ORDER)}")
    for item in stages:
        if not force and _already_done(config,item):
            progress(f"{item}: already completed for this config; resuming past it"); continue
        started=time.perf_counter(); progress(f"{item}: starting")
        if item == "audit": stage_audit(config)
        elif item == "human_audit": stage_human_audit(config)
        elif item == "e1":
            result=run_e1(config)
            primary=result["matches"][np.isclose(result["matches"].NLL_caliper,.05)]
            evidence="completed" if primary.selected.any() else "no_common_support"
            update_status(config,{"work_package":"E1","question":"Does the deficit persist at matched acquisition?","implementation_status":"validated","evidence_status":evidence,"prerequisites":"additional-seed A/B training","outputs":str(config.output_dir/"e1_acquisition"),"failed_checks":"" if evidence=="completed" else "No strict .05-nat A/B match passed.","scientific_interpretation_limit":"Duration does not isolate all initialization effects; unsupported strict matches remain inconclusive."})
        elif item == "e2":
            run_e2(config);update_status(config,{"work_package":"E2","question":"Do recurrent history or probe updates alter transfer?","implementation_status":"validated","evidence_status":"completed_computational_awaiting_human_data","prerequisites":"cached additional-seed checkpoints","outputs":str(config.output_dir/"e2_probe_experience"),"failed_checks":"No new human task observations supplied.","scientific_interpretation_limit":"Carry/update effects do not establish Bayesian inference or a matching human optimizer."})
        elif item == "e3":
            run_e3(config);update_status(config,{"work_package":"E3","question":"Does cue, layout, or their conjunction drive the deficit?","implementation_status":"validated","evidence_status":"completed_computational_awaiting_human_data","prerequisites":"cached original-cohort checkpoints","outputs":str(config.output_dir/"e3_cue_decomposition"),"failed_checks":"Human graded layout intentionally uninstantiated.","scientific_interpretation_limit":"Factorial sensitivity localizes inputs but is not a unique configural mechanism."})
        elif item == "e4": stage_e4(config,human_events)
        elif item == "e5":
            result=run_e5(config)
            missing_support=result["matches"].groupby(["condition","seed","candidate_arm"]).support.any().pipe(lambda x: int((~x).sum()))
            evidence="completed" if missing_support==0 else "completed_with_partial_no_common_support"
            update_status(config,{"work_package":"E5","question":"Does cross-layout prior structure improve transfer?","implementation_status":"validated","evidence_status":evidence,"prerequisites":"additional prespecified seeds","outputs":str(config.output_dir/"e5_prior_experience"),"failed_checks":"" if missing_support==0 else f"{missing_support} arm/seed comparisons lacked A/B support through cap 680.","scientific_interpretation_limit":"A dose-specific structural contrast cannot recreate or identify lifetime human experience; unsupported matched comparisons remain inconclusive."})
        elif item == "report": stage_report(config)
        _mark(config,item,started);progress(f"{item}: completed in {time.perf_counter()-started:.1f}s")
    return config
