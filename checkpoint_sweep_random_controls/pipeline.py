"""Two-stage familiar selection then frozen nonfamiliar checkpoint sweep."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from human_svg_geometry_transfer.geometry import load_coordinate_registry

from .controls import file_sha256, prepare_sources
from .familiar import checkpoint_state_space, familiar_probes, h8_and_human_reference, summarize_familiar, transfer_indices
from .plotting import make_figures
from .selection import score_and_select, write_locked_selection
from .statistics import frontier_metrics, mixed_effects_models, planned_contrasts
from .training import train_all


SOURCE_FILES=("configs/checkpoint_sweep_random_controls.json","checkpoint_sweep_random_controls/config.py",
    "checkpoint_sweep_random_controls/controls.py","checkpoint_sweep_random_controls/training.py",
    "checkpoint_sweep_random_controls/familiar.py","checkpoint_sweep_random_controls/selection.py",
    "checkpoint_sweep_random_controls/probes.py","checkpoint_sweep_random_controls/statistics.py",
    "checkpoint_sweep_random_controls/plotting.py","checkpoint_sweep_random_controls/pipeline.py",
    "scripts/run_checkpoint_sweep_random_controls.py")


def _progress(message):print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] CHECKPOINT-SWEEP | {message}",flush=True)
def _csv(output,name,frame):
    output.mkdir(parents=True,exist_ok=True)
    if len(frame.columns)==0:frame=pd.DataFrame(columns=["status"])
    frame.to_csv(output/name,index=False)
def _read(path):
    try:return pd.read_csv(path,low_memory=False)
    except pd.errors.EmptyDataError:return pd.DataFrame(columns=["status"])


def _familiar_phase(config):
    output=config.output_dir;output.mkdir(parents=True,exist_ok=True);shutil.copyfile(config.path,output/"checkpoint_sweep_locked_config.json")
    upstream,freeze,source_states,control_manifest,initialization_audit=prepare_sources(config)
    _csv(output,"no_pretraining_control_manifest.csv",control_manifest);_csv(output,"initialization_reconstruction_audit.csv",initialization_audit)
    registry=load_coordinate_registry(config.coordinate_registry,config.payload["coordinate_assignment"])
    _progress("training structured, shuffled, random, and H8 familiar trajectories")
    runs,checkpoint_manifest=train_all(config,registry,source_states,_progress)
    training=pd.concat([run.training for run in runs],ignore_index=True)
    _csv(output,"target_checkpoint_manifest.csv",checkpoint_manifest);_csv(output,"checkpoint_sweep_training.csv",training)
    _progress("running familiar-only probes at 12 checkpoints")
    raw=familiar_probes(config,registry,runs);metrics,hidden,controller=summarize_familiar(config,raw);transfer=transfer_indices(config,runs)
    _csv(output,"checkpoint_familiar_metrics.csv",metrics);_csv(output,"checkpoint_transfer_indices.csv",transfer)
    _csv(output,"checkpoint_hidden_separation.csv",hidden);_csv(output,"checkpoint_controller_separation.csv",controller)
    state=checkpoint_state_space(config,training,_progress);_csv(output,"checkpoint_state_space.csv",state)
    references=h8_and_human_reference(config,metrics,transfer);_csv(output,"checkpoint_h8_reference.csv",references)
    components,selected,seed_gate=score_and_select(config,metrics,transfer,state,hidden,controller,references)
    _csv(output,"familiar_match_component_scores.csv",components);_csv(output,"checkpoint_familiar_gate_seed_details.csv",seed_gate)
    hashes={"analysis_config":config.config_sha256,"upstream_freeze_manifest":file_sha256(config.upstream_output/"confirmatory_freeze_manifest.json"),
            "no_pretraining_control_manifest":file_sha256(output/"no_pretraining_control_manifest.csv"),
            "initialization_reconstruction_audit":file_sha256(output/"initialization_reconstruction_audit.csv"),
            "checkpoint_familiar_metrics":file_sha256(output/"checkpoint_familiar_metrics.csv"),
            "checkpoint_transfer_indices":file_sha256(output/"checkpoint_transfer_indices.csv"),
            "checkpoint_state_space":file_sha256(output/"checkpoint_state_space.csv"),
            "checkpoint_hidden_separation":file_sha256(output/"checkpoint_hidden_separation.csv"),
            "checkpoint_controller_separation":file_sha256(output/"checkpoint_controller_separation.csv"),
            "checkpoint_h8_reference":file_sha256(output/"checkpoint_h8_reference.csv")}
    hashes.update({f"source:{name}":file_sha256(config.repo_root/name) for name in SOURCE_FILES})
    selection_hash=write_locked_selection(config,selected,hashes)
    _progress(f"familiar-only selection frozen: {selection_hash[:12]}…")
    return {"runs":runs,"training":training,"metrics":metrics,"transfer":transfer,"state":state,"hidden":hidden,"controller":controller,
            "references":references,"components":components,"selected":selected,"checkpoint_manifest":checkpoint_manifest}


def _report(output,selected,components,frontier,selected_metrics,contrasts,initialization):
    eligible=selected[selected.gate_status=="selected_gate_passed"]
    primary=selected_metrics.query("row_type == 'seed_summary'") if len(selected_metrics) and "row_type" in selected_metrics else pd.DataFrame()
    primary_summary=primary.groupby("group",as_index=False).agg(between_endpoint_fraction=("between_endpoint_fraction","mean"),mean_RT_deficit=("mean_RT_deficit","mean")) if len(primary) else pd.DataFrame()
    text=f"""# Checkpoint Sweep and Random-Control Report

## Integrity

- Reconstructed zero-pretraining controls: **{len(initialization)} architecture × seed states**.
- Initialization hashes matching both structured and shuffled stored metadata: **{int(initialization.seed_matched_initialization.sum())}/{len(initialization)}**.
- Primary checkpoint selection used familiar A/B data only.
- Selection granularity: one group-level checkpoint across 20 seeds; no per-seed primary optimization.
- All-checkpoint Mixed rows are explicitly exploratory.

## Familiar-only selections

{selected.to_markdown(index=False)}

Gate-passing selected groups: **{len(eligible)}/6**.

## Selected-checkpoint first Mixed

{primary_summary.to_markdown(index=False) if len(primary_summary) else 'No group had a gate-passing familiar-matched checkpoint; primary Mixed probing was correctly withheld.'}

## Planned contrasts

{contrasts[contrasts.analysis_stage.eq('selected_checkpoint_Mixed')].to_markdown(index=False) if len(contrasts) and (contrasts.analysis_stage=='selected_checkpoint_Mixed').any() else 'No eligible selected-checkpoint Mixed contrasts.'}

The exploratory checkpoint frontier is reported separately and was generated only after the familiar selection and primary probe outputs were hash-locked.
"""
    (Path(output)/"CHECKPOINT_SWEEP_AND_RANDOM_CONTROL_REPORT.md").write_text(text)


def _probe_phase(config, familiar=None):
    # Lazy import is the software firewall: familiar selection completes and is
    # locked before this module can read or generate any Mixed/Novel outcome.
    from .probes import freeze_primary,run_exploratory,run_primary
    output=config.output_dir;registry=load_coordinate_registry(config.coordinate_registry,config.payload["coordinate_assignment"])
    manifest=_read(output/"target_checkpoint_manifest.csv")
    required=len(config.checkpoints)*(len(config.groups)*len(config.seeds)+len(config.h8_seeds))
    if len(manifest)!=required or not manifest.checkpoint_exists.astype(bool).all():raise RuntimeError("T3: durable target checkpoint manifest is incomplete.")
    _progress("selection lock verified; running primary selected-checkpoint Mixed then Novel")
    mixed_raw,mixed,novel_raw,novel=run_primary(config,registry)
    names=("selected_checkpoint_mixed_raw.csv","selected_checkpoint_mixed_metrics.csv","selected_checkpoint_novel_raw.csv","selected_checkpoint_novel_metrics.csv")
    for name,frame in zip(names,(mixed_raw,mixed,novel_raw,novel)):_csv(output,name,frame)
    freeze_primary(output,names);_progress("primary probe outputs frozen; starting explicitly exploratory all-checkpoint Mixed sweep")
    all_raw,all_metrics=run_exploratory(config,registry,manifest);_csv(output,"all_checkpoint_mixed_raw.csv",all_raw);_csv(output,"all_checkpoint_mixed_metrics.csv",all_metrics)
    familiar_metrics=_read(output/"checkpoint_familiar_metrics.csv");transfer=_read(output/"checkpoint_transfer_indices.csv")
    hidden=_read(output/"checkpoint_hidden_separation.csv");controller=_read(output/"checkpoint_controller_separation.csv")
    frontier=frontier_metrics(familiar_metrics,transfer,hidden,controller,all_metrics);_csv(output,"checkpoint_frontier_metrics.csv",frontier)
    contrasts,bootstrap=planned_contrasts(config,frontier,mixed);_csv(output,"checkpoint_sweep_planned_contrasts.csv",contrasts);_csv(output,"checkpoint_sweep_bootstrap.csv",bootstrap)
    mixed_models=mixed_effects_models(frontier,mixed);_csv(output,"checkpoint_sweep_mixed_effects.csv",mixed_models)
    training=_read(output/"checkpoint_sweep_training.csv");state=_read(output/"checkpoint_state_space.csv");components=_read(output/"familiar_match_component_scores.csv")
    selected=_read(output/"selected_familiar_checkpoints.csv");references=_read(output/"checkpoint_h8_reference.csv")
    make_figures(output,training,familiar_metrics,transfer,state,components,selected,mixed_raw,mixed,frontier,controller,references)
    initialization=_read(output/"initialization_reconstruction_audit.csv");_report(output,selected,components,frontier,mixed,contrasts,initialization)
    status={"status":"complete","selection_sha256":(output/"selected_familiar_checkpoints.sha256").read_text().split()[0],
            "primary_gate_passing_groups":int(selected.gate_status.eq("selected_gate_passed").sum()),"primary_probe_frozen":True,
            "exploratory_rows":len(all_raw)}
    (output/"checkpoint_sweep_completion_status.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    _progress("checkpoint sweep and random-control pipeline complete")
    return {"mixed":mixed,"novel":novel,"exploratory":all_metrics,"frontier":frontier,"contrasts":contrasts}


def run_pipeline(config,*,phase="all"):
    if phase not in {"all","familiar","probes"}:raise ValueError("phase must be all/familiar/probes")
    familiar=_familiar_phase(config) if phase in {"all","familiar"} else None
    if phase=="familiar":return familiar
    probes=_probe_phase(config,familiar)
    return {"familiar":familiar,"probes":probes}
