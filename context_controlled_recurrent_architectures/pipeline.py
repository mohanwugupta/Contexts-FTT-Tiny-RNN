"""Hard-gated architecture experiment from synthetic ecology through intervention."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import torch

from human_svg_geometry_transfer.geometry import load_coordinate_registry
from human_configured_state.data.load_human import load_human_token_data
from human_meta_prior_transfer.config import load_human_meta_transfer_config

from .analysis import decode_context_belief,paired_bootstrap,select_architectures,synthetic_gates
from .human import analyze_human,exact_h8_human_run,interventions,train_human_model
from .models import build_model,model_hash,parameter_count
from .plotting import human_figures,synthetic_figures
from .preflight import load_pilot_selection,verify_freeze_manifest
from .synthetic import load_completed_run,record_chunks,train_run,validate_runs


def _progress(message): print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] CONTEXT-CONTROL | {message}",flush=True)
def _csv(output,name,frame): output.mkdir(parents=True,exist_ok=True);frame.to_csv(output/name,index=False)
def _json(output,name,payload): output.mkdir(parents=True,exist_ok=True);(output/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")


def _worker_init():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass


def _train_cell(job):
    config,architecture,condition,seed,checkpoint=job
    train_run(config,architecture,condition,seed,config.worlds,config.profile,_progress,checkpoint_path=checkpoint)
    return architecture,condition,seed,str(checkpoint)


def _validate_cell(job):
    config,architecture,condition,seed,checkpoint=job
    run=load_completed_run(config,architecture,condition,seed,config.worlds,config.profile,checkpoint)
    result=validate_runs(config,[run],config.validation_worlds)
    _progress(f"validated {config.profile} {architecture}_{condition} seed {seed}")
    return result


def _execute(function,jobs,workers):
    if workers<=1:return [function(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as pool:return list(pool.map(function,jobs))


def _migrate_checkpoint(source,target):
    source=Path(source);target=Path(target)
    if target.exists() or not source.exists() or source.resolve()==target.resolve():return
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(source,temporary);os.replace(temporary,target);_progress(f"migrated checkpoint to local storage: {target.name}")
    finally:
        if temporary.exists():temporary.unlink()


def _stream_records(output,name,checkpoints,kind,columns=None,expected_rows=None):
    destination=Path(output)/name;temporary=destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp");first=True;rows_written=0
    try:
        for checkpoint in checkpoints:
            for frame in record_chunks(checkpoint,kind):
                if columns is not None:frame=frame[columns]
                frame.to_csv(temporary,mode="w" if first else "a",header=first,index=False);first=False;rows_written+=len(frame)
        if first:pd.DataFrame(columns=columns).to_csv(temporary,index=False)
        if expected_rows is not None and rows_written!=expected_rows:raise RuntimeError(f"Incomplete {kind} records: expected {expected_rows}, observed {rows_written}")
        os.replace(temporary,destination)
    finally:
        if temporary.exists():temporary.unlink()


def _registry(config):
    rows=[]
    for spec in config.architecture_specs:
        model=build_model(spec["id"],0); rows.append({"architecture":spec["id"],"label":spec["label"],"family":spec["family"],"hidden_size":spec["hidden_size"],"context_size":spec["context_size"],
            "complexity_rank":spec["complexity_rank"],"parameter_count":parameter_count(model),"synthetic_objective":"next_token_prediction_only","latent_label_input":False,"RT_loss":False})
    return pd.DataFrame(rows)


def _write_empty_human(output,status="not_run_synthetic_gate_not_passed"):
    for name in ("context_control_human_training.csv","context_control_human_familiar_gates.csv","context_control_human_probe_raw.csv","context_control_human_mixed_metrics.csv","context_control_human_novel_metrics.csv","context_control_human_hidden_geometry.csv","context_control_human_context_state.csv","context_control_human_gate_vectors.csv",
                 "context_state_swap.csv","gate_swap.csv","controller_ablation.csv","artificial_interpolation_control.csv"):
        pd.DataFrame({"status":[status]}).to_csv(output/name,index=False)


def _report(output,config,gates,selection,human=None,human_firewall_open=False):
    passing=gates[gates.gate_passed].architecture.tolist();selected=selection[selection.selected_for_human_transfer].architecture.tolist() if "selected_for_human_transfer" in selection else []
    human_status="completed" if human is not None else ("closed by default" if config.profile=="confirmatory" and not human_firewall_open else "not reached")
    (output/"CONTEXT_CONTROLLED_RECURRENT_ARCHITECTURES_REPORT.md").write_text(f"""# Context-Controlled Recurrent Architectures Report

## Run status

- Profile: `{config.profile}`
- Synthetic worlds per cell: {config.worlds}
- Paired seeds: {len(config.seeds)}
- Human Mixed/Novel used for architecture selection: **no**
- Synthetic objective: **next-token prediction only**
- Architectures passing the current synthetic gate: {passing or 'none'}
- Architectures selected for human transfer: {selected or 'none'}
- Human transfer: {human_status}

## Synthetic hard gates

{gates.to_markdown(index=False)}

The primary decision uses the two-dimensional split-versus-ambiguity frontier. Strong differentiation without ambiguous transfer, or ambiguous transfer without differentiation, fails. Structured/shuffled comparisons are paired before permutation and all validation probes are frozen.

## Architecture selection

{selection.to_markdown(index=False)}

The simplest independently passing architecture is preferred and no more than two architectures can advance. Human probe behavior is unavailable to this selection function by construction.
""")


def _human_program(config,runs,selection,registry):
    selected=selection[selection.selected_for_human_transfer].sort_values("selection_rank").architecture.tolist();human_runs=[]
    for seed in config.seeds:
        jobs={(run.architecture,run.condition,run.seed):run for run in runs}
        winner=selected[0]
        for index,architecture in enumerate(selected):
            structured=jobs[(architecture,"structured",seed)];human_runs.append(train_human_model(config,registry,architecture,"structured",seed,structured.model.state_dict()))
            if index==0:
                shuffled=jobs[(architecture,"shuffled",seed)];human_runs.append(train_human_model(config,registry,architecture,"shuffled",seed,shuffled.model.state_dict()))
                random=build_model(architecture,seed+400003);human_runs.append(train_human_model(config,registry,architecture,"random_init",seed,random.state_dict()))
            if architecture in {"B","C"} and config.payload["human_transfer"]["restricted_plasticity_for_context_models"]:
                human_runs.append(train_human_model(config,registry,architecture,"structured",seed,structured.model.state_dict(),restricted=True))
        human_runs.append(exact_h8_human_run(config,registry,seed))
    human=analyze_human(config,human_runs)
    # Unfitted correspondence is computed only after synthetic selection and target transfer.
    transfer_config=load_human_meta_transfer_config(config.human_meta_config,smoke=False);observed=load_human_token_data(transfer_config.human_profile_source,"Combined",target_probe="Both")
    profile=observed[(observed.context.eq("Both"))&observed.probe_order.eq(1)].groupby("transition_position").log_rt.mean().reindex(range(1,8)).to_numpy(float);hz=(profile-profile.mean())/max(profile.std(),1e-12);rows=[]
    for key,one in human["mixed"].groupby(["group","architecture","source","seed","checkpoint"]):
        values=one.sort_values("transition_position").probe_RT.to_numpy(float);mz=(values-values.mean())/max(values.std(),1e-12);rows.append(dict(zip(["group","architecture","source","seed","checkpoint"],key))|{"human_profile_correlation_unfitted":float(np.corrcoef(mz,hz)[0,1]),"human_profile_z_RMSE_unfitted":float(np.sqrt(np.mean((mz-hz)**2))),"human_probe_used_for_selection":False})
    human["mixed"]=human["mixed"].merge(pd.DataFrame(rows),on=["group","architecture","source","seed","checkpoint"],how="left",validate="many_to_one")
    passing=set(human["gates"].query("row_type == 'group' and gate_passed == True").group);causal=interventions(config,registry,human_runs,passing)
    return human,causal


def run_pipeline(config,*,open_human_firewall=False,workers=1,checkpoint_dir=None):
    output=config.output_dir
    workers=max(1,int(workers))
    if open_human_firewall and config.profile!="confirmatory": raise ValueError("The human firewall can only be opened in confirmatory mode.")
    frozen=None
    if config.profile=="confirmatory": frozen=verify_freeze_manifest(config,output)
    output.mkdir(parents=True,exist_ok=True);registry_table=_registry(config)
    if checkpoint_dir is None:
        checkpoint_root=(Path.home()/"ResearchCheckpoints"/"Contexts-FTT-Tiny-RNN"/"context_controlled_recurrent_architectures") if config.profile=="confirmatory" else output/"checkpoints"
    else:
        value=Path(checkpoint_dir).expanduser();checkpoint_root=value.resolve() if value.is_absolute() else (config.repo_root/value).resolve()
    _json(output,"context_control_architecture_config.json",{"analysis_version":config.payload["analysis_version"],"profile":config.profile,"config_sha256":config.config_sha256,"architectures":config.payload["architectures"],"human_probe_selection_access":False,"workers":workers,"checkpoint_dir":str(checkpoint_root)})
    _json(output,"latent_cause_ecology_config.json",config.payload["synthetic"]);_csv(output,"architecture_registry.csv",registry_table)
    architecture_ids=[row["id"] for row in config.architecture_specs]
    if config.profile=="confirmatory":
        architecture_ids=load_pilot_selection(output)
        if tuple(architecture_ids)!=tuple(frozen["selected_architectures"]): raise RuntimeError("Pilot selection differs from the frozen preflight manifest.")
    jobs=[]
    for architecture in architecture_ids:
        for condition in ("structured","shuffled"):
            for seed in config.seeds:
                filename=f"{config.profile}_{architecture}_{condition}_seed_{seed}.pt";checkpoint=checkpoint_root/filename
                _migrate_checkpoint(output/"checkpoints"/filename,checkpoint);jobs.append((config,architecture,condition,seed,checkpoint))
    _progress(f"training {len(jobs)} independent cells with {workers} worker(s); checkpoints: {checkpoint_root}")
    descriptors=_execute(_train_cell,jobs,workers)
    runs=[load_completed_run(config,architecture,condition,seed,config.worlds,config.profile,Path(checkpoint),include_records=config.profile!="confirmatory") for architecture,condition,seed,checkpoint in descriptors]
    validation_parts=_execute(_validate_cell,jobs,workers) if workers>1 else [validate_runs(config,runs,config.validation_worlds,_progress)]
    validation={key:pd.concat([part[key] for part in validation_parts],ignore_index=True) for key in validation_parts[0]}
    belief_summary,belief_raw,continuum=decode_context_belief(validation["belief_raw"],validation["continuum"]);gates,seed_metrics=synthetic_gates(config,validation,belief_summary,config.profile)
    selection,selected=select_architectures(config,gates,registry_table);bootstrap=paired_bootstrap(validation,draws=100 if config.profile=="test" else 10000)
    evidence_columns=["phase","architecture","structure_condition","seed","world_index","clear_A_count","clear_B_count","alternating_clear_count","ambiguous_count","ambiguous_true_A_count","ambiguous_true_B_count","oracle_p_A_mean_clear_A","oracle_p_A_mean_clear_B","oracle_p_A_mean_ambiguous","explicit_latent_label_input"]
    checkpoints=[Path(item[3]) for item in descriptors]
    if config.profile=="confirmatory":
        _progress("streaming chunked training and manifest records to final CSV files")
        expected_rows=len(checkpoints)*config.worlds
        _stream_records(output,"synthetic_training_curves.csv",checkpoints,"training",expected_rows=expected_rows);_stream_records(output,"synthetic_world_manifest.csv",checkpoints,"manifest",expected_rows=expected_rows);_stream_records(output,"synthetic_context_evidence.csv",checkpoints,"manifest",evidence_columns,expected_rows)
    else:
        training=pd.concat([run.training for run in runs],ignore_index=True);manifest=pd.concat([run.manifest for run in runs],ignore_index=True)
        _csv(output,"synthetic_training_curves.csv",training);_csv(output,"synthetic_world_manifest.csv",manifest);_csv(output,"synthetic_context_evidence.csv",manifest[evidence_columns])
    _csv(output,"synthetic_split_metrics.csv",validation["split"]);_csv(output,"synthetic_shared_structure.csv",validation["shared"]);_csv(output,"synthetic_ambiguity_metrics.csv",validation["ambiguity"]);_csv(output,"synthetic_context_belief_decode.csv",belief_summary);_csv(output,"synthetic_context_belief_raw.csv",belief_raw)
    _csv(output,"synthetic_switch_recovery.csv",validation["switch"]);_csv(output,"synthetic_uncertainty_continuum.csv",continuum);_csv(output,"synthetic_conflicting_evidence.csv",validation["conflict"]);_csv(output,"synthetic_validation_audit.csv",validation["audit"]);_csv(output,"synthetic_gate_results.csv",gates);_csv(output,"synthetic_seed_metrics.csv",seed_metrics);_csv(output,"synthetic_paired_bootstrap.csv",bootstrap);_csv(output,"architecture_selection.csv",selection)
    if config.profile=="pilot":
        _csv(output,"synthetic_gate_results_pilot.csv",gates);_csv(output,"architecture_selection_pilot.csv",selection)
    synthetic_figures(output,registry_table,validation,gates,continuum)
    human=None;causal=None
    if config.profile=="confirmatory" and selected and open_human_firewall:
        coordinate_registry=load_coordinate_registry(config.coordinate_registry,{"A":0,"B":1,"Mixed":2,"Novel":3});human,causal=_human_program(config,runs,selection,coordinate_registry)
        for filename,key in (("context_control_human_training.csv","training"),("context_control_human_familiar_gates.csv","gates"),("context_control_human_probe_raw.csv","probes"),("context_control_human_mixed_metrics.csv","mixed"),("context_control_human_novel_metrics.csv","novel"),("context_control_human_hidden_geometry.csv","geometry"),("context_control_human_context_state.csv","context_state"),("context_control_human_gate_vectors.csv","gate_vectors")):_csv(output,filename,human[key])
        for filename,key in (("context_state_swap.csv","context_state_swap"),("gate_swap.csv","gate_swap"),("controller_ablation.csv","controller_ablation"),("artificial_interpolation_control.csv","artificial_interpolation")):_csv(output,filename,causal[key])
    else:_write_empty_human(output,"not_run_human_firewall_closed" if config.profile=="confirmatory" and selected else "not_run_synthetic_gate_not_passed")
    human_figures(output,human,causal);_report(output,config,gates,selection,human,open_human_firewall);_progress("pipeline complete")
    return {"runs":runs,"validation":validation,"gates":gates,"selection":selection,"human":human,"interventions":causal}
