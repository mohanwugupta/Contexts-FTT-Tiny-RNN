"""Hash-gated primary and explicitly exploratory nonfamiliar probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from human_context_control_transfer.analysis import canonicalize, probe_metrics, vector_columns
from human_context_control_transfer.engine import probe_run
from human_context_control_transfer.h8 import probe_h8
from human_context_control_transfer.human_data import load_profiles

from .selection import verify_selection_lock
from .training import load_checkpoint_run


_PROFILE_CACHE={}


def _human_profiles(config):
    key=(str(config.combined_data),str(config.novel_data))
    if key not in _PROFILE_CACHE:_PROFILE_CACHE[key]=load_profiles(config)
    return _PROFILE_CACHE[key]


def _probe(config, registry, row, probe_name, context, analysis_type):
    run=load_checkpoint_run(config,row); sequence=tuple(config.payload["human_training"]["token_sequence"]); checkpoint=int(row["checkpoint"])
    contexts=("A","B",context)
    if run.architecture=="H8": frame=probe_h8(run,registry,checkpoint,probe_name,contexts,sequence,exposure_index=1,role=analysis_type)
    else: frame=probe_run(run,registry,checkpoint,probe_name,contexts,sequence,exposure_index=1,role=analysis_type)
    frame["analysis_type"]=analysis_type
    if not frame.probe_exposure_index.eq(1).all(): raise AssertionError("T8: first-exposure probe rows changed.")
    return frame


def _controller_distances(canonical):
    if canonical.empty:return pd.DataFrame()
    columns=vector_columns(canonical,"context_state_");rows=[]
    keys=["group","architecture","prior_condition","seed","checkpoint","probe_name"]
    for key,one in canonical[canonical.architecture=="B"].groupby(keys):
        ccols=[c for c in columns if one[c].notna().all()]
        if not ccols:continue
        lookup=one.set_index(["canonical_context","transition_position"])
        raw_lookup=one.set_index(["context","transition_position"])
        target="Mixed" if "Mixed" in set(one.canonical_context) else "Novel"
        for position in range(1,8):
            fast,slow,probe=(lookup.loc[(name,position)][ccols].to_numpy(float) for name in ("Fast","Slow",target))
            actual_a=raw_lookup.loc[("A",position)][ccols].to_numpy(float);actual_b=raw_lookup.loc[("B",position)][ccols].to_numpy(float)
            rows.append(dict(zip(keys,key))|{"probe_context":target,"transition_position":position,
                "controller_distance_to_Fast":float(np.linalg.norm(probe-fast)),"controller_distance_to_Slow":float(np.linalg.norm(probe-slow)),
                "controller_distance_to_A":float(np.linalg.norm(probe-actual_a)),"controller_distance_to_B":float(np.linalg.norm(probe-actual_b))})
    return pd.DataFrame(rows)


def _metrics(canonical, human_profiles):
    mixed,novel,_,hidden,controller=probe_metrics(canonical,human_profiles)
    distances=_controller_distances(canonical)
    merge=["group","architecture","prior_condition","seed","checkpoint","probe_name","probe_context","transition_position"]
    def combine(frame,target):
        value=frame.copy()
        h=hidden[hidden.probe_context==target] if len(hidden) else hidden
        c=controller[controller.probe_context==target] if len(controller) else controller
        for extra in (h,c,distances[distances.probe_context==target] if len(distances) else distances):
            if len(extra):value=value.merge(extra,on=merge,how="left",validate="one_to_one")
        return value
    return combine(mixed,"Mixed"),combine(novel,"Novel")


def run_primary(config,registry):
    payload,selection_hash=verify_selection_lock(config.output_dir); profiles=_human_profiles(config); mixed_parts=[];novel_parts=[]
    for selected in payload["selections"]:
        if selected["gate_status"]!="selected_gate_passed":continue
        row={"group":selected["group"],"architecture":selected["architecture"],"prior_condition":selected["prior_condition"],
             "seed":None,"checkpoint":int(selected["selected_checkpoint"])}
        for seed in config.seeds:
            row["seed"]=seed; mixed_parts.append(_probe(config,registry,row,"SelectedMixed","Mixed","primary_familiar_selected"));novel_parts.append(_probe(config,registry,row,"SelectedNovel","Novel","secondary_selected"))
    mixed_raw=canonicalize(pd.concat(mixed_parts,ignore_index=True)) if mixed_parts else pd.DataFrame(columns=["status"])
    novel_raw=canonicalize(pd.concat(novel_parts,ignore_index=True)) if novel_parts else pd.DataFrame(columns=["status"])
    mixed,novel=_metrics(pd.concat([x for x in (mixed_raw,novel_raw) if "context" in x],ignore_index=True) if any("context" in x for x in (mixed_raw,novel_raw)) else pd.DataFrame(),profiles) if mixed_parts else (pd.DataFrame(columns=["status"]),pd.DataFrame(columns=["status"]))
    for frame in (mixed_raw,novel_raw,mixed,novel):frame["selection_artifact_sha256"]=selection_hash
    return mixed_raw,mixed,novel_raw,novel


def freeze_primary(output,files):
    output=Path(output); hashes={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files}
    payload={"schema_version":1,"status":"primary_familiar_selected_probes_frozen","files":hashes}
    path=output/"primary_probe_freeze.json";path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");return payload


def run_exploratory(config,registry,checkpoint_manifest):
    verify_selection_lock(config.output_dir); freeze_path=Path(config.output_dir)/"primary_probe_freeze.json"
    if not freeze_path.exists():raise FileNotFoundError("Exploratory Mixed sweep requires frozen primary probe outputs.")
    frozen=json.loads(freeze_path.read_text())
    for name,expected in frozen["files"].items():
        if hashlib.sha256((Path(config.output_dir)/name).read_bytes()).hexdigest()!=expected:raise RuntimeError("Primary probe output changed before exploratory sweep.")
    parts=[]
    for row in checkpoint_manifest.to_dict("records"):
        parts.append(_probe(config,registry,row,f"ExploratoryMixed{row['checkpoint']}","Mixed","exploratory_all_checkpoint_sweep"))
    raw=canonicalize(pd.concat(parts,ignore_index=True));profiles=_human_profiles(config);metrics,_,_,_,_=probe_metrics(raw,profiles)
    raw["analysis_type"]="exploratory";metrics["analysis_type"]="exploratory"
    return raw,metrics
