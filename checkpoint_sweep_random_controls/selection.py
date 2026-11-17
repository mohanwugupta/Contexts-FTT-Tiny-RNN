"""Familiar-only checkpoint scoring and cryptographic selection barrier.

This module deliberately has no imports from probe/human Mixed or Novel code.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = ("profile_match", "switch_cost_match", "behavioral_separation_match", "CrossSplit_evidence", "hidden_separation")
FORBIDDEN = ("mixed", "novel", "nocontext", "no_context")


def assert_familiar_only(*frames, source_names=()):
    for name in source_names:
        if any(token in str(name).lower() for token in FORBIDDEN): raise ValueError(f"T4: selection cannot read nonfamiliar source {name!r}.")
    for frame in frames:
        if "context" in frame and frame.context.astype(str).str.lower().isin(FORBIDDEN).any():
            raise ValueError("T4: selection received Mixed/Novel/NoContext rows.")
        if "probe_context" in frame and frame.probe_context.astype(str).str.lower().isin(FORBIDDEN).any():
            raise ValueError("T4: selection received nonfamiliar probe metrics.")


def _zprofile(values):
    values = np.asarray(values, float); return (values-values.mean())/max(values.std(), 1e-12)


def _profile(metrics, group, checkpoint):
    one = metrics[(metrics.group==group)&(metrics.checkpoint==checkpoint)&(metrics.row_type=="transition")]
    vectors=[]
    for context in ("A","B"):
        vector=one[one.context==context].groupby("transition_position").normalized_RT.mean().reindex(range(1,8)).to_numpy(float)
        if np.isnan(vector).any(): return None
        vectors.append(vector)
    fast,slow=(vectors[0],vectors[1]) if np.mean(vectors[0])<=np.mean(vectors[1]) else (vectors[1],vectors[0])
    return np.concatenate([fast,slow])


def _cross_split_table(state_space):
    fits=state_space[(state_space.fit_status=="fit")&(state_space.fit_scope=="seed")].copy()
    if not len(fits): return pd.DataFrame(columns=["group","seed","checkpoint","state_space_winner","CrossSplit_evidence","Cross_or_Split"])
    pivot=fits.pivot_table(index=["group","seed","checkpoint"],columns="model_label",values="BIC",aggfunc="first").reset_index()
    best=fits[fits.best_model==True][["group","seed","checkpoint","model_label"]].rename(columns={"model_label":"state_space_winner"})
    result=pivot.merge(best,on=["group","seed","checkpoint"],how="left")
    result["CrossSplit_evidence"]=np.maximum(result["Lump"]-result["Cross"],result["Lump"]-result["Split"])
    result["Cross_or_Split"]=result.state_space_winner.isin(["Cross","Split"])
    return result


def score_and_select(config, familiar_metrics, transfer, state_space, hidden, controller, references):
    assert_familiar_only(familiar_metrics, transfer, state_space, hidden, controller, references,
                         source_names=("checkpoint_familiar_metrics.csv","checkpoint_transfer_indices.csv","checkpoint_state_space.csv"))
    summaries=familiar_metrics.query("row_type == 'seed_summary'").copy(); ss=_cross_split_table(state_space)
    rt_transfer=transfer[transfer.metric=="RT"][["group","seed","checkpoint","switch_cost","cross_context_transfer_index"]]
    seed=summaries.merge(hidden,on=["group","architecture","prior_condition","seed","checkpoint"],suffixes=("","_hidden"))
    seed=seed.merge(rt_transfer,on=["group","seed","checkpoint"],how="left").merge(ss,on=["group","seed","checkpoint"],how="left")
    seed=seed.merge(controller[["group","seed","checkpoint","controller_separation_exceeds_null"]],on=["group","seed","checkpoint"],how="left")
    initial=seed[seed.checkpoint==0].set_index(["group","seed"]).familiar_NLL
    chance=float(config.payload["selection"]["chance_accuracy"]); threshold=float(config.payload["selection"]["nonzero_switch_cost"])
    seed["G1_sequence_learning"]=[bool(row.checkpoint>0 and row.familiar_accuracy>chance and row.familiar_NLL<initial.get((row.group,row.seed),np.inf)) for row in seed.itertuples()]
    seed["G2_hidden_separation"]=seed.hidden_separation_exceeds_null.astype(bool)
    seed["G3_nonzero_switch_cost"]=seed.switch_cost.abs().ge(threshold).fillna(False)
    seed["G4_Cross_or_Split"]=seed.Cross_or_Split.eq(True)
    seed["G5_negative_confidence_RT"]=seed.confidence_RT_correlation.lt(0).fillna(False)
    seed["G6_controller_separation"]=np.where(seed.architecture.eq("B"),seed.controller_separation_exceeds_null.eq(True),True)
    seed["seed_gate_without_group_state_space"]=seed[["G1_sequence_learning","G2_hidden_separation","G3_nonzero_switch_cost","G5_negative_confidence_RT","G6_controller_separation"]].all(axis=1)
    group_gate=[]; required=int(math.ceil(float(config.payload["selection"]["seed_fraction"])*len(config.seeds)))
    for (group,architecture,prior,checkpoint),one in seed[seed.group!="H8"].groupby(["group","architecture","prior_condition","checkpoint"]):
        count=int(one.seed_gate_without_group_state_space.sum()); state_count=int(one.G4_Cross_or_Split.sum())
        group_gate.append({"group":group,"architecture":architecture,"prior_condition":prior,"checkpoint":checkpoint,
            "passing_seed_count":count,"required_passing_seeds":required,"Cross_or_Split_seed_count":state_count,
            "majority_Cross_or_Split":state_count>len(one)/2,"minimum_familiar_gate_passed":count>=required and state_count>len(one)/2})
    gates=pd.DataFrame(group_gate)
    # H8 and human familiar references; only z-profile shape is cross-scale.
    human=references[references.reference_type=="human_familiar"].iloc[0]
    human_a=np.fromstring(human.human_A_profile,sep=";");human_b=np.fromstring(human.human_B_profile,sep=";")
    human_fast,human_slow=(human_a,human_b) if human_a.mean()<=human_b.mean() else (human_b,human_a)
    human_profile=np.r_[human_fast,human_slow]
    rows=[]
    for (group,architecture,prior,checkpoint),one in seed[seed.group!="H8"].groupby(["group","architecture","prior_condition","checkpoint"]):
        profile=_profile(familiar_metrics,group,checkpoint); h8_profile=_profile(familiar_metrics,"H8",checkpoint)
        profile_match=-float(np.mean([np.sqrt(np.mean((_zprofile(profile)-_zprofile(reference))**2)) for reference in (h8_profile,human_profile)])) if profile is not None and h8_profile is not None else np.nan
        h8_transfer=rt_transfer[(rt_transfer.group=="H8")&(rt_transfer.checkpoint==checkpoint)].cross_context_transfer_index.mean()
        human_transfer=float(human.cross_context_transfer_index_RT);switch=one.cross_context_transfer_index.mean()
        references_transfer=[value for value in (h8_transfer,human_transfer) if np.isfinite(value)]
        switch_match=-float(np.mean([abs(switch-value) for value in references_transfer])) if np.isfinite(switch) and references_transfer else np.nan
        h8_sep=summaries[(summaries.group=="H8")&(summaries.checkpoint==checkpoint)].AB_RT_mean_absolute_difference.mean()
        separation=one.AB_RT_mean_absolute_difference.mean(); behavior_match=-abs(separation-h8_sep) if np.isfinite(h8_sep) else np.nan
        rows.append({"group":group,"architecture":architecture,"prior_condition":prior,"checkpoint":checkpoint,
            "profile_match":profile_match,"switch_cost_match":switch_match,"behavioral_separation_match":behavior_match,
            "CrossSplit_evidence":one.CrossSplit_evidence.mean(),"hidden_separation":one.coherent_AB_hidden_distance.mean()})
    components=pd.DataFrame(rows).merge(gates,on=["group","architecture","prior_condition","checkpoint"],how="left")
    for component in COMPONENTS:
        mean=components[component].mean(); std=components[component].std(ddof=0)
        components[f"z_{component}"]=(components[component]-mean)/(std if np.isfinite(std) and std>1e-12 else 1.)
        components[f"rank_{component}"]=components.groupby("group")[component].rank(ascending=False,method="average")
    weights=config.payload["selection"]["component_weights"]
    components["S_familiar"]=sum(float(weights[c])*components[f"z_{c}"] for c in COMPONENTS)
    components["mean_component_rank"]=components[[f"rank_{c}" for c in COMPONENTS]].mean(axis=1)
    components["analysis_inputs"]="familiar_A_B_only"; selected=[]
    for group,one in components.groupby("group",sort=True):
        eligible=one[one.minimum_familiar_gate_passed==True].sort_values(["S_familiar","checkpoint"],ascending=[False,True])
        unconstrained=one.sort_values(["S_familiar","checkpoint"],ascending=[False,True]).iloc[0]
        if len(eligible):
            winner=eligible.iloc[0]; checkpoint=int(winner.checkpoint); status="selected_gate_passed"
        else: winner=unconstrained; checkpoint=None; status="no_familiar_matched_checkpoint"
        record={"group":group,"architecture":winner.architecture,"prior_condition":winner.prior_condition,
            "selected_checkpoint":checkpoint,"S_familiar":float(winner.S_familiar),"gate_status":status,
            "unconstrained_best_checkpoint":int(unconstrained.checkpoint),"selection_granularity":"one_group_level_checkpoint_across_20_seeds",
            "mean_component_rank":float(winner.mean_component_rank)}
        record.update({name:float(winner[name]) if np.isfinite(winner[name]) else None for name in COMPONENTS}); selected.append(record)
    selected_frame=pd.DataFrame(selected)
    if len(selected_frame)!=6 or selected_frame.group.nunique()!=6: raise AssertionError("T11: selection must emit exactly one group-level decision for each primary group.")
    return components, selected_frame, seed


def write_locked_selection(config, selected, source_hashes):
    output=Path(config.output_dir); output.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in selected.to_dict("records"):
        records.append({key:(None if pd.isna(value) else value) for key,value in row.items()})
    payload={"schema_version":1,"analysis_version":config.payload["analysis_version"],"selection_used_nonfamiliar_data":False,
             "checkpoint_selection_scope":"group_not_seed","component_weights":config.payload["selection"]["component_weights"],
             "source_hashes":source_hashes,"selections":records}
    path=output/"selected_familiar_checkpoints.json"; path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    digest=hashlib.sha256(path.read_bytes()).hexdigest(); (output/"selected_familiar_checkpoints.sha256").write_text(digest+"  selected_familiar_checkpoints.json\n")
    selected.to_csv(output/"selected_familiar_checkpoints.csv",index=False)
    return digest


def verify_selection_lock(output):
    output=Path(output); path=output/"selected_familiar_checkpoints.json"; sidecar=output/"selected_familiar_checkpoints.sha256"
    if not path.exists() or not sidecar.exists(): raise FileNotFoundError("T5: locked familiar selection and SHA-256 sidecar are required before probing.")
    expected=sidecar.read_text().split()[0]; observed=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected!=observed: raise RuntimeError("T5: selected_familiar_checkpoints.json changed after familiar-only selection.")
    payload=json.loads(path.read_text())
    if payload.get("selection_used_nonfamiliar_data") is not False: raise RuntimeError("T4: selection artifact is not familiar-only.")
    return payload, observed
