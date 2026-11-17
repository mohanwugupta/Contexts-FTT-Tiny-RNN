"""Generate probe-time A/B counterfactuals from frozen familiar fits only."""

from __future__ import annotations

import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED=("familiar_model_selection.json","familiar_model_selection.sha256",
          "historical_state_space_reproduction.csv","human_state_space_parameter_estimates.csv")
FAMILIAR_MODELS=("Lump","Cross","Split")


def canonicalize_profiles(a,b):
    """Relabel whole profiles from their means; never swap transitions."""
    a=np.asarray(a,float); b=np.asarray(b,float); fast_is_a=float(a.mean())<=float(b.mean())
    return (a.copy(),b.copy(),"A","B") if fast_is_a else (b.copy(),a.copy(),"B","A")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def validate_familiar_inputs(directory):
    root=Path(directory)
    missing=[name for name in REQUIRED if not (root/name).exists()]
    if missing: raise FileNotFoundError(f"Missing corrected familiar outputs: {missing}")
    expected=(root/"familiar_model_selection.sha256").read_text().split()[0]
    actual=sha256(root/"familiar_model_selection.json")
    if expected!=actual: raise ValueError("Frozen familiar selection hash does not match its sidecar.")
    selection=json.loads((root/"familiar_model_selection.json").read_text())
    if selection.get("mixed_rows_used",0) or selection.get("novel_rows_used",0) or selection.get("no_context_rows_used",0):
        raise ValueError("Familiar selection artifact reports probe contamination.")
    hashes={name:sha256(root/name) for name in REQUIRED}
    return selection,hashes


def select_familiar_models(directory,delta_threshold=20.0):
    historical=pd.read_csv(Path(directory)/"historical_state_space_reproduction.csv")
    primary=historical[historical.fit_scope.eq("pre_probe_AB")]
    totals=primary.groupby("model",as_index=False).bic.sum()
    totals["model"]=totals.model.str.replace("SS_","",regex=False).str.title()
    totals=totals.sort_values("bic").reset_index(drop=True)
    totals["delta_bic"]=totals.bic-totals.bic.min()
    winner=str(totals.iloc[0].model)
    plausible=totals.loc[totals.delta_bic<=delta_threshold,"model"].tolist()
    return winner,plausible,totals


def _parameter_tables(directory,combined_tokens):
    estimates=pd.read_csv(Path(directory)/"human_state_space_parameter_estimates.csv")
    estimates=estimates[estimates.model.isin(FAMILIAR_MODELS)].copy()
    first=(combined_tokens[combined_tokens.context.isin(["A","B"])]
        .sort_values(["participant_id","block_order","original_sequence_in_block"])
        .groupby("participant_id").context.first().map({"A":"AFirst","B":"BFirst"}))
    estimates["counterbalance"]=estimates.participant_id.map(first)
    individual={(str(pid),model):dict(zip(g.parameter,g.estimate))
        for (pid,model),g in estimates.groupby(["participant_id","model"],sort=False)}
    template=estimates.groupby(["counterbalance","model","parameter"],as_index=False).estimate.mean()
    templates={(cb,model):dict(zip(g.parameter,g.estimate))
        for (cb,model),g in template.groupby(["counterbalance","model"],sort=False)}
    initial_rows=[]
    for participant,rows in combined_tokens[combined_tokens.context.isin(["A","B"])].groupby("participant_id",sort=False):
        matrix,meta=_profile_matrix(rows)
        initial_rows.append(pd.DataFrame({"participant_id":participant,"counterbalance":first.get(participant),
            "transition_position":range(1,8),"initial_rt":matrix.iloc[0].to_numpy(float)}))
    initial=pd.concat(initial_rows,ignore_index=True).groupby(["counterbalance","transition_position"]).initial_rt.mean()
    initial_templates={cb:g.droplevel("counterbalance").sort_index().to_numpy(float) for cb,g in initial.groupby(level=0)}
    return individual,templates,initial_templates


def _profile_matrix(rows):
    index=["block_id","original_sequence_in_block"]
    matrix=rows.pivot_table(index=index,columns="transition_position",values="observed_rt",aggfunc="first")
    meta=rows.drop_duplicates(index).set_index(index).loc[matrix.index]
    if list(matrix.columns.astype(int))!=list(range(1,8)): raise ValueError("Anchor simulation requires transitions 1..7.")
    return matrix,meta


def _vectors(parameters,model):
    intercept=np.array([parameters[f"intercept_k{k}"] for k in range(1,8)])
    loading=np.array([parameters[f"skill_loading_k{k}"] for k in range(1,8)])
    if model=="Lump":
        return intercept,loading,np.array([parameters[f"A_shared_k{k}"] for k in range(1,8)]),None,None
    aa=np.array([parameters[f"A_A_k{k}"] for k in range(1,8)])
    ab=np.zeros(7) if model=="Split" else np.array([parameters[f"A_cross_k{k}"] for k in range(1,8)])
    bb=np.array([parameters[f"A_B_k{k}"] for k in range(1,8)])
    return intercept,loading,aa,ab,bb


def _simulate_schedule(matrix,meta,parameters,model):
    familiar=np.flatnonzero(meta.context.isin(["A","B"]).to_numpy())
    if not len(familiar): return []
    initial=matrix.iloc[familiar[0]].to_numpy(float)
    intercept,loading,aa,cross,bb=_vectors(parameters,model)
    xa=initial.copy(); xb=initial.copy(); shared=initial.copy(); checkpoint=0; result=[]
    for (_,source_row) in meta.iterrows():
        context=str(source_row.context)
        if model=="Lump": anchor_a=anchor_b=intercept+loading*shared
        else: anchor_a=intercept+loading*xa; anchor_b=intercept+loading*xb
        result.append((checkpoint,anchor_a.copy(),anchor_b.copy()))
        if context=="A":
            if model=="Lump": shared=aa*shared
            else: xa=aa*xa+cross*xb
            checkpoint+=1
        elif context=="B":
            if model=="Lump": shared=aa*shared
            else: xb=bb*xb+cross*xa
            checkpoint+=1
    return result


def _append_probe_rows(rows,participant_uid,pid,dataset,model,source,cb,source_row,block,sequence,
                       checkpoint,anchor_a,anchor_b):
    fast,slow,fast_label,slow_label=canonicalize_profiles(anchor_a,anchor_b)
    for k in range(7):
        rows.append({"dataset":dataset,"participant_uid":participant_uid,"participant_id":pid,
            "anchor_model":model,"anchor_source":source,"counterbalance":cb,
            "probe_type":source_row.probe_type,"probe_phase":source_row.probe_phase,
            "probe_exposure_index":int(source_row.probe_exposure_index),"block":block,"sequence":sequence,
            "familiar_state_checkpoint":checkpoint,"transition_position":k+1,
            "A_anchor":float(anchor_a[k]),"B_anchor":float(anchor_b[k]),
            "Fast_anchor":float(fast[k]),"Slow_anchor":float(slow[k]),
            "fast_raw_label":fast_label,"slow_raw_label":slow_label})


def generate_anchors(tokens,familiar_dir,models,*,return_familiar_profiles=False):
    combined=tokens[tokens.dataset.eq("Combined")]
    individual,_,_=_parameter_tables(familiar_dir,combined)
    rows=[]; template_rows=[]; familiar_profile_rows=[]
    # First simulate every corrected Combined fit with its matched decoder and
    # initialization. These decoded predictions—not separately averaged
    # parameters—define counterbalance templates for out-of-cohort Novel data.
    for participant_uid,participant in combined.groupby("participant_uid",sort=False):
        matrix,meta=_profile_matrix(participant)
        pid=str(meta.participant_id.iloc[0]); cb=str(meta.counterbalance_order.iloc[0]); dataset=str(meta.dataset.iloc[0])
        for model in models:
            parameters=individual.get((pid,model))
            if parameters is None: raise ValueError(f"No participant-specific frozen {model} parameters for {participant_uid}.")
            simulated=_simulate_schedule(matrix,meta,parameters,model)
            for ((block,sequence),source_row),(checkpoint,anchor_a,anchor_b) in zip(meta.iterrows(),simulated):
                for k in range(7):
                    template_rows.append({"participant_uid":participant_uid,"counterbalance":cb,"anchor_model":model,
                        "familiar_state_checkpoint":checkpoint,"transition_position":k+1,
                        "A_anchor":anchor_a[k],"B_anchor":anchor_b[k]})
                    if str(source_row.context) in {"A","B"}:
                        familiar_profile_rows.append({"participant_uid":participant_uid,"anchor_model":model,
                            "probe_type":"Familiar","probe_exposure_index":checkpoint,
                            "block":block,"sequence":sequence,"transition_position":k+1,
                            "A_anchor":anchor_a[k],"B_anchor":anchor_b[k]})
                if bool(source_row.eligible_probe):
                    _append_probe_rows(rows,participant_uid,pid,dataset,model,"participant_specific",cb,
                        source_row,block,sequence,checkpoint,anchor_a,anchor_b)
    template=pd.DataFrame(template_rows).drop_duplicates([
        "participant_uid","counterbalance","anchor_model","familiar_state_checkpoint","transition_position"])
    template=template.groupby(["counterbalance","anchor_model","familiar_state_checkpoint","transition_position"],as_index=False)[["A_anchor","B_anchor"]].mean()
    lookup={(cb,model,checkpoint):(g.sort_values("transition_position").A_anchor.to_numpy(float),
                                    g.sort_values("transition_position").B_anchor.to_numpy(float))
            for (cb,model,checkpoint),g in template.groupby(["counterbalance","anchor_model","familiar_state_checkpoint"])}
    available={key:sorted(g.familiar_state_checkpoint.unique()) for key,g in template.groupby(["counterbalance","anchor_model"])}
    novel=tokens[tokens.dataset.eq("Novel")]
    for participant_uid,participant in novel.groupby("participant_uid",sort=False):
        matrix,meta=_profile_matrix(participant); pid=str(meta.participant_id.iloc[0]); cb=str(meta.counterbalance_order.iloc[0])
        for model in models:
            checkpoints=available[(cb,model)]
            for (block,sequence),source_row in meta.iterrows():
                if not bool(source_row.eligible_probe): continue
                requested=int(source_row.familiar_state_checkpoint)
                matched=requested if requested in checkpoints else min(checkpoints,key=lambda value:abs(value-requested))
                anchor_a,anchor_b=lookup[(cb,model,matched)]
                _append_probe_rows(rows,participant_uid,pid,"Novel",model,"counterbalance_prediction_template",cb,
                    source_row,block,sequence,matched,anchor_a,anchor_b)
    anchor_frame=pd.DataFrame(rows)
    familiar_frame=pd.DataFrame(familiar_profile_rows)
    return (anchor_frame,familiar_frame) if return_familiar_profiles else anchor_frame


def attach_observations(anchors,tokens):
    probes=tokens[tokens.eligible_probe].copy()
    keys=["dataset","participant_uid","participant_id","probe_type","probe_phase","probe_exposure_index",
          "block_id","original_sequence_in_block","transition_position"]
    observed=probes[keys+["observed_rt"]].rename(columns={"block_id":"block","original_sequence_in_block":"sequence"})
    merge_keys=["dataset","participant_uid","participant_id","probe_type","probe_phase","probe_exposure_index",
                "block","sequence","transition_position"]
    result=anchors.merge(observed,on=merge_keys,how="inner",validate="many_to_one")
    if result.empty: raise ValueError("No probe observations matched frozen familiar anchors.")
    return result


def anchor_diagnostics(anchors,absolute_warning_threshold=5000.0):
    """Describe frozen predictions without clipping or probe-driven repair.

    Large decoded values can arise when a familiar fit has a nearly
    unidentified state/loading combination and is projected beyond its fitted
    A/B window.  They are scientific sensitivity information, not values that
    the retrieval stage is allowed to tune or silently winsorize.
    """
    rows=[]
    for (dataset,probe_type,model,source),group in anchors.groupby(
        ["dataset","probe_type","anchor_model","anchor_source"],sort=False
    ):
        values=group[["A_anchor","B_anchor"]].to_numpy(float).reshape(-1)
        finite=values[np.isfinite(values)]
        rows.append({"dataset":dataset,"probe_type":probe_type,"anchor_model":model,
            "anchor_source":source,"n_anchor_values":len(values),
            "finite_fraction":float(len(finite)/len(values)),
            "minimum":float(np.min(finite)) if len(finite) else np.nan,
            "p01":float(np.quantile(finite,.01)) if len(finite) else np.nan,
            "median":float(np.median(finite)) if len(finite) else np.nan,
            "p99":float(np.quantile(finite,.99)) if len(finite) else np.nan,
            "maximum":float(np.max(finite)) if len(finite) else np.nan,
            "absolute_gt_5000_fraction":float(np.mean(np.abs(finite)>absolute_warning_threshold)) if len(finite) else np.nan,
            "range_warning":bool(len(finite)!=len(values) or (len(finite) and np.any(np.abs(finite)>absolute_warning_threshold)))})
    return pd.DataFrame(rows)
