"""Reproduction, forward validation, recovery, and export helpers."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import os

import numpy as np
import pandas as pd

from state_space_bridge.state_space import fit_model_family
from .data import ParticipantData
from .models import MODEL_NAMES, AdjudicationFit, fit_family, fit_model, _lump_states, _two_states, _q_states


CHECKPOINT_VERSION="human-state-space-checkpoints-v4-diagnostics"
_THREADPOOL_GUARD=None
_RECOVERY_STATE=None


def _worker_init():
    global _THREADPOOL_GUARD
    for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[variable]="1"
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_GUARD=threadpool_limits(limits=1)
    except ImportError:
        _THREADPOOL_GUARD=None


def _real_fit_worker(task):
    p_index,data,multistarts,seed,m3b_ridge=task
    return p_index,fit_family(data,multistarts=multistarts,random_seed=seed,m3b_ridge=m3b_ridge)


def _data_hash(data: ParticipantData) -> str:
    digest=hashlib.sha256()
    digest.update(data.participant_id.encode())
    digest.update(np.ascontiguousarray(data.y).tobytes())
    digest.update("|".join(data.contexts).encode())
    digest.update("|".join(map(str,data.blocks)).encode())
    digest.update(np.ascontiguousarray(data.sequence_in_block).tobytes())
    return digest.hexdigest()


def _cache_key(stage,data,**settings):
    payload={"version":CHECKPOINT_VERSION,"stage":stage,"data_hash":_data_hash(data),**settings}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _load_checkpoint(path,key):
    if path is None or not path.exists(): return None
    try:
        with path.open("rb") as stream: payload=pickle.load(stream)
        return payload.get("value") if payload.get("key")==key else None
    except (OSError,EOFError,pickle.UnpicklingError,AttributeError,ValueError):
        return None


def _save_checkpoint(path,key,value):
    if path is None: return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    with temporary.open("wb") as stream:
        pickle.dump({"key":key,"value":value},stream,protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
    temporary.replace(path)


def historical_reproduction(traces: pd.DataFrame):
    rows=[]
    keys=["dataset","counterbalance_group","fit_scope"]
    for identity,trace in traces.groupby(keys,sort=False,dropna=False):
        trace=trace.sort_values("global_step",kind="stable")
        first=fit_model_family(trace)
        reference=fit_model_family(trace.copy())
        for name,fit in first.items():
            ref=reference[name]
            rows.append({**dict(zip(keys,identity)),"model":name,
                "rmse":fit.rmse,"bic":fit.bic,"log_likelihood":fit.log_likelihood,
                "reference_rmse":ref.rmse,"reference_bic":ref.bic,
                "max_abs_parameter_difference":float(np.max(np.abs(fit.parameters-ref.parameters))),
                "max_abs_prediction_difference":float(np.max(np.abs(fit.predicted-ref.predicted))),
                "within_tolerance":bool(np.allclose(fit.parameters,ref.parameters,atol=1e-10,rtol=0)
                                        and np.allclose(fit.predicted,ref.predicted,atol=1e-10,rtol=0)),
                "canonical_engine":"state_space_bridge.state_space",
                "canonical_R_source":"state-space_SRN.Rmd"})
    result=pd.DataFrame(rows)
    if result.empty or not result.within_tolerance.all():
        raise RuntimeError("STOP: historical Lump/Cross/Split reproduction failed.")
    return result


def fit_real_data(participants, *, multistarts, random_seed, m3b_ridge, progress=None,
                  checkpoint_dir=None, workers=1):
    fit_rows=[]; parameter_rows=[]; subject_rows=[]; latent_rows=[]; decomposition_rows=[]; lookup={}
    fits_by_index={}; pending=[]; metadata={}
    for p_index,data in enumerate(participants):
        seed=random_seed+10007*p_index
        key=_cache_key("real",data,multistarts=multistarts,random_seed=seed,m3b_ridge=m3b_ridge)
        path=Path(checkpoint_dir)/f"real_participant_{p_index+1:04d}.pkl" if checkpoint_dir else None
        metadata[p_index]=(key,path,seed)
        fits=_load_checkpoint(path,key)
        if fits is not None:
            fits_by_index[p_index]=fits
            if progress: progress(f"resumed participant {p_index+1}/{len(participants)} ({data.participant_id})")
        else:
            pending.append((p_index,data,multistarts,seed,m3b_ridge))

    if pending and workers>1:
        if progress: progress(f"parallel participant fitting: {len(pending)} pending across {workers} workers")
        with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as executor:
            futures={executor.submit(_real_fit_worker,task):task[0] for task in pending}
            completed=0
            for future in as_completed(futures):
                p_index,fits=future.result(); fits_by_index[p_index]=fits
                key,path,_=metadata[p_index]; _save_checkpoint(path,key,fits); completed+=1
                if progress: progress(f"completed participant {p_index+1}/{len(participants)} ({participants[p_index].participant_id}); {completed}/{len(pending)} new fits")
    else:
        _worker_init()
        for task in pending:
            p_index,fits=_real_fit_worker(task); fits_by_index[p_index]=fits
            key,path,_=metadata[p_index]; _save_checkpoint(path,key,fits)
            if progress: progress(f"fitted participant {p_index+1}/{len(participants)} ({participants[p_index].participant_id})")

    for p_index,data in enumerate(participants):
        fits=fits_by_index[p_index]
        lookup[data.participant_id]=fits
        best=min(fits,key=lambda m:fits[m].bic)
        for model,fit in fits.items():
            start_spread=float(max(fit.start_objectives)-min(fit.start_objectives))
            common={"participant_id":data.participant_id,"model":model,"n_sequences":data.n_sequences,
                    "n_observations":int(data.y.size),"rss":fit.rss,"rmse":fit.rmse,
                    "log_likelihood":fit.log_likelihood,"aic":fit.aic,"bic":fit.bic,
                    "sigma":fit.sigma,"converged":fit.converged,"optimizer_message":fit.optimizer_message,
                    "at_parameter_bound":fit.at_bound,"multistart_objective_spread":start_spread,
                    "verification_objective_gap":fit.verification_objective_gap,
                    "curvature_condition":fit.curvature_condition,
                    "optimization_robust":fit.optimization_robust,
                    "participant_bic_winner":model==best}
            fit_rows.append(common); subject_rows.append(common.copy())
            for kind,values in (("dynamic",fit.parameters),("decoder",fit.decoder)):
                for name,value in values.items():
                    parameter_rows.append({"participant_id":data.participant_id,"model":model,
                                           "parameter_type":kind,"parameter":name,"estimate":value})
            for n in range(data.n_sequences):
                for k in range(7):
                    observed_context=data.contexts[n]
                    exp_value=fit.expression_a[n,k] if observed_context=="A" else fit.expression_b[n,k]
                    latent_rows.append({"participant_id":data.participant_id,"model":model,
                        "global_sequence":n+1,"block_id":data.blocks[n],"sequence_in_block":data.sequence_in_block[n],
                        "context":observed_context,"transition_position":k+1,"observed":data.y[n,k],
                        "predicted":fit.predicted[n,k],"shared_skill":fit.skill_shared[n,k],
                        "skill_A":fit.skill_a[n,k],"skill_B":fit.skill_b[n,k],
                        "persistent_expression_A":fit.expression_a[n,k],
                        "persistent_expression_B":fit.expression_b[n,k],"observed_expression":exp_value})
                    decomposition_rows.append({"participant_id":data.participant_id,"model":model,
                        "global_sequence":n+1,"block_id":data.blocks[n],"context":observed_context,
                        "transition_position":k+1,"observed_rt":data.y[n,k],"predicted_rt":fit.predicted[n,k],
                        "memory_component":fit.memory_contribution[n,k],
                        "expression_component":fit.expression_contribution[n,k],
                        "residual":data.y[n,k]-fit.predicted[n,k]})
    return ({"human_state_space_fit_metrics":pd.DataFrame(fit_rows),
             "human_state_space_parameter_estimates":pd.DataFrame(parameter_rows),
             "human_state_space_subject_scores":pd.DataFrame(subject_rows),
             "human_state_space_latent_states":pd.DataFrame(latent_rows),
             "human_state_space_rt_decomposition":pd.DataFrame(decomposition_rows)},lookup)


def _predict(data: ParticipantData, fit: AdjudicationFit):
    model=fit.model; y=data.y; cx=data.contexts; n=data.n_sequences
    intercept=np.array([fit.decoder[f"intercept_k{k+1}"] for k in range(7)])
    if model in {"Lump","M3a","M3b","M4"}:
        p=np.array([fit.parameters[f"A_shared_k{k+1}"] for k in range(7)])
        shared=_lump_states(p,y)
        skill=np.array([fit.decoder[f"skill_loading_k{k+1}"] for k in range(7)])
        pred=intercept+shared*skill
        if model in {"M3a","M3b"}:
            d=np.array([fit.decoder[f"context_contrast_k{k+1}"] for k in range(7)])
            pred += np.where(cx=="A",.5,-.5)[:,None]*d
        elif model=="M4":
            q,*_=_q_states(cx,fit.parameters["expression_retention"],fit.parameters["expression_learning_rate"])
            loading=np.array([fit.decoder[f"expression_loading_k{k+1}"] for k in range(7)])
            pred += q[:,None]*loading
        return pred
    if model in {"Cross","Split"}:
        split=model=="Split"; vals=[]
        for k in range(7):
            if split: vals.extend([fit.parameters[f"A_A_k{k+1}"],fit.parameters[f"A_B_k{k+1}"]])
            else: vals.extend([fit.parameters[f"A_A_k{k+1}"],fit.parameters[f"A_cross_k{k+1}"],fit.parameters[f"A_B_k{k+1}"]])
        read,*_=_two_states(np.array(vals),y,cx,split)
        loading=np.array([fit.decoder[f"skill_loading_k{k+1}"] for k in range(7)])
        return intercept+read*loading
    if model=="M5":
        shared_p=np.array([fit.parameters[f"A_shared_k{k+1}"] for k in range(7)])
        eta=[]
        for k in range(7): eta.extend([fit.parameters[f"A_eta_A_k{k+1}"],fit.parameters[f"A_eta_B_k{k+1}"]])
        shared=_lump_states(shared_p,y); specific,*_=_two_states(np.array(eta),y,cx,True)
        sl=np.array([fit.decoder[f"shared_loading_k{k+1}"] for k in range(7)])
        el=np.array([fit.decoder[f"specific_loading_k{k+1}"] for k in range(7)])
        return intercept+shared*sl+specific*el
    raise ValueError(model)


def _cv_fold_worker(task):
    (task_id,p_index,participant_id,fold_number,fold,train,through,actual,heldout_contexts,
     test_block,multistarts,random_seed,m3b_ridge)=task
    fold_rows=[]
    for mi,model in enumerate(MODEL_NAMES):
        fit=fit_model(train,model,multistarts=multistarts,
            random_seed=random_seed+p_index*100003+fold*101+mi,m3b_ridge=m3b_ridge)
        pred=_predict(through,fit)[-len(actual):]
        residual=actual-pred; sigma2=max(fit.sigma**2,np.finfo(float).eps)
        ll=float(np.sum(-.5*(np.log(2*np.pi*sigma2)+residual**2/sigma2)))
        mask=heldout_contexts=="A"
        a_profile=actual[mask].mean(axis=0) if np.any(mask) else actual.mean(axis=0)
        p_profile=pred[mask].mean(axis=0) if np.any(mask) else pred.mean(axis=0)
        corr=float(np.corrcoef(a_profile,p_profile)[0,1]) if np.std(a_profile)>0 and np.std(p_profile)>0 else np.nan
        fold_rows.append({"participant_id":participant_id,"model":model,"fold":fold_number,
            "training_blocks":fold,"heldout_block":test_block,"n_train_sequences":train.n_sequences,
            "n_heldout_sequences":len(actual),"heldout_log_likelihood":ll,
            "heldout_rmse":float(np.sqrt(np.mean(residual**2))),"heldout_profile_correlation":corr,
            "parameters_frozen":True,"heldout_rows_used_for_estimation":0,
            "optimization_robust":fit.optimization_robust})
    return task_id,fold_rows


def forward_cv(participants, *, multistarts, random_seed, m3b_ridge, minimum_blocks=2,
               max_folds_per_participant=None, progress=None, checkpoint_dir=None,workers=1):
    rows=[]; pending=[]; metadata={}; task_id=0
    for p_index,data in enumerate(participants):
        block_order=list(dict.fromkeys(data.blocks.tolist()))
        folds=list(range(minimum_blocks,len(block_order)))
        if max_folds_per_participant is not None:
            folds=folds[:int(max_folds_per_participant)]
        for fold in folds:
            train_blocks=set(block_order[:fold]); test_block=block_order[fold]
            train_mask=np.array([b in train_blocks for b in data.blocks]); test_mask=data.blocks==test_block
            if len(set(data.contexts[train_mask]))<2: continue
            train=replace(data,contexts=data.contexts[train_mask],blocks=data.blocks[train_mask],
                sequence_in_block=data.sequence_in_block[train_mask],global_sequence=np.arange(1,train_mask.sum()+1),
                y=data.y[train_mask],historical_y=data.historical_y[train_mask])
            through_mask=train_mask|test_mask
            through=replace(data,contexts=data.contexts[through_mask],blocks=data.blocks[through_mask],
                sequence_in_block=data.sequence_in_block[through_mask],global_sequence=np.arange(1,through_mask.sum()+1),
                y=data.y[through_mask],historical_y=data.historical_y[through_mask])
            key=_cache_key("cv",train,multistarts=multistarts,random_seed=random_seed,
                m3b_ridge=m3b_ridge,test_block=str(test_block),through_hash=_data_hash(through))
            path=(Path(checkpoint_dir)/f"cv_participant_{p_index+1:04d}_fold_{fold-minimum_blocks+1:03d}.pkl") if checkpoint_dir else None
            cached=_load_checkpoint(path,key)
            if cached is not None:
                rows.extend(cached)
                if progress: progress(f"resumed CV {data.participant_id}: block {test_block}")
                continue
            actual=data.y[test_mask]; heldout_contexts=data.contexts[test_mask]
            task=(task_id,p_index,data.participant_id,fold-minimum_blocks+1,fold,train,through,
                  actual,heldout_contexts,test_block,multistarts,random_seed,m3b_ridge)
            pending.append(task); metadata[task_id]=(path,key,data.participant_id,test_block); task_id+=1
    if pending and workers>1:
        if progress: progress(f"parallel forward CV: {len(pending)} pending folds across {workers} workers")
        with ProcessPoolExecutor(max_workers=workers,initializer=_worker_init) as executor:
            futures=[executor.submit(_cv_fold_worker,task) for task in pending]
            for completed,future in enumerate(as_completed(futures),1):
                identifier,fold_rows=future.result(); path,key,pid,block=metadata[identifier]
                rows.extend(fold_rows); _save_checkpoint(path,key,fold_rows)
                if progress: progress(f"completed CV {pid}: block {block}; {completed}/{len(pending)} new folds")
    else:
        _worker_init()
        for task in pending:
            identifier,fold_rows=_cv_fold_worker(task); path,key,pid,block=metadata[identifier]
            rows.extend(fold_rows); _save_checkpoint(path,key,fold_rows)
            if progress: progress(f"completed CV {pid}: block {block}")
    detail=pd.DataFrame(rows)
    detail=detail.sort_values(["participant_id","fold","model"],kind="stable").reset_index(drop=True)
    summary=detail.groupby("model",as_index=False).agg(
        heldout_log_likelihood=("heldout_log_likelihood","sum"),
        mean_heldout_rmse=("heldout_rmse","mean"),median_profile_correlation=("heldout_profile_correlation","median"),
        participants=("participant_id","nunique"),folds=("fold","size"))
    summary=summary.sort_values("heldout_log_likelihood",ascending=False).reset_index(drop=True)
    summary["predictive_rank"]=np.arange(1,len(summary)+1)
    return detail,summary


def _recovery_worker_init(source,generating,true,multistarts,random_seed,m3b_ridge,participant_count):
    global _RECOVERY_STATE
    _worker_init()
    _RECOVERY_STATE=(source,generating,true,multistarts,random_seed,m3b_ridge,participant_count)


def _recovery_replicate_worker(replicate):
    source,generating,true,multistarts,random_seed,m3b_ridge,participant_count=_RECOVERY_STATE
    rng=np.random.default_rng(random_seed+MODEL_NAMES.index(generating)*1000003+replicate*1009)
    aggregate_bic={model:0.0 for model in MODEL_NAMES}
    aggregate_robust={model:True for model in MODEL_NAMES}; recovered={}
    for stratum_index,(template,units,weight) in enumerate(source):
        synthetic=[np.maximum(base+rng.normal(0,noise,size=base.shape),1.0) for base,noise in units]
        mean_y=np.mean(synthetic,axis=0)
        simulated=replace(template,y=mean_y,historical_y=mean_y.mean(axis=1))
        fits=fit_family(simulated,multistarts=multistarts,
            random_seed=random_seed+replicate*1009+MODEL_NAMES.index(generating)*100003+stratum_index*7919,
            m3b_ridge=m3b_ridge)
        for fitted,fit in fits.items():
            aggregate_bic[fitted]+=weight*fit.bic
            aggregate_robust[fitted]=aggregate_robust[fitted] and fit.optimization_robust
        for name,value in fits[generating].parameters.items():
            recovered.setdefault(name,[]).extend([value]*weight)
    selected=min(aggregate_bic,key=aggregate_bic.get)
    replicate_rows=[{"generating_model":generating,"replicate":replicate+1,"fitted_model":fitted,
        "bic":bic,"selected_model":selected,"selected":fitted==selected,
        "selected_correctly":selected==generating,"simulated_participants":participant_count,
        "schedule_strata":len(source),"all_locked_participants_used":True,
        "optimization_robust":aggregate_robust[fitted]} for fitted,bic in aggregate_bic.items()]
    replicate_params=[]
    for name,value in true.items():
        if name in recovered:
            estimate=float(np.mean(recovered[name]))
            replicate_params.append({"generating_model":generating,"replicate":replicate+1,"parameter":name,
                "true_value":value,"recovered_value":estimate,"error":estimate-value,
                "simulated_participants":participant_count})
    return replicate,replicate_rows,replicate_params


def model_recovery(participants: list[ParticipantData], generators: dict[str,dict[str,AdjudicationFit]], *, replicates, multistarts, random_seed, m3b_ridge, progress=None, checkpoint_dir=None,workers=1):
    """Population-size recovery with exact schedule/missingness strata.

    Every locked participant contributes a noisy synthetic trajectory. To keep
    the prespecified 7x7 recovery feasible, participants with identical A/B
    schedules are averaged and the candidate-model BIC is summed across those
    strata. No participant or schedule is dropped.
    """
    rows=[]; params=[]
    strata={}
    for data in participants:
        signature=(tuple(data.contexts),tuple(data.blocks),tuple(data.sequence_in_block))
        strata.setdefault(signature,[]).append(data)
    for generating in MODEL_NAMES:
        true_values={}
        for data in participants:
            for name,value in generators[data.participant_id][generating].parameters.items():
                true_values.setdefault(name,[]).append(value)
        true={name:float(np.mean(values)) for name,values in true_values.items()}
        source=[]
        for members in strata.values():
            units=[]
            for data in members:
                generator=generators[data.participant_id][generating]
                units.append((_predict(data,generator),max(generator.sigma,20.0)))
            source.append((members[0],units,len(members)))
        generator_fingerprint=hashlib.sha256(json.dumps(
            {data.participant_id:generators[data.participant_id][generating].parameters for data in participants},
            sort_keys=True,separators=(",",":")).encode()).hexdigest()
        pending=[]; metadata={}
        for replicate in range(replicates):
            key_payload={"version":CHECKPOINT_VERSION,"stage":"recovery","generating":generating,
                "replicate":replicate,"multistarts":multistarts,"random_seed":random_seed,
                "m3b_ridge":m3b_ridge,"generator_fingerprint":generator_fingerprint,
                "data_hashes":[_data_hash(data) for data in participants]}
            key=hashlib.sha256(json.dumps(key_payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
            path=(Path(checkpoint_dir)/f"recovery_{generating}_replicate_{replicate+1:04d}.pkl") if checkpoint_dir else None
            cached=_load_checkpoint(path,key)
            if cached is not None:
                rows.extend(cached["rows"]); params.extend(cached["params"])
                if progress: progress(f"resumed recovery {generating} replicate {replicate+1}/{replicates}")
                continue
            pending.append(replicate); metadata[replicate]=(path,key)
        initializer_args=(source,generating,true,multistarts,random_seed,m3b_ridge,len(participants))
        if pending and workers>1:
            if progress: progress(f"parallel recovery {generating}: {len(pending)} pending replicates across {workers} workers")
            with ProcessPoolExecutor(max_workers=workers,initializer=_recovery_worker_init,initargs=initializer_args) as executor:
                futures=[executor.submit(_recovery_replicate_worker,replicate) for replicate in pending]
                for completed,future in enumerate(as_completed(futures),1):
                    replicate,replicate_rows,replicate_params=future.result(); path,key=metadata[replicate]
                    rows.extend(replicate_rows); params.extend(replicate_params)
                    _save_checkpoint(path,key,{"rows":replicate_rows,"params":replicate_params})
                    if progress: progress(f"completed recovery {generating} replicate {replicate+1}/{replicates}; {completed}/{len(pending)} new")
        else:
            _recovery_worker_init(*initializer_args)
            for replicate in pending:
                _,replicate_rows,replicate_params=_recovery_replicate_worker(replicate); path,key=metadata[replicate]
                rows.extend(replicate_rows); params.extend(replicate_params)
                _save_checkpoint(path,key,{"rows":replicate_rows,"params":replicate_params})
                if progress: progress(f"completed recovery {generating} replicate {replicate+1}/{replicates}")
    detail=pd.DataFrame(rows)
    detail=detail.sort_values(["generating_model","replicate","fitted_model"],kind="stable").reset_index(drop=True)
    selected=detail[detail.selected].copy()
    confusion=pd.crosstab(selected.generating_model,selected.selected_model,normalize="index").reindex(index=MODEL_NAMES,columns=MODEL_NAMES,fill_value=0).reset_index()
    parameter=pd.DataFrame(params)
    if not parameter.empty:
        grouped=parameter.groupby(["generating_model","parameter"])
        stats=grouped.error.agg(bias="mean",rmse=lambda x:float(np.sqrt(np.mean(x**2)))).reset_index()
        intervals=grouped.recovered_value.quantile([.025,.975]).unstack().reset_index().rename(columns={.025:"recovery_q025",.975:"recovery_q975"})
        parameter=parameter.merge(stats).merge(intervals)
        parameter["covered_95"]=parameter.true_value.between(parameter.recovery_q025,parameter.recovery_q975)
        parameter["coverage_method"]="empirical across schedule-matched recovery replicates"
    return detail,confusion,parameter
