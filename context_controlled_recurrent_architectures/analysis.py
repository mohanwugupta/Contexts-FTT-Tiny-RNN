"""Frozen diagnostic decoding, hard gates, selection, and paired statistics."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _ridge_decoder(x,y,train):
    xm=x[train].mean(0);xs=x[train].std(0);xs[xs<1e-8]=1;z=(x-xm)/xs;zt=z[train];yt=y[train]
    zm=zt.mean(0);ym=yt.mean();zc=zt-zm;yc=yt-ym
    beta=zc.T@np.linalg.solve(zc@zc.T+1e-3*np.eye(len(zc)),yc);intercept=float(ym-zm@beta)
    return xm,xs,intercept,beta,intercept+z@beta


def decode_context_belief(belief_raw,continuum):
    world_summaries=[]; decoded=[]; continuum_parts=[]
    keys=["phase","architecture","structure_condition","seed"]
    for key,all_worlds in belief_raw.groupby(keys,sort=True):
        first_decoder=None
        for world_index,one in all_worlds.groupby("validation_world",sort=True):
            candidates=sorted([c for c in one if c.startswith("representation_")],key=lambda c:int(c.split("_")[-1]));columns=[c for c in candidates if one[c].notna().all()]
            x=one[columns].to_numpy(float);y=one.oracle_p_A.to_numpy(float);train=one.evidence_regime.ne("ambiguous").to_numpy();test=~train
            xm,xs,intercept,beta,raw_prediction=_ridge_decoder(x,y,train);prediction=np.clip(raw_prediction,0,1)
            evaluation=prediction[test];oracle=y[test];correlation=float(np.corrcoef(evaluation,oracle)[0,1]) if np.std(evaluation)>0 and np.std(oracle)>0 else 0.
            world_summaries.append(dict(zip(keys,key))|{"validation_world":world_index,"decoder":"posthoc_within_world_ridge","decoder_fit_regime":"clear_only","decoder_test_regime":"ambiguous_only",
                "posterior_correlation":correlation,"posterior_Brier":float(np.mean((evaluation-oracle)**2)),"calibration_intercept":float(np.mean(evaluation-oracle)),"decoder_used_for_training":False})
            value=one.copy();value["decoded_p_A"]=prediction;value["decoder_partition"]=np.where(train,"clear_fit","ambiguous_held_out");decoded.append(value)
            if first_decoder is None:first_decoder=(columns,xm,xs,intercept,beta)
        subset=continuum
        for column,item in zip(keys,key):subset=subset[subset[column].eq(item)]
        if len(subset) and first_decoder is not None:
            columns,xm,xs,intercept,beta=first_decoder;cx=subset[columns].to_numpy(float);cp=np.clip(intercept+((cx-xm)/xs)@beta,0,1);part=subset.copy();part["decoded_p_A"]=cp;continuum_parts.append(part)
    world=pd.DataFrame(world_summaries);summaries=world.groupby(keys,as_index=False).agg(posterior_correlation=("posterior_correlation","mean"),posterior_Brier=("posterior_Brier","mean"),calibration_intercept=("calibration_intercept","mean"),validation_worlds=("validation_world","nunique"))
    summaries["decoder"]="posthoc_within_world_ridge";summaries["decoder_fit_regime"]="clear_only";summaries["decoder_test_regime"]="ambiguous_only";summaries["decoder_used_for_training"]=False
    return summaries,pd.concat(decoded,ignore_index=True),pd.concat(continuum_parts,ignore_index=True) if continuum_parts else continuum


def synthetic_gates(config,validation,belief_summary,phase):
    split=validation["split"]; shared=validation["shared"]; ambiguity=validation["ambiguity"]; switch=validation["switch"]
    keys=["phase","architecture","structure_condition","seed"]
    split_seed=split.groupby(keys,as_index=False).agg(AB_hidden_distance=("AB_hidden_distance","mean"),clear_accuracy=("clear_accuracy","mean"),stable=("numerically_stable","all"))
    shared_seed=shared.groupby(keys,as_index=False).shared_token_hidden_cosine.mean()
    ambiguity_seed=ambiguity.groupby(keys,as_index=False).ambiguous_query_NLL.mean()
    switch_seed=switch.groupby(keys,as_index=False).prediction_recovery_sequences.mean()
    combined=split_seed.merge(shared_seed,on=keys).merge(ambiguity_seed,on=keys).merge(switch_seed,on=keys).merge(belief_summary,on=keys,how="left")
    rows=[]; required_fraction=float(config.payload["gates"]["seed_fraction"])
    for architecture,one in combined.groupby("architecture"):
        structured=one[one.structure_condition.eq("structured")].set_index("seed"); shuffled=one[one.structure_condition.eq("shuffled")].set_index("seed"); seeds=structured.index.intersection(shuffled.index); required=int(math.ceil(required_fraction*len(seeds)))
        split_direction=structured.loc[seeds].AB_hidden_distance>shuffled.loc[seeds].AB_hidden_distance
        ambiguity_direction=structured.loc[seeds].ambiguous_query_NLL<shuffled.loc[seeds].ambiguous_query_NLL
        joint=split_direction&ambiguity_direction
        checks={"G1_clear_prediction":int(((structured.clear_accuracy>float(config.payload["gates"]["chance_accuracy"]))&structured.stable).sum())>=required,
                "G2_structured_greater_split":int(split_direction.sum())>=required,
                "G3_shared_structure":int((structured.shared_token_hidden_cosine>float(config.payload["gates"]["minimum_shared_hidden_cosine"])).sum())>=required,
                "G4_lower_ambiguous_NLL":int(ambiguity_direction.sum())>=required,
                "G5_context_belief_tracking":int((structured.posterior_correlation>float(config.payload["gates"]["minimum_belief_correlation"])).sum())>=required,
                "G6_switch_reconfiguration":int((structured.prediction_recovery_sequences<=float(config.payload["gates"]["maximum_switch_recovery_sequences"])).sum())>=required,
                "G7_joint_seed_consistency":int(joint.sum())>=required}
        rows.append({"phase":phase,"architecture":architecture,"n_paired_seeds":len(seeds),"required_passing_seeds":required,"split_expected_seed_count":int(split_direction.sum()),
                     "ambiguity_expected_seed_count":int(ambiguity_direction.sum()),"joint_expected_seed_count":int(joint.sum()),
                     "structured_mean_split":structured.AB_hidden_distance.mean(),"shuffled_mean_split":shuffled.AB_hidden_distance.mean(),
                     "structured_mean_ambiguous_NLL":structured.ambiguous_query_NLL.mean(),"shuffled_mean_ambiguous_NLL":shuffled.ambiguous_query_NLL.mean(),
                     **checks,"gate_passed":bool(all(checks.values())),"failure_reason":", ".join(name for name,value in checks.items() if not value)})
    return pd.DataFrame(rows),combined


def select_architectures(config,gates,registry):
    passing=set(gates[gates.gate_passed].architecture); ordered=registry[registry.architecture.isin(passing)].sort_values(["complexity_rank","parameter_count"])
    maximum=int(config.payload["selection"]["maximum_human_architectures"]); selected=ordered.head(maximum).architecture.tolist()
    result=registry.copy(); result["synthetic_gate_passed"]=result.architecture.isin(passing); result["selected_for_human_transfer"]=result.architecture.isin(selected)
    result["selection_rank"]=result.architecture.map({architecture:index+1 for index,architecture in enumerate(selected)})
    result["selection_used_human_probe_data"]=False
    return result,selected


def paired_bootstrap(validation,draws=10000):
    rows=[]; rng=np.random.default_rng(91003)
    for table,metric,direction in ((validation["split"],"AB_hidden_distance","higher"),(validation["ambiguity"],"ambiguous_query_NLL","lower")):
        seed=table.groupby(["phase","architecture","structure_condition","seed"],as_index=False)[metric].mean()
        for (phase,architecture),one in seed.groupby(["phase","architecture"]):
            pivot=one.pivot(index="seed",columns="structure_condition",values=metric).dropna(); delta=(pivot.structured-pivot.shuffled).to_numpy(float)
            if not len(delta): continue
            samples=rng.choice(delta,size=(draws,len(delta)),replace=True).mean(1)
            rows.append({"phase":phase,"architecture":architecture,"metric":metric,"contrast":"structured - shuffled","expected_direction":direction,"n_paired_seeds":len(delta),
                         "mean_paired_difference":float(delta.mean()),"CI95_lower":float(np.quantile(samples,.025)),"CI95_upper":float(np.quantile(samples,.975)),
                         "standardized_effect":float(delta.mean()/delta.std(ddof=1)) if len(delta)>1 and delta.std(ddof=1)>0 else np.nan,"bootstrap_draws":draws})
    return pd.DataFrame(rows)
