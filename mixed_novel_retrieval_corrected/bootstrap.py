from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd

from .models import fit_r0,fit_r1,geometry,participant_cv_r0_r1


_STATE=None
_THREADS=None


def _init(frame,seed):
    global _STATE,_THREADS
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
        os.environ[key]="1"
    try:
        from threadpoolctl import threadpool_limits
        _THREADS=threadpool_limits(limits=1)
    except ImportError: pass
    _STATE=(frame,seed)


def _draw(index):
    frame,seed=_STATE; rng=np.random.default_rng(seed+index*104729)
    participants=frame.participant_uid.unique(); sampled=rng.choice(participants,len(participants),replace=True)
    pieces=[]
    for copy,participant in enumerate(sampled):
        part=frame[frame.participant_uid.eq(participant)].copy(); part["participant_uid"]=part.participant_uid+f"#boot{copy}"
        pieces.append(part)
    sample=pd.concat(pieces,ignore_index=True); r0=fit_r0(sample); r1=fit_r1(sample); geo=geometry(sample)
    return {"bootstrap":index+1,"q_R0":r0.parameters["q_A"],"q_R1":r1.parameters["q_A"],
        "log_likelihood_R0":r0.log_likelihood,"log_likelihood_R1":r1.log_likelihood,
        "delta_log_likelihood_R1_minus_R0":r1.log_likelihood-r0.log_likelihood,
        "profile_r2_R0":r0.profile_r2,"profile_r2_R1":r1.profile_r2,
        "between_endpoint_fraction":float(geo.between_endpoint_fraction.mean()),
        "off_axis_distance":float(geo.off_axis_distance.mean())}


def participant_bootstrap(frame,replicates,seed,workers=1,progress=None):
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,initializer=_init,initargs=(frame,seed)) as pool:
            rows=[]
            for index,row in enumerate(pool.map(_draw,range(replicates)),1):
                rows.append(row)
                if progress and (index==1 or index%max(1,replicates//10)==0): progress(f"bootstrap {index}/{replicates}")
    else:
        _init(frame,seed); rows=[]
        for index in range(replicates): rows.append(_draw(index))
    return pd.DataFrame(rows)


def percentile_summary(draws,columns):
    rows=[]
    for column in columns:
        values=draws[column].dropna()
        rows.append({"metric":column,"estimate":float(values.mean()),"ci_low":float(values.quantile(.025)),
                     "ci_high":float(values.quantile(.975)),"bootstrap_replicates":len(values)})
    return pd.DataFrame(rows)


def anchor_uncertainty(primary,anchor_only,replicates,seed,folds=5,progress=None):
    """Propagate familiar donor uncertainty without looking at probe outcomes."""
    unit_keys=["participant_uid","counterbalance","probe_exposure_index","block","sequence"]
    donors=anchor_only[anchor_only.probe_type.eq("Mixed")].drop_duplicates(unit_keys+["transition_position"])
    rows=[]
    for draw in range(replicates):
        rng=np.random.default_rng(seed+700001+draw*7919); replacement={}
        for cb,g in donors.groupby("counterbalance"):
            ids=g.participant_uid.unique(); sampled=rng.choice(ids,len(ids),replace=True)
            sample=pd.concat([g[g.participant_uid.eq(pid)] for pid in sampled],ignore_index=True)
            profile=sample.groupby("transition_position")[["A_anchor","B_anchor"]].mean()
            replacement[cb]=profile
        perturbed=primary.copy()
        perturbed[["A_anchor","B_anchor"]]=perturbed[["A_anchor","B_anchor"]].astype(float)
        for cb,profile in replacement.items():
            mask=perturbed.counterbalance.eq(cb); pos=perturbed.loc[mask,"transition_position"]
            perturbed.loc[mask,"A_anchor"]=pos.map(profile.A_anchor).to_numpy()
            perturbed.loc[mask,"B_anchor"]=pos.map(profile.B_anchor).to_numpy()
        r0=fit_r0(perturbed); r1=fit_r1(perturbed)
        predictive=participant_cv_r0_r1(perturbed,folds,seed)
        rows.append({"draw":draw+1,"bootstrap_id":draw+1,"analysis":"propagated_familiar_anchor_uncertainty",
            "q_R0":r0.parameters["q_A"],"q_R1":r1.parameters["q_A"],
            "LL_R0":predictive["R0"],"LL_R1":predictive["R1"],"delta_LL":predictive["R1"]-predictive["R0"],
            "fitted_LL_R0":r0.log_likelihood,"fitted_LL_R1":r1.log_likelihood,
            "delta_log_likelihood_R1_minus_R0":predictive["R1"]-predictive["R0"]})
        if progress and ((draw+1)==1 or (draw+1)%max(1,replicates//10)==0):
            progress(f"anchor uncertainty {draw+1}/{replicates}")
    return pd.DataFrame(rows)
