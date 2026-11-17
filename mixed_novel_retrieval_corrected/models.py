"""Matched categorical, continuous, structured, and geometry models."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class RetrievalFit:
    model: str
    parameters: dict[str,float]
    predicted: np.ndarray
    log_likelihood: float
    aicc: float
    bic: float
    rmse: float
    correlation: float
    profile_r2: float
    converged: bool
    n_parameters: int


def continuous_prediction(a,b,q):
    return q*np.asarray(a)+(1-q)*np.asarray(b)


def _normal_logpdf(y,mean,sigma):
    return -.5*np.log(2*np.pi*sigma*sigma)-.5*((y-mean)/sigma)**2


def categorical_loglik(y,a,b,q,sigma):
    q=np.clip(q,1e-12,1-1e-12)
    return np.logaddexp(np.log(q)+_normal_logpdf(y,a,sigma),np.log1p(-q)+_normal_logpdf(y,b,sigma))


def categorical_profile_loglik(y,a,b,profile_codes,q,sigma):
    """Log likelihood for one categorical A/B draw per complete profile."""
    y=np.asarray(y,float); a=np.asarray(a,float); b=np.asarray(b,float)
    codes=np.asarray(profile_codes,int); q=float(np.clip(q,1e-12,1-1e-12))
    result=[]
    for code in np.unique(codes):
        keep=codes==code
        ll_a=float(_normal_logpdf(y[keep],a[keep],sigma).sum())
        ll_b=float(_normal_logpdf(y[keep],b[keep],sigma).sum())
        result.append(np.logaddexp(np.log(q)+ll_a,np.log1p(-q)+ll_b))
    return np.asarray(result,float)


def _profile_codes(frame):
    candidates=["dataset","participant_uid","probe_type","probe_phase","probe_exposure_index","block","sequence"]
    keys=[column for column in candidates if column in frame]
    if not keys: raise ValueError("Retrieval data require a profile identity.")
    codes,uniques=pd.factorize(pd.MultiIndex.from_frame(frame[keys]),sort=False)
    counts=pd.Series(codes).value_counts()
    if not counts.eq(7).all(): raise ValueError("Categorical retrieval requires complete seven-transition profiles.")
    return codes,len(uniques)


def _summary(model,parameters,predicted,ll,y,k,converged,positions=None,n_units=None):
    n=int(n_units or len(y)); aic=2*k-2*ll
    aicc=aic+(2*k*(k+1)/(n-k-1) if n>k+1 else np.inf)
    bic=k*np.log(n)-2*ll; residual=y-predicted; rmse=float(np.sqrt(np.mean(residual**2)))
    position_values=np.asarray(positions if positions is not None else np.tile(np.arange(1,8),len(y)//7),int)
    by=pd.DataFrame({"k":position_values,"y":y,"p":predicted}).groupby("k")[["y","p"]].mean()
    correlation=float(np.corrcoef(by.y,by.p)[0,1]) if by.y.std()>0 and by.p.std()>0 else np.nan
    denominator=float(np.sum((by.y-by.y.mean())**2))
    r2=float(1-np.sum((by.y-by.p)**2)/denominator) if denominator>0 else np.nan
    return RetrievalFit(model,parameters,predicted,float(ll),float(aicc),float(bic),rmse,correlation,r2,bool(converged),k)


def fit_r0(frame):
    y=frame.observed_rt.to_numpy(float); a=frame.A_anchor.to_numpy(float); b=frame.B_anchor.to_numpy(float)
    codes,n_profiles=_profile_codes(frame)
    initial_sigma=max(float(np.std(y)),10.0)
    def objective(z): return -float(categorical_profile_loglik(y,a,b,codes,z[0],np.exp(z[1])).sum())
    fit=minimize(objective,[.5,np.log(initial_sigma)],method="L-BFGS-B",bounds=[(0,1),(np.log(1),np.log(5000))])
    q=float(fit.x[0]); sigma=float(np.exp(fit.x[1])); predicted=continuous_prediction(a,b,q)
    return _summary("R0_categorical",{"q_A":q,"sigma":sigma},predicted,-fit.fun,y,2,fit.success,
        frame.transition_position.to_numpy(int),n_profiles)


def fit_r1(frame):
    y=frame.observed_rt.to_numpy(float); a=frame.A_anchor.to_numpy(float); b=frame.B_anchor.to_numpy(float)
    _,n_profiles=_profile_codes(frame)
    initial_sigma=max(float(np.std(y)),10.0)
    def objective(z):
        return -float(_normal_logpdf(y,continuous_prediction(a,b,z[0]),np.exp(z[1])).sum())
    fit=minimize(objective,[.5,np.log(initial_sigma)],method="L-BFGS-B",bounds=[(0,1),(np.log(1),np.log(5000))])
    q=float(fit.x[0]); sigma=float(np.exp(fit.x[1])); predicted=continuous_prediction(a,b,q)
    return _summary("R1_scalar_continuous",{"q_A":q,"sigma":sigma},predicted,-fit.fun,y,2,fit.success,
        frame.transition_position.to_numpy(int),n_profiles)


def fit_r2(frame,shrinkage=10.0):
    y=frame.observed_rt.to_numpy(float); a=frame.A_anchor.to_numpy(float); b=frame.B_anchor.to_numpy(float)
    _,n_profiles=_profile_codes(frame)
    positions=frame.transition_position.to_numpy(int)-1; initial_sigma=max(float(np.std(y)),10.0)
    def objective(z):
        q=z[:7]; prediction=q[positions]*a+(1-q[positions])*b
        penalty=shrinkage*float(np.sum((q-q.mean())**2))
        return -float(_normal_logpdf(y,prediction,np.exp(z[7])).sum())+penalty
    fit=minimize(objective,np.r_[np.repeat(.5,7),np.log(initial_sigma)],method="L-BFGS-B",
        bounds=[(0,1)]*7+[(np.log(1),np.log(5000))])
    q=fit.x[:7]; sigma=float(np.exp(fit.x[7])); predicted=q[positions]*a+(1-q[positions])*b
    ll=float(_normal_logpdf(y,predicted,sigma).sum())
    params={"q_common":float(q.mean()),"sigma":sigma,"shrinkage":float(shrinkage)}
    params.update({f"q_A_k{k+1}":float(value) for k,value in enumerate(q)})
    return _summary("R2_transition_specific",params,predicted,ll,y,8,fit.success,
        frame.transition_position.to_numpy(int),n_profiles)


def familiar_basis(anchor_frame,rank=2):
    if "observed_rt" in anchor_frame.columns:
        raise ValueError("R3 basis generation accepts familiar anchors only, never probe outcomes.")
    if "probe_type" in anchor_frame and not anchor_frame.probe_type.eq("Familiar").all():
        raise ValueError("R3 basis generation accepts familiar A/B rows only, never probe rows.")
    keys=["participant_uid","anchor_model","probe_type","probe_exposure_index","block","sequence"]
    units=anchor_frame.drop_duplicates(keys+["transition_position"])
    a=units.pivot_table(index=keys,columns="transition_position",values="A_anchor").to_numpy(float)
    b=units.pivot_table(index=keys,columns="transition_position",values="B_anchor").to_numpy(float)
    profiles=np.vstack([a,b]); center=profiles.mean(axis=0)
    _,_,vt=np.linalg.svd(profiles-center,full_matrices=False)
    basis=vt[:rank].copy()
    for row in basis:
        pivot=np.argmax(np.abs(row))
        if row[pivot]<0: row*=-1
    return center,basis


def fit_r3(frame,basis):
    y=frame.observed_rt.to_numpy(float); positions=frame.transition_position.to_numpy(int)-1
    _,n_profiles=_profile_codes(frame)
    baseline=.5*(frame.A_anchor.to_numpy(float)+frame.B_anchor.to_numpy(float))
    design=basis[:,positions].T
    alpha,*_=np.linalg.lstsq(design,y-baseline,rcond=None)
    predicted=baseline+design@alpha; sigma=max(float(np.sqrt(np.mean((y-predicted)**2))),1.0)
    ll=float(_normal_logpdf(y,predicted,sigma).sum()); params={"sigma":sigma}
    params.update({f"alpha_{j+1}":float(v) for j,v in enumerate(alpha)})
    return _summary("R3_familiar_low_rank",params,predicted,ll,y,len(alpha)+1,True,positions+1,n_profiles)


def fit_all(frame,basis,shrinkage=10.0):
    return {"R0":fit_r0(frame),"R1":fit_r1(frame),"R2":fit_r2(frame,shrinkage),"R3":fit_r3(frame,basis)}


def prediction_frame(frame,fits):
    rows=[]
    for key,fit in fits.items():
        current=frame.copy(); current["retrieval_model"]=fit.model; current["predicted_rt"]=fit.predicted
        current["residual"]=current.observed_rt-current.predicted_rt; rows.append(current)
    return pd.concat(rows,ignore_index=True)


def model_comparison(fits,predictive=None):
    rows=[]
    for key,fit in fits.items():
        rows.append({"model_code":key,"model":fit.model,"log_likelihood":fit.log_likelihood,"aicc":fit.aicc,
            "bic":fit.bic,"rmse":fit.rmse,"profile_correlation":fit.correlation,"profile_r2":fit.profile_r2,
            "n_parameters":fit.n_parameters,"converged":fit.converged,
            "predictive_log_likelihood":np.nan if predictive is None else predictive.get(key,np.nan)})
    result=pd.DataFrame(rows); criterion="predictive_log_likelihood" if predictive and np.isfinite(result.predictive_log_likelihood).any() else "log_likelihood"
    result["primary_rank"]=result[criterion].rank(method="min",ascending=False).astype(int)
    return result.sort_values("primary_rank")


def parameter_table(fits,**identity):
    rows=[]
    for key,fit in fits.items():
        for name,value in fit.parameters.items(): rows.append({**identity,"model_code":key,"model":fit.model,"parameter":name,"estimate":value})
    return pd.DataFrame(rows)


def participant_cv(frame,basis,folds=5,seed=0,shrinkage=10.0):
    participants=np.array(sorted(frame.participant_uid.unique())); rng=np.random.default_rng(seed); rng.shuffle(participants)
    assignments={p:i%min(folds,len(participants)) for i,p in enumerate(participants)}; scores={key:0.0 for key in ["R0","R1","R2","R3"]}
    for fold in sorted(set(assignments.values())):
        test=frame[frame.participant_uid.map(assignments).eq(fold)]; train=frame[~frame.participant_uid.map(assignments).eq(fold)]
        fits=fit_all(train,basis,shrinkage)
        for key,fit in fits.items():
            y=test.observed_rt.to_numpy(float); a=test.A_anchor.to_numpy(float); b=test.B_anchor.to_numpy(float)
            sigma=fit.parameters["sigma"]
            if key=="R0":
                codes,_=_profile_codes(test)
                ll=categorical_profile_loglik(y,a,b,codes,fit.parameters["q_A"],sigma)
            elif key=="R1": ll=_normal_logpdf(y,continuous_prediction(a,b,fit.parameters["q_A"]),sigma)
            elif key=="R2":
                q=np.array([fit.parameters[f"q_A_k{k}"] for k in range(1,8)]); pos=test.transition_position.to_numpy(int)-1
                ll=_normal_logpdf(y,q[pos]*a+(1-q[pos])*b,sigma)
            else:
                pos=test.transition_position.to_numpy(int)-1; base=.5*(a+b)
                alpha=np.array([fit.parameters[f"alpha_{j+1}"] for j in range(basis.shape[0])])
                ll=_normal_logpdf(y,base+basis[:,pos].T@alpha,sigma)
            scores[key]+=float(ll.sum())
    return scores


def participant_cv_r0_r1(frame,folds=5,seed=0):
    """Matched participant-fold prediction for the two primary models only."""
    participants=np.array(sorted(frame.participant_uid.unique())); rng=np.random.default_rng(seed); rng.shuffle(participants)
    assignments={p:i%min(folds,len(participants)) for i,p in enumerate(participants)}
    scores={"R0":0.0,"R1":0.0}
    for fold in sorted(set(assignments.values())):
        test=frame[frame.participant_uid.map(assignments).eq(fold)]
        train=frame[~frame.participant_uid.map(assignments).eq(fold)]
        r0=fit_r0(train); r1=fit_r1(train)
        y=test.observed_rt.to_numpy(float); a=test.A_anchor.to_numpy(float); b=test.B_anchor.to_numpy(float)
        codes,_=_profile_codes(test)
        scores["R0"]+=float(categorical_profile_loglik(y,a,b,codes,r0.parameters["q_A"],r0.parameters["sigma"]).sum())
        scores["R1"]+=float(_normal_logpdf(y,continuous_prediction(a,b,r1.parameters["q_A"]),r1.parameters["sigma"]).sum())
    return scores


def geometry(frame):
    rows=[]
    keys=["dataset","participant_uid","participant_id","anchor_model","probe_type","probe_phase","probe_exposure_index","block","sequence"]
    for identity,g in frame.groupby(keys,sort=False):
        g=g.sort_values("transition_position"); p=g.observed_rt.to_numpy(float); a=g.A_anchor.to_numpy(float); b=g.B_anchor.to_numpy(float)
        d=b-a; denom=float(d@d); q_proj=float((p-a)@d/denom) if denom>0 else np.nan
        projection=a+q_proj*d if np.isfinite(q_proj) else a
        fast=np.minimum(a,b); slow=np.maximum(a,b); outside=np.maximum.reduce([fast-p,p-slow,np.zeros(7)])
        rows.append({**dict(zip(keys,identity)),"q_projection_A_to_B":q_proj,
            "q_A_projection":1-q_proj if np.isfinite(q_proj) else np.nan,
            "off_axis_distance":float(np.linalg.norm(p-projection)),"inside_segment":bool(0<=q_proj<=1) if np.isfinite(q_proj) else False,
            "between_endpoint_fraction":float(np.mean((p>=fast)&(p<=slow))),"off_endpoint_deficit":float(outside.mean()),
            "distance_to_A":float(np.linalg.norm(p-a)),"distance_to_B":float(np.linalg.norm(p-b)),
            "distance_to_corridor":float(np.linalg.norm(p-projection)),"prediction_deficit":float(np.mean(p-.5*(a+b)))})
    return pd.DataFrame(rows)
