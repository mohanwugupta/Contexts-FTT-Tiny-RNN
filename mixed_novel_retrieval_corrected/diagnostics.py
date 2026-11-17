"""Absolute-fit and posterior-predictive diagnostics for frozen retrieval fits."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde,kurtosis,skew

from .models import (
    _normal_logpdf,categorical_loglik,categorical_profile_loglik,
    continuous_prediction,fit_r0,fit_r1,
)


PROFILE_KEYS=["dataset","participant_uid","participant_id","probe_type","probe_phase",
              "probe_exposure_index","block","sequence"]


def _profile_arrays(frame):
    meta=[]; ys=[]; aa=[]; bb=[]
    for identity,group in frame.groupby(PROFILE_KEYS,sort=False,dropna=False):
        group=group.sort_values("transition_position")
        if group.transition_position.tolist()!=list(range(1,8)):
            raise ValueError("Absolute-fit diagnostics require complete transitions 1..7.")
        meta.append(dict(zip(PROFILE_KEYS,identity)))
        ys.append(group.observed_rt.to_numpy(float))
        aa.append(group.A_anchor.to_numpy(float)); bb.append(group.B_anchor.to_numpy(float))
    return pd.DataFrame(meta),np.asarray(ys),np.asarray(aa),np.asarray(bb)


def _simulation_long(meta,a,b,y,model,latent=None):
    n=len(meta); result=meta.loc[meta.index.repeat(7)].reset_index(drop=True)
    result["simulation_id"]=np.repeat(np.arange(1,n+1),7)
    result["transition"]=np.tile(np.arange(1,8),n)
    result["A_anchor"]=a.reshape(-1); result["B_anchor"]=b.reshape(-1)
    result["simulated_RT"]=y.reshape(-1); result["simulation_model"]=model
    result["latent_endpoint"]=np.repeat(latent,7) if latent is not None else "continuous"
    return result


def simulate_R0_profiles(frame,fit,n_profiles=5000,seed=0):
    """Simulate one invariant A/B endpoint identity for each seven-RT profile."""
    meta,_,a,b=_profile_arrays(frame); rng=np.random.default_rng(seed)
    donor=rng.integers(0,len(meta),size=int(n_profiles)); sampled=meta.iloc[donor].reset_index(drop=True)
    aa=a[donor]; bb=b[donor]; choose_a=rng.random(int(n_profiles))<fit.parameters["q_A"]
    latent=np.where(choose_a,"A","B"); mean=np.where(choose_a[:,None],aa,bb)
    values=mean+rng.normal(0,fit.parameters["sigma"],size=mean.shape)
    return _simulation_long(sampled,aa,bb,values,"R0",latent)


def simulate_R1_profiles(frame,fit,n_profiles=5000,seed=1):
    """Simulate profiles around the fitted scalar-interpolation mean."""
    meta,_,a,b=_profile_arrays(frame); rng=np.random.default_rng(seed)
    donor=rng.integers(0,len(meta),size=int(n_profiles)); sampled=meta.iloc[donor].reset_index(drop=True)
    aa=a[donor]; bb=b[donor]
    mean=continuous_prediction(aa,bb,fit.parameters["q_A"])
    values=mean+rng.normal(0,fit.parameters["sigma"],size=mean.shape)
    return _simulation_long(sampled,aa,bb,values,"R1",None)


def profile_geometry_table(frame,value_column,id_column,source):
    rows=[]
    for identifier,group in frame.groupby(id_column,sort=False):
        group=group.sort_values("transition" if "transition" in group else "transition_position")
        p=group[value_column].to_numpy(float); a=group.A_anchor.to_numpy(float); b=group.B_anchor.to_numpy(float)
        d=b-a; denominator=float(d@d)
        q=float((p-a)@d/denominator) if denominator>0 else np.nan
        projection=a+q*d if np.isfinite(q) else a
        rows.append({"source":source,"profile_id":identifier,"q_proj":q,
            "off_axis_distance":float(np.linalg.norm(p-projection)),"mean_RT":float(np.mean(p)),
            "profile_variance":float(np.var(p,ddof=1)),"distance_to_A":float(np.linalg.norm(p-a)),
            "distance_to_B":float(np.linalg.norm(p-b))})
    return pd.DataFrame(rows)


def posterior_geometry(primary,r0_simulation,r1_simulation):
    observed=primary.rename(columns={"transition_position":"transition","observed_rt":"observed_RT"})
    observed_geometry=profile_geometry_table(observed,"observed_RT","participant_uid","Observed")
    r0_geometry=profile_geometry_table(r0_simulation,"simulated_RT","simulation_id","R0")
    r1_geometry=profile_geometry_table(r1_simulation,"simulated_RT","simulation_id","R1")
    return pd.concat([observed_geometry,r0_geometry,r1_geometry],ignore_index=True)


def fit_probe_timeline(observed,winner):
    frame=observed[observed.anchor_model.eq(winner)].copy()
    pieces=[]
    groups=["dataset","probe_type","probe_phase","probe_exposure_index"]
    for _,group in frame.groupby(groups,sort=False,dropna=False):
        r0=fit_r0(group); r1=fit_r1(group)
        current=group.copy()
        current["R0_expected_RT"]=r0.predicted
        current["R1_predicted_RT"]=r1.predicted
        current["R0_q_A"]=r0.parameters["q_A"]; current["R1_q_A"]=r1.parameters["q_A"]
        current["R0_sigma"]=r0.parameters["sigma"]; current["R1_sigma"]=r1.parameters["sigma"]
        current["R0_residual_expected"]=current.observed_rt-current.R0_expected_RT
        current["R1_residual"]=current.observed_rt-current.R1_predicted_RT
        pieces.append(current)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()


def build_full_experiment_fit_long(tokens,observed,familiar_dir,winner):
    """Combine exact frozen familiar predictions with all fitted probe rows."""
    latent=pd.read_csv(Path(familiar_dir)/"human_state_space_latent_states.csv")
    latent=latent[latent.model.eq(winner)].copy()
    combined=tokens[tokens.dataset.eq("Combined")]
    timeline=(combined[["participant_id","block_id","original_sequence_in_block","transition_position",
        "participant_uid","counterbalance_order","global_sequence"]]
        .drop_duplicates(["participant_id","block_id","original_sequence_in_block","transition_position"]))
    familiar=latent.merge(timeline,left_on=["participant_id","block_id","sequence_in_block","transition_position"],
        right_on=["participant_id","block_id","original_sequence_in_block","transition_position"],
        how="left",validate="one_to_one")
    if familiar.global_sequence_y.isna().any():
        raise ValueError("Frozen familiar predictions could not be mapped to the complete experiment timeline.")
    familiar_long=pd.DataFrame({"participant":familiar.participant_id,"participant_uid":familiar.participant_uid,
        "dataset":"Combined","counterbalance":familiar.counterbalance_order,"trial":familiar.global_sequence_y.astype(int),
        "block":familiar.block_id,"sequence":familiar.sequence_in_block,"transition":familiar.transition_position,
        "context":familiar.context,"probe_type":"Familiar","probe_exposure_index":pd.NA,"is_probe":False,
        "observed_RT":familiar.observed,"familiar_model":winner,"A_anchor":np.nan,"B_anchor":np.nan,
        "familiar_predicted_RT":familiar.predicted,"R0_expected_RT":np.nan,"R1_predicted_RT":np.nan,
        "R0_residual_expected":np.nan,"R1_residual":np.nan,"phase":"Familiar"})
    familiar_long["familiar_residual"]=familiar_long.observed_RT-familiar_long.familiar_predicted_RT

    probe=fit_probe_timeline(observed,winner)
    token_timeline=(tokens[["dataset","participant_uid","block_id","original_sequence_in_block","global_sequence"]]
        .drop_duplicates(["dataset","participant_uid","block_id","original_sequence_in_block"]))
    probe=probe.merge(token_timeline,left_on=["dataset","participant_uid","block","sequence"],
        right_on=["dataset","participant_uid","block_id","original_sequence_in_block"],how="left",validate="many_to_one")
    if probe.global_sequence.isna().any(): raise ValueError("Probe predictions could not be mapped to chronological time.")
    probe_long=pd.DataFrame({"participant":probe.participant_id,"participant_uid":probe.participant_uid,
        "dataset":probe.dataset,"counterbalance":probe.counterbalance,"trial":probe.global_sequence.astype(int),
        "block":probe.block,"sequence":probe.sequence,"transition":probe.transition_position,
        "context":probe.probe_type,"probe_type":probe.probe_type,"probe_exposure_index":probe.probe_exposure_index,
        "is_probe":True,"observed_RT":probe.observed_rt,"familiar_model":winner,
        "A_anchor":probe.A_anchor,"B_anchor":probe.B_anchor,"familiar_predicted_RT":np.nan,
        "R0_expected_RT":probe.R0_expected_RT,"R1_predicted_RT":probe.R1_predicted_RT,
        "R0_residual_expected":probe.R0_residual_expected,"R1_residual":probe.R1_residual,"phase":probe.probe_phase})
    probe_long["familiar_residual"]=np.nan
    result=pd.concat([familiar_long,probe_long],ignore_index=True)
    result["chronological_index"]=(result.trial.astype(int)-1)*7+result.transition.astype(int)
    identity=["dataset","participant_uid","block","sequence","transition"]
    if result.duplicated(identity).any():
        raise AssertionError("Each analyzed familiar/probe transition must appear exactly once in the long table.")
    return result.sort_values(["dataset","participant_uid","chronological_index","is_probe"],kind="stable").reset_index(drop=True)


def likelihood_decomposition(primary,r0,r1):
    rows=[]
    for _,group in primary.groupby(PROFILE_KEYS,sort=False,dropna=False):
        group=group.sort_values("transition_position"); y=group.observed_rt.to_numpy(float)
        a=group.A_anchor.to_numpy(float); b=group.B_anchor.to_numpy(float); d=b-a; denominator=float(d@d)
        q_proj=float((y-a)@d/denominator) if denominator>0 else np.nan
        projected=a+q_proj*d if np.isfinite(q_proj) else a
        r1_mean=continuous_prediction(a,b,r1.parameters["q_A"])
        ll0=float(categorical_profile_loglik(y,a,b,np.zeros(7,int),r0.parameters["q_A"],r0.parameters["sigma"])[0])
        ll1=float(_normal_logpdf(y,r1_mean,r1.parameters["sigma"]).sum())
        rows.append({"participant":group.participant_id.iloc[0],"LL_R0":ll0,"LL_R1":ll1,
            "delta_LL_R1_minus_R0":ll1-ll0,"distance_to_A":float(np.linalg.norm(y-a)),
            "distance_to_B":float(np.linalg.norm(y-b)),"distance_to_R1_mean":float(np.linalg.norm(y-r1_mean)),
            "q_proj":q_proj,"off_axis_distance":float(np.linalg.norm(y-projected))})
    return pd.DataFrame(rows)


def transition_likelihood_diagnostics(primary,r0,r1):
    rows=[]
    for transition,group in primary.groupby("transition_position",sort=True):
        y=group.observed_rt.to_numpy(float); a=group.A_anchor.to_numpy(float); b=group.B_anchor.to_numpy(float)
        prediction=continuous_prediction(a,b,r1.parameters["q_A"])
        # Descriptive marginal decomposition only; R0 was fit as a whole-profile mixture.
        ll0=float(categorical_loglik(y,a,b,r0.parameters["q_A"],r0.parameters["sigma"]).sum())
        ll1=float(_normal_logpdf(y,prediction,r1.parameters["sigma"]).sum())
        rows.append({"transition":int(transition),"LL_R0":ll0,"LL_R1":ll1,"delta_LL":ll1-ll0,
            "observed_mean":float(y.mean()),"A_anchor":float(a.mean()),"B_anchor":float(b.mean()),
            "R1_prediction":float(prediction.mean()),"decomposition_type":"descriptive marginal; q not refit"})
    return pd.DataFrame(rows)


def _kde_modes(values):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if len(values)<5 or np.std(values)==0: return np.nan
    try:
        grid=np.linspace(np.quantile(values,.005),np.quantile(values,.995),512)
        density=gaussian_kde(values)(grid)
        return int(np.sum((density[1:-1]>density[:-2])&(density[1:-1]>density[2:])))
    except (ValueError,np.linalg.LinAlgError): return np.nan


def distribution_diagnostics(primary,r0_simulation,r1_simulation,geometry):
    observed=primary.rename(columns={"observed_rt":"value"})
    sources={"Observed":observed.value.to_numpy(float),"R0":r0_simulation.simulated_RT.to_numpy(float),
             "R1":r1_simulation.simulated_RT.to_numpy(float)}
    rows=[]
    for source,values in sources.items():
        geo=geometry[geometry.source.eq(source)]; means=geo.mean_RT.to_numpy(float)
        rows.append({"source":source,"mean":float(np.mean(values)),"SD":float(np.std(values,ddof=1)),
            "skewness":float(skew(values,bias=False)),"kurtosis":float(kurtosis(values,fisher=True,bias=False)),
            "q_proj_mean":float(geo.q_proj.mean()),"q_proj_SD":float(geo.q_proj.std(ddof=1)),
            "q_proj_q025":float(geo.q_proj.quantile(.025)),"q_proj_q25":float(geo.q_proj.quantile(.25)),
            "q_proj_median":float(geo.q_proj.median()),"q_proj_q75":float(geo.q_proj.quantile(.75)),
            "q_proj_q975":float(geo.q_proj.quantile(.975)),
            "off_axis_distance_mean":float(geo.off_axis_distance.mean()),
            "off_axis_distance_SD":float(geo.off_axis_distance.std(ddof=1)),
            "hartigan_dip_statistic":np.nan,"hartigan_dip_available":False,
            "estimated_KDE_modes":_kde_modes(means)})
    return pd.DataFrame(rows)


def posterior_predictive_summary(geometry,replicates=1000,seed=0):
    metrics={"median_q_proj":lambda d:float(d.q_proj.median()),
        "SD_q_proj":lambda d:float(d.q_proj.std(ddof=1)),
        "mean_off_axis_distance":lambda d:float(d.off_axis_distance.mean()),
        "mean_RT":lambda d:float(d.mean_RT.mean()),
        "mean_profile_variance":lambda d:float(d.profile_variance.mean())}
    observed=geometry[geometry.source.eq("Observed")]; n=len(observed); rng=np.random.default_rng(seed); rows=[]
    for model in ["R0","R1"]:
        pool=geometry[geometry.source.eq(model)].reset_index(drop=True)
        for name,function in metrics.items():
            draws=[]
            for _ in range(int(replicates)):
                sample=pool.iloc[rng.integers(0,len(pool),size=n)]
                draws.append(function(sample))
            estimate=function(observed); low,high=np.quantile(draws,[.025,.975])
            rows.append({"model":model,"metric":name,"observed":estimate,"predictive_median":float(np.median(draws)),
                "predictive_ci_low":float(low),"predictive_ci_high":float(high),
                "observed_within_95_predictive_interval":bool(low<=estimate<=high)})
    return pd.DataFrame(rows)


def uncertainty_summary(frame,delta_column="delta_LL"):
    values=frame[delta_column].dropna()
    return pd.DataFrame([{"P_delta_LL_lt_0":float((values<0).mean()),"P_delta_LL_gt_0":float((values>0).mean()),
        "median_delta_LL":float(values.median()),"ci_low_delta_LL":float(values.quantile(.025)),
        "ci_high_delta_LL":float(values.quantile(.975)),"replicates":len(values)}])


MANDATORY_CSV=("full_experiment_fit_long.csv","first_mixed_distribution_diagnostics.csv",
    "R0_R1_likelihood_decomposition.csv","transition_likelihood_diagnostics.csv",
    "anchor_uncertainty_R0_R1.csv","participant_bootstrap_R0_R1.csv")
MANDATORY_FIGURES=tuple(f"figure_{letter}_{name}.png" for letter,name in [
    ("A","full_familiar_fit"),("B","full_probe_timeline"),("C","first_mixed_profile_fit"),
    ("D","first_mixed_raw_distributions"),("E","participant_projection_distribution"),
    ("F","posterior_predictive_geometry"),("G","residuals_full_timeline"),
    ("H","observed_vs_predicted"),("I","participant_delta_LL"),("J","anchor_uncertainty_delta_LL")])


def assert_absolute_fit_bundle(output):
    root=Path(output); missing=[name for name in MANDATORY_CSV+MANDATORY_FIGURES if not (root/name).exists()]
    if missing:
        raise RuntimeError("Absolute-fit diagnostics incomplete; do not interpret model-selection results. "
                           f"Missing: {missing}")
    return True
