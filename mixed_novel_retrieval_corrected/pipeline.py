from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .anchors import validate_familiar_inputs,select_familiar_models,generate_anchors,attach_observations,anchor_diagnostics,sha256,FAMILIAR_MODELS
from .bootstrap import participant_bootstrap,percentile_summary,anchor_uncertainty
from .config import RetrievalConfig
from .data import load_all,probe_timeline
from .models import fit_all,fit_r0,fit_r1,familiar_basis,participant_cv,prediction_frame,model_comparison,parameter_table,geometry
from .diagnostics import (simulate_R0_profiles,simulate_R1_profiles,posterior_geometry,
    build_full_experiment_fit_long,likelihood_decomposition,transition_likelihood_diagnostics,
    distribution_diagnostics,posterior_predictive_summary,uncertainty_summary,assert_absolute_fit_bundle)


def _progress(config,message):
    if config.show_progress:
        print(f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] RETRIEVAL-CORRECTED | {message}",flush=True)


def _fit_by_exposure(frame,basis,config):
    params=[]; profiles=[]
    for (phase,index),group in frame.groupby(["probe_phase","probe_exposure_index"],sort=False):
        if group.participant_uid.nunique()<2: continue
        fits={"R0":fit_r0(group),"R1":fit_r1(group)}
        params.append(parameter_table(fits,probe_phase=phase,probe_exposure_index=index))
        profiles.append(prediction_frame(group,fits))
    return (pd.concat(params,ignore_index=True) if params else pd.DataFrame(),
            pd.concat(profiles,ignore_index=True) if profiles else pd.DataFrame())


def _probe_metrics(frame,basis,config):
    rows=[]
    if frame.empty: return pd.DataFrame()
    for (phase,index),group in frame.groupby(["probe_phase","probe_exposure_index"],sort=False):
        fits={"R0":fit_r0(group),"R1":fit_r1(group)}
        for code,fit in fits.items(): rows.append({"probe_phase":phase,"probe_exposure_index":index,"model_code":code,
            "model":fit.model,"q_A":fit.parameters["q_A"],"log_likelihood":fit.log_likelihood,"aicc":fit.aicc,
            "bic":fit.bic,"rmse":fit.rmse,"profile_correlation":fit.correlation,"profile_r2":fit.profile_r2})
    return pd.DataFrame(rows)


def _report(output,tables,winner,plausible,selection,hash_ok):
    comp=tables["first_mixed_model_comparison"].set_index("model_code")
    delta=float(comp.loc["R1","predictive_log_likelihood"]-comp.loc["R0","predictive_log_likelihood"])
    q=float(tables["first_mixed_parameter_estimates"].query("model_code=='R1' and parameter=='q_A'").estimate.iloc[0])
    ci=tables["first_mixed_bootstrap_summary"].set_index("metric"); low=ci.loc["q_R1","ci_low"]; high=ci.loc["q_R1","ci_high"]
    geometry_n=tables["novel_geometry"]
    novel_off=float(geometry_n.off_axis_distance.mean()) if len(geometry_n) else np.nan
    warned=tables["familiar_anchor_diagnostics"].query("range_warning").anchor_model.drop_duplicates().tolist()
    warning_text=(f"Decoded anchor range warnings occurred for **{', '.join(warned)}**; values were retained and flagged rather than clipped."
        if warned else "All decoded anchors passed the registered finite/range audit.")
    long=tables["full_experiment_fit_long"]
    familiar=long[~long.is_probe]; familiar_residual=familiar.observed_RT-familiar.familiar_predicted_RT
    familiar_rmse=float(np.sqrt(np.mean(familiar_residual**2)))
    familiar_corr=float(np.corrcoef(familiar.observed_RT,familiar.familiar_predicted_RT)[0,1])
    familiar_denom=float(np.sum((familiar.observed_RT-familiar.observed_RT.mean())**2))
    familiar_r2=float(1-np.sum(familiar_residual**2)/familiar_denom)
    probe_long=long[long.is_probe]
    probe_r0_rmse=float(np.sqrt(np.mean(probe_long.R0_residual_expected**2)))
    probe_r1_rmse=float(np.sqrt(np.mean(probe_long.R1_residual**2)))
    dist=tables["first_mixed_distribution_diagnostics"].set_index("source")
    ppc=tables["posterior_predictive_summary"]
    r0_pass=bool(ppc.query("model=='R0'").observed_within_95_predictive_interval.all())
    r1_pass=bool(ppc.query("model=='R1'").observed_within_95_predictive_interval.all())
    dec=tables["R0_R1_likelihood_decomposition"]; fraction_r0=float((dec.delta_LL_R1_minus_R0<0).mean())
    fraction_r1=float((dec.delta_LL_R1_minus_R0>0).mean())
    anchor_summary=tables["anchor_uncertainty_R0_R1_summary"].iloc[0]
    participant_summary=tables["participant_bootstrap_R0_R1_summary"].iloc[0]
    winner_anchor_warning=bool(tables["familiar_anchor_diagnostics"].query("anchor_model==@winner").range_warning.any())
    relative=("The endpoint-mixture likelihood exceeded the scalar-interpolation likelihood."
              if delta<0 else "The scalar-interpolation likelihood exceeded the endpoint-mixture likelihood.")
    if delta<0 and r0_pass:
        adequacy_conclusion="The endpoint-mixture model won relatively and passed the registered descriptive posterior-predictive coverage checks; the data are consistent with categorical endpoint retrieval."
    elif delta<0:
        adequacy_conclusion="Although the endpoint-mixture model achieved higher likelihood, it failed at least one registered absolute-fit check, so likelihood alone does not support a categorical mechanism."
    elif r1_pass:
        adequacy_conclusion="The scalar-interpolation model won relatively and passed the registered descriptive posterior-predictive coverage checks; the data are consistent with continuous interpolation."
    else:
        adequacy_conclusion="The scalar-interpolation model won relatively but failed at least one registered absolute-fit check; neither mechanism should be accepted from relative likelihood alone."
    text=f"""# Corrected Mixed / Novel retrieval report

## Relative model comparison

The frozen familiar winner is **{winner}**. Plausible familiar sensitivity models were **{', '.join(plausible)}**. On the first analyzed Mixed exposure, participant-fold ΔLL(R1−R0) was **{delta:.2f}**. {relative} The scalar continuous estimate was q(A) = **{q:.3f}**, participant-bootstrap 95% CI **[{low:.3f}, {high:.3f}]**.

R0 and R1 used identical observations, anchors, residual family, and two free parameters. R0 was evaluated as a Gaussian mixture over endpoint profiles; its plotted point prediction is only the mixture expectation.

## Absolute model adequacy

- **Full familiar trajectory:** {len(familiar):,} frozen familiar transition observations were checked. RMSE = **{familiar_rmse:.2f} ms**, R² = **{familiar_r2:.3f}**, correlation = **{familiar_corr:.3f}**.
- **Full probe timeline:** every analyzed Mixed, Novel, and NoContextProbe transition is present once in `full_experiment_fit_long.csv`, with both anchors and R0/R1 predictions. Across the complete probe timeline, descriptive expected-mean RMSE was **{probe_r0_rmse:.2f} ms** for R0 and **{probe_r1_rmse:.2f} ms** for R1.
- **Probe-time anchors:** the selected **{winner}** anchors had range-warning status **{winner_anchor_warning}** over the complete timeline. Alternative-model warnings remain in `familiar_anchor_diagnostics.csv`.
- **Raw first-Mixed distribution:** observed mean = **{dist.loc['Observed','mean']:.2f} ms**, SD = **{dist.loc['Observed','SD']:.2f}**, skewness = **{dist.loc['Observed','skewness']:.2f}**, estimated KDE modes of participant means = **{dist.loc['Observed','estimated_KDE_modes']:.0f}**.
- **Participant projections:** observed median q_proj = **{dist.loc['Observed','q_proj_median']:.3f}**, SD = **{dist.loc['Observed','q_proj_SD']:.3f}**; projections were not clamped.
- **R0 posterior prediction:** **{int(ppc.query("model=='R0'").observed_within_95_predictive_interval.sum())}/{len(ppc.query("model=='R0'"))}** registered aggregate diagnostics fell inside its 95% predictive intervals.
- **R1 posterior prediction:** **{int(ppc.query("model=='R1'").observed_within_95_predictive_interval.sum())}/{len(ppc.query("model=='R1'"))}** registered aggregate diagnostics fell inside its 95% predictive intervals.
- **Likelihood decomposition:** **{fraction_r0:.1%}** of participants favored R0 and **{fraction_r1:.1%}** favored R1; median participant ΔLL(R1−R0) = **{dec.delta_LL_R1_minus_R0.median():.3f}**.
- **Participant bootstrap:** median ΔLL(R1−R0) = **{participant_summary.median_delta_LL:.2f}**, 95% interval **[{participant_summary.ci_low_delta_LL:.2f}, {participant_summary.ci_high_delta_LL:.2f}]**. q intervals are reported in `first_mixed_bootstrap_summary.csv`.
- **Anchor uncertainty:** P(ΔLL<0) = **{anchor_summary.P_delta_LL_lt_0:.3f}**, P(ΔLL>0) = **{anchor_summary.P_delta_LL_gt_0:.3f}**, median predictive ΔLL = **{anchor_summary.median_delta_LL:.2f}**, 95% interval **[{anchor_summary.ci_low_delta_LL:.2f}, {anchor_summary.ci_high_delta_LL:.2f}]**.

These posterior-predictive checks are descriptive coverage diagnostics, not binary hypothesis tests. **{adequacy_conclusion}**

## Structured retrieval

R2 uses seven transition responsibilities with fixed hierarchical shrinkage toward their common mean. R3 uses a rank-{selection['r3_rank']} basis estimated exclusively from frozen familiar anchors. Their results are secondary to R0 versus R1.

## Novel and NoContextProbe

Novel is described geometrically rather than required to be a midpoint. Its mean off-axis distance was **{novel_off:.2f} ms**. `NoContextTraining` rows were explicitly excluded from `NoContextProbe`; only `No Context Test` was analyzed as removal of context after A/B learning.

## Firewall and uncertainty

- Probe rows used to estimate familiar parameters or update familiar states: **0**.
- Familiar input hashes unchanged after retrieval: **{hash_ok}**.
- Generated probe-anchor hash unchanged throughout retrieval fitting: **True**.
- Mixed was not used to select the familiar winner or the R3 basis.
- Anchors were fixed in the primary fit. Participant-bootstrap and familiar-anchor uncertainty results are exported separately.
- If the Cross/Split sensitivity changes the R0/R1 conclusion, retrieval inference remains contingent on familiar-model uncertainty.
- {warning_text}

## Interpretation guardrail

The analysis concerns retrieval/expression from frozen familiar states. It does not reinterpret probe experience as familiar learning and does not impose q=.5 for Mixed, Novel, or NoContextProbe.
"""
    (Path(output)/"MIXED_NOVEL_RETRIEVAL_CORRECTED_REPORT.md").write_text(text,encoding="utf-8")


def run_pipeline(combined_path,novel_path,familiar_dir,output_dir,*,config: RetrievalConfig):
    config.validate(); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    _progress(config,"validating immutable familiar inputs")
    selection,before_hashes=validate_familiar_inputs(familiar_dir)
    winner,plausible,familiar_comparison=select_familiar_models(familiar_dir,config.familiar_plausibility_delta_bic)
    if winner not in FAMILIAR_MODELS: raise AssertionError("Familiar winner must be Lump, Cross, or Split.")
    _progress(config,f"familiar winner {winner}; sensitivity {plausible}")
    contract,tokens=load_all(combined_path,novel_path)
    timeline=probe_timeline(contract); timeline.to_csv(output/"probe_timeline.csv",index=False)
    if timeline.query("probe_type=='NoContextTraining'").empty: _progress(config,"NoContextTraining label absent")
    models=list(FAMILIAR_MODELS)
    anchors,familiar_profiles=generate_anchors(tokens,familiar_dir,models,return_familiar_profiles=True)
    anchor_path=output/"probe_familiar_anchors.csv"
    anchors.to_csv(anchor_path,index=False)
    anchor_hash_before=sha256(anchor_path)
    diagnostics=anchor_diagnostics(anchors)
    if not np.isfinite(anchors[["A_anchor","B_anchor"]].to_numpy(float)).all():
        raise ValueError("Frozen familiar anchors contain non-finite values.")
    observed=attach_observations(anchors,tokens)
    primary=observed[(observed.anchor_model==winner)&(observed.probe_type=="Mixed")&(observed.probe_exposure_index==1)].copy()
    if primary.empty or not primary.probe_exposure_index.eq(1).all(): raise AssertionError("Primary rows must be first Mixed exposure.")
    if primary.groupby("participant_uid").transition_position.nunique().ne(7).any(): raise AssertionError("Primary profiles require seven transitions.")
    anchor_only=anchors[anchors.anchor_model.eq(winner)].copy()
    familiar_basis_input=familiar_profiles[familiar_profiles.anchor_model.eq(winner)].copy()
    basis_center,basis=familiar_basis(familiar_basis_input,config.r3_rank)
    basis_table=pd.DataFrame({"transition_position":np.arange(1,8),"familiar_center":basis_center})
    for index,row in enumerate(basis,1): basis_table[f"basis_{index}"]=row
    basis_table["basis_source"]="frozen familiar anchors only"
    _progress(config,f"fitting first Mixed R0-R3 to {primary.participant_uid.nunique()} participants")
    fits=fit_all(primary,basis,config.r2_shrinkage)
    predictive=participant_cv(primary,basis,config.participant_cv_folds,config.random_seed,config.r2_shrinkage)
    comparison=model_comparison(fits,predictive)
    parameters=parameter_table(fits,probe_type="Mixed",probe_exposure_index=1)
    predictions=prediction_frame(primary,fits)
    first_geometry=geometry(primary)

    _progress(config,f"bootstrapping first Mixed ({config.bootstrap_replicates} draws)")
    bootstrap=participant_bootstrap(primary,config.bootstrap_replicates,config.random_seed,
        config.effective_workers,progress=lambda x:_progress(config,x))
    bootstrap_summary=percentile_summary(bootstrap,["q_R0","q_R1","delta_log_likelihood_R1_minus_R0",
        "between_endpoint_fraction","profile_r2_R1","off_axis_distance"])

    mixed=observed[(observed.anchor_model==winner)&observed.probe_type.eq("Mixed")]
    mixed_params,mixed_profiles=_fit_by_exposure(mixed,basis,config)
    novel=observed[(observed.anchor_model==winner)&observed.probe_type.eq("Novel")]
    no_context=observed[(observed.anchor_model==winner)&observed.probe_type.eq("NoContextProbe")]
    novel_metrics=_probe_metrics(novel,basis,config); novel_geometry=geometry(novel) if len(novel) else pd.DataFrame()
    no_context_metrics=_probe_metrics(no_context,basis,config); no_context_geometry=geometry(no_context) if len(no_context) else pd.DataFrame()

    sensitivity=[]
    for model in FAMILIAR_MODELS:
        group=observed[(observed.anchor_model==model)&(observed.probe_type=="Mixed")&(observed.probe_exposure_index==1)]
        if group.empty: continue
        for code,fit in {"R0":fit_r0(group),"R1":fit_r1(group)}.items():
            sensitivity.append({"anchor_model":model,"selected_from_probe":False,"model_code":code,
                "q_A":fit.parameters["q_A"],"log_likelihood":fit.log_likelihood,"aicc":fit.aicc,
                "bic":fit.bic,"rmse":fit.rmse,"profile_r2":fit.profile_r2})
    sensitivity=pd.DataFrame(sensitivity)
    _progress(config,f"propagating anchor uncertainty through matched R0/R1 CV ({config.bootstrap_replicates} draws)")
    uncertainty=anchor_uncertainty(primary,anchor_only,config.bootstrap_replicates,config.random_seed,
        config.participant_cv_folds,progress=lambda x:_progress(config,x))

    _progress(config,f"simulating {config.posterior_predictive_profiles} posterior-predictive profiles per primary model")
    r0_simulation=simulate_R0_profiles(primary,fits["R0"],config.posterior_predictive_profiles,config.random_seed+101)
    r1_simulation=simulate_R1_profiles(primary,fits["R1"],config.posterior_predictive_profiles,config.random_seed+202)
    posterior_geo=posterior_geometry(primary,r0_simulation,r1_simulation)
    decomposition=likelihood_decomposition(primary,fits["R0"],fits["R1"])
    transition_diagnostics=transition_likelihood_diagnostics(primary,fits["R0"],fits["R1"])
    distribution=distribution_diagnostics(primary,r0_simulation,r1_simulation,posterior_geo)
    ppc_summary=posterior_predictive_summary(posterior_geo,config.posterior_summary_replicates,config.random_seed+303)
    full_long=build_full_experiment_fit_long(tokens,observed,familiar_dir,winner)
    expected_familiar=len(pd.read_csv(Path(familiar_dir)/"human_state_space_latent_states.csv").query("model==@winner"))
    if int((~full_long.is_probe).sum())!=expected_familiar:
        raise AssertionError("Complete familiar trajectory row count does not match the frozen familiar analysis.")
    if int(full_long.is_probe.sum())!=len(observed[observed.anchor_model.eq(winner)]):
        raise AssertionError("Complete probe timeline does not contain every analyzed probe transition exactly once.")
    participant_r0_r1=bootstrap.rename(columns={"bootstrap":"bootstrap_id","log_likelihood_R0":"LL_R0",
        "log_likelihood_R1":"LL_R1","delta_log_likelihood_R1_minus_R0":"delta_LL"})[
        ["bootstrap_id","q_R0","q_R1","LL_R0","LL_R1","delta_LL"]].copy()
    anchor_r0_r1=uncertainty[["bootstrap_id","q_R0","q_R1","LL_R0","LL_R1","delta_LL"]].copy()
    participant_uncertainty_summary=uncertainty_summary(participant_r0_r1)
    anchor_uncertainty_summary=uncertainty_summary(anchor_r0_r1)

    tables={"first_mixed_model_comparison":comparison,"first_mixed_parameter_estimates":parameters,
        "first_mixed_profile_predictions":predictions,"first_mixed_geometry":first_geometry,
        "first_mixed_bootstrap":bootstrap,"first_mixed_bootstrap_summary":bootstrap_summary,
        "mixed_all_exposures_parameters":mixed_params,"mixed_all_exposures_profiles":mixed_profiles,
        "novel_retrieval_metrics":novel_metrics,"novel_geometry":novel_geometry,
        "no_context_probe_metrics":no_context_metrics,"no_context_probe_geometry":no_context_geometry,
        "familiar_model_sensitivity":sensitivity,"anchor_uncertainty_sensitivity":uncertainty,
        "corrected_familiar_model_comparison":familiar_comparison,
        "familiar_anchor_diagnostics":diagnostics,"familiar_low_rank_basis":basis_table,
        "full_experiment_fit_long":full_long,"first_mixed_distribution_diagnostics":distribution,
        "R0_R1_likelihood_decomposition":decomposition,"transition_likelihood_diagnostics":transition_diagnostics,
        "anchor_uncertainty_R0_R1":anchor_r0_r1,"participant_bootstrap_R0_R1":participant_r0_r1,
        "anchor_uncertainty_R0_R1_summary":anchor_uncertainty_summary,
        "participant_bootstrap_R0_R1_summary":participant_uncertainty_summary,
        "posterior_predictive_summary":ppc_summary,"posterior_predictive_geometry":posterior_geo,
        "posterior_predictive_profiles":pd.concat([r0_simulation,r1_simulation],ignore_index=True)}
    for name,table in tables.items(): table.to_csv(output/f"{name}.csv",index=False)
    # Mandatory plots are part of the inference gate and are rendered even
    # when --no-plots suppresses the older, optional figure set.
    _progress(config,"rendering mandatory absolute-fit figures A-J")
    from .diagnostic_plotting import make_absolute_fit_figures
    make_absolute_fit_figures(output,full_long,primary,r0_simulation,r1_simulation,posterior_geo,
        decomposition,transition_diagnostics,anchor_r0_r1)
    diagnostics_complete=assert_absolute_fit_bundle(output)
    after_hashes={name:sha256(Path(familiar_dir)/name) for name in before_hashes}
    hash_ok=before_hashes==after_hashes
    if not hash_ok: raise RuntimeError("A frozen familiar input changed during retrieval analysis.")
    anchor_hash_after=sha256(anchor_path)
    if anchor_hash_before!=anchor_hash_after: raise RuntimeError("The frozen probe anchor file changed during retrieval fitting.")
    record={"analysis_version":config.analysis_version,"profile":config.profile,"familiar_winner":winner,
        "plausible_familiar_models":plausible,"r3_rank":config.r3_rank,"r3_basis_source":"decoded familiar A/B rows only",
        "combined_path":str(Path(combined_path).resolve()),"novel_path":str(Path(novel_path).resolve()),
        "familiar_dir":str(Path(familiar_dir).resolve()),"familiar_hashes_before":before_hashes,
        "familiar_hashes_after":after_hashes,"familiar_hashes_unchanged":hash_ok,
        "probe_anchor_hash_before_fitting":anchor_hash_before,"probe_anchor_hash_after_fitting":anchor_hash_after,
        "probe_anchor_hash_unchanged":anchor_hash_before==anchor_hash_after,
        "FIT_DIAGNOSTICS_COMPLETE":diagnostics_complete,
        "probe_rows_used_for_familiar_fit":0,"probe_driven_familiar_updates":0,
        "no_context_training_rows_treated_as_probe":0,"automatic_midpoint_rules":0,
        "workers":config.effective_workers,"bootstrap_replicates":config.bootstrap_replicates,
        "posterior_predictive_profiles_per_model":config.posterior_predictive_profiles}
    (output/"retrieval_run_config.json").write_text(json.dumps(record,indent=2,sort_keys=True),encoding="utf-8")
    _report(output,tables,winner,plausible,{"r3_rank":config.r3_rank},hash_ok)
    if config.make_plots:
        _progress(config,"rendering ten registered figures")
        from .plotting import make_figures
        make_figures(output,tables)
    _progress(config,f"complete: {output}")
    return tables
