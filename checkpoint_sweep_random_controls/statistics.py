"""Frontier assembly and paired preregistered contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def frontier_metrics(familiar, transfer, hidden, controller, exploratory_mixed):
    summary=familiar.query("row_type == 'seed_summary'").copy()
    rt=transfer[transfer.metric=="RT"][["group","seed","checkpoint","switch_cost","cross_context_transfer_index"]]
    keys=["group","architecture","prior_condition","seed","checkpoint"]
    result=summary.merge(rt,on=["group","seed","checkpoint"],how="left").merge(hidden,on=keys,how="left",suffixes=("","_hidden"))
    if len(controller):result=result.merge(controller,on=keys,how="left",suffixes=("","_controller"))
    if len(exploratory_mixed) and "row_type" in exploratory_mixed:
        mixed=exploratory_mixed.query("row_type == 'seed_summary'")
        columns=["group","architecture","prior_condition","seed","checkpoint","between_endpoint_fraction","mean_RT_deficit","NLL_minus_mean_familiar","human_profile_correlation_unfitted"]
        result=result.merge(mixed[[c for c in columns if c in mixed]],on=keys,how="left")
    result["analysis_type"]="exploratory_checkpoint_frontier"
    threshold=1e-3
    is_b=result.architecture.eq("B"); controller_split=result.get("controller_separation_exceeds_null",pd.Series(False,index=result.index)).eq(True)
    hidden_split=result.hidden_separation_exceeds_null.eq(True); behavior_split=result.AB_RT_mean_absolute_difference.ge(threshold)
    result["B_mechanism"]="not_applicable"
    result.loc[is_b&~controller_split,"B_mechanism"]="controller_collapse"
    result.loc[is_b&controller_split&~hidden_split,"B_mechanism"]="controller_split_task_core_collapse"
    result.loc[is_b&controller_split&hidden_split&~behavior_split,"B_mechanism"]="representation_split_output_shared"
    result.loc[is_b&controller_split&hidden_split&behavior_split,"B_mechanism"]="controller_task_behavior_split"
    return result


def _paired(frame,left,right,metric,keys):
    a=frame[frame.group==left].set_index(keys)[metric];b=frame[frame.group==right].set_index(keys)[metric]
    return pd.concat([a.rename("left"),b.rename("right")],axis=1,join="inner").dropna().assign(difference=lambda x:x.left-x.right)


def planned_contrasts(config,frontier,selected_mixed):
    rows=[];boots=[];rng=np.random.default_rng(int(config.payload["random_seed"]));draws=int(config.payload["bootstrap_draws"])
    comparisons=(("S","Sh"),("S","Rand"),("Sh","Rand"))
    familiar_metrics=("AB_RT_mean_absolute_difference","coherent_AB_hidden_distance","cross_context_transfer_index","switch_cost")
    for architecture in ("A","B"):
        for checkpoint in config.checkpoints:
            one=frontier[(frontier.architecture==architecture)&(frontier.checkpoint==checkpoint)]
            for l,r in comparisons:
                left=f"{architecture}-{l}";right=f"{architecture}-{r}"
                for metric in familiar_metrics:
                    paired=_paired(one,left,right,metric,["seed"])
                    if not len(paired):continue
                    delta=paired.difference.to_numpy(float);samples=rng.choice(delta,size=(draws,len(delta)),replace=True).mean(1)
                    sd=delta.std(ddof=1) if len(delta)>1 else np.nan
                    common={"analysis_stage":"familiar_trajectory","architecture":architecture,"checkpoint":checkpoint,"contrast":f"{left} - {right}","metric":metric,"n_paired_seeds":len(delta),
                        "paired_mean_difference":delta.mean(),"standardized_effect":delta.mean()/sd if np.isfinite(sd) and sd>0 else np.nan,"positive_direction_count":int((delta>0).sum())}
                    rows.append(common);boots.append(common|{"bootstrap_draws":draws,"CI95_low":np.quantile(samples,.025),"CI95_high":np.quantile(samples,.975)})
    if len(selected_mixed) and "row_type" in selected_mixed:
        summary=selected_mixed.query("row_type == 'seed_summary'")
        for architecture in ("A","B"):
            one=summary[summary.architecture==architecture]
            for l,r in comparisons:
                left=f"{architecture}-{l}";right=f"{architecture}-{r}"
                for metric in ("between_endpoint_fraction","mean_RT_deficit","NLL_minus_mean_familiar","human_profile_correlation_unfitted"):
                    if metric not in one:continue
                    paired=_paired(one,left,right,metric,["seed"])
                    if not len(paired):continue
                    delta=paired.difference.to_numpy(float);samples=rng.choice(delta,size=(draws,len(delta)),replace=True).mean(1)
                    sd=delta.std(ddof=1) if len(delta)>1 else np.nan
                    common={"analysis_stage":"selected_checkpoint_Mixed","architecture":architecture,"checkpoint":np.nan,"contrast":f"{left} - {right}","metric":metric,"n_paired_seeds":len(delta),
                        "paired_mean_difference":delta.mean(),"standardized_effect":delta.mean()/sd if np.isfinite(sd) and sd>0 else np.nan,"positive_direction_count":int((delta>0).sum())}
                    rows.append(common);boots.append(common|{"bootstrap_draws":draws,"CI95_low":np.quantile(samples,.025),"CI95_high":np.quantile(samples,.975)})
    return pd.DataFrame(rows),pd.DataFrame(boots)


def mixed_effects_models(frontier, selected_mixed):
    """Preregistered condition × checkpoint and selected-Mixed random-intercept models."""
    from statsmodels.formula.api import mixedlm
    rows=[]
    specifications=[]
    for architecture in ("A","B"):
        data=frontier[(frontier.architecture==architecture)&(frontier.group!="H8")].copy()
        for metric in ("AB_RT_mean_absolute_difference","coherent_AB_hidden_distance","cross_context_transfer_index","switch_cost"):
            specifications.append(("familiar_trajectory",architecture,metric,data,
                f"{metric} ~ C(prior_condition) * checkpoint"))
        if len(selected_mixed) and "row_type" in selected_mixed:
            selected=selected_mixed.query("row_type == 'seed_summary'")
            selected=selected[selected.architecture==architecture].copy()
            for metric in ("between_endpoint_fraction","mean_RT_deficit","NLL_minus_mean_familiar"):
                if metric in selected:specifications.append(("selected_checkpoint_Mixed",architecture,metric,selected,f"{metric} ~ C(prior_condition)"))
    for stage,architecture,metric,data,formula in specifications:
        columns=["seed","prior_condition","checkpoint",metric] if "checkpoint" in formula else ["seed","prior_condition",metric]
        one=data[columns].dropna()
        if one.seed.nunique()<2 or one.prior_condition.nunique()<2:continue
        try:
            fit=mixedlm(formula,one,groups=one.seed).fit(reml=False,method="lbfgs",disp=False)
            for term in fit.params.index:rows.append({"analysis_stage":stage,"architecture":architecture,"metric":metric,"formula":formula+" + (1|seed)",
                "term":term,"estimate":fit.params[term],"standard_error":fit.bse[term],"z_value":fit.tvalues[term],"p_value":fit.pvalues[term],"converged":bool(fit.converged),"fit_status":"fit"})
        except (ValueError,np.linalg.LinAlgError) as error:
            rows.append({"analysis_stage":stage,"architecture":architecture,"metric":metric,"formula":formula+" + (1|seed)","term":"FIT_FAILED","converged":False,"fit_status":"failed","failure_reason":str(error)})
    return pd.DataFrame(rows)
