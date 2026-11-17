from __future__ import annotations
import numpy as np
import pandas as pd
from .config import HIGH,ExperimentConfig


def seed_summary(metrics,config):
    rows=[]
    for (family,seed,probe,intervention,clamp),group in metrics.groupby(["family","seed","probe","intervention","context_clamp"],sort=False):
        repeated=metrics[(metrics.family.eq(family))&(metrics.seed.eq(seed))&(metrics.probe.eq(probe))&metrics.intervention.eq("repeated_midpoint")]
        baseline=metrics[(metrics.family.eq(family))&(metrics.seed.eq(seed))&(metrics.probe.eq(probe))&metrics.intervention.eq("natural_baseline")]
        if intervention=="natural_baseline":
            immediate=remaining=mean_remaining=full=behavior=0.;post_h=group.hidden_midpoint_residual.mean();post_z=group.task_midpoint_residual.mean();upper=float(repeated.raw_rescue.sum());retained=0.;unstable=abs(upper)<config.rescue_epsilon
        elif intervention=="repeated_midpoint":
            immediate=float(group.raw_rescue.mean());remaining=full=float(group.raw_rescue.sum());mean_remaining=float(group.raw_rescue.mean());behavior=float(group.raw_rescue.mean())
            post_h=float(group.hidden_midpoint_residual.mean());post_z=float(group.task_midpoint_residual.mean());upper=remaining;retained=1.;unstable=abs(upper)<config.rescue_epsilon
        else:
            immediate=float(group[group.lag_from_patch.eq(0)].raw_rescue.mean())
            per_position=[];per_position_mean=[];upper_position=[]
            for position,part in group.groupby("injection_position"):
                remaining_part=part[part.lag_from_patch.ge(0)].raw_rescue
                per_position.append(remaining_part.sum());per_position_mean.append(remaining_part.mean())
                upper_position.append(repeated[repeated.transition.ge(position)].raw_rescue.sum())
            remaining=float(np.mean(per_position));mean_remaining=float(np.mean(per_position_mean));full=float(group.groupby("injection_position").raw_rescue.sum().mean())
            post=group[group.lag_from_patch.gt(0)];behavior=float(post.raw_rescue.mean()) if len(post) else np.nan
            post_h=float(post.hidden_midpoint_residual.mean()) if len(post) else np.nan;post_z=float(post.task_midpoint_residual.mean()) if len(post) else np.nan
            upper=float(np.mean(upper_position));unstable=abs(upper)<config.rescue_epsilon;retained=remaining/(upper+config.epsilon) if not unstable else np.nan
        rows.append({"family":family,"seed":seed,"probe":probe,"intervention":intervention,"context_clamp":clamp,
                     "immediate_rescue":immediate,"remaining_sequence_rescue":remaining,"full_sequence_rescue":full,
                     "mean_remaining_ce_difference":mean_remaining,
                     "mean_post_patch_hidden_residual":post_h,"mean_post_patch_task_residual":post_z,
                     "behavioral_persistence_score":behavior,"hidden_persistence_score":-post_h,"task_persistence_score":-post_z,
                     "repeated_patch_upper_bound_rescue":upper,"fraction_upper_bound_retained":retained,
                     "upper_bound_denominator_unstable":unstable})
    return pd.DataFrame(rows)


def _focus(summary,probe,clamp):
    return summary[(summary.probe.eq(probe))&summary.intervention.eq("single_midpoint")&summary.context_clamp.eq(clamp)].set_index(["family","seed"])


def primary_comparisons(summary,config):
    d0={p:_focus(summary,p,"D0_natural") for p in ("Mixed","NoContext","NovelMean")}
    d1={p:_focus(summary,p,"D1_zero") for p in ("Mixed","NoContext","NovelMean")}
    d2={p:_focus(summary,p,"D2_composite") for p in ("Mixed","NoContext","NovelMean")}
    definitions=[
        ("Mixed minus Novel post-injection hidden drift",d0["Mixed"].mean_post_patch_hidden_residual-d0["NovelMean"].mean_post_patch_hidden_residual,"lower"),
        ("Mixed minus NoContext post-injection hidden drift",d0["Mixed"].mean_post_patch_hidden_residual-d0["NoContext"].mean_post_patch_hidden_residual,"lower"),
        ("Mixed minus Novel remaining rescue",d0["Mixed"].remaining_sequence_rescue-d0["NovelMean"].remaining_sequence_rescue,"higher"),
        ("Mixed minus NoContext remaining rescue",d0["Mixed"].remaining_sequence_rescue-d0["NoContext"].remaining_sequence_rescue,"higher"),
        ("Novel D1 minus D0 persistence",d1["NovelMean"].behavioral_persistence_score-d0["NovelMean"].behavioral_persistence_score,"higher"),
        ("Novel D2 minus D0 persistence",d2["NovelMean"].behavioral_persistence_score-d0["NovelMean"].behavioral_persistence_score,"higher"),
        ("NoContext D2 minus D0 persistence",d2["NoContext"].behavioral_persistence_score-d0["NoContext"].behavioral_persistence_score,"higher"),
        ("Mixed D2 minus D0 equality",d2["Mixed"].behavioral_persistence_score-d0["Mixed"].behavioral_persistence_score,"zero"),
        ("NoContext D1 minus D0 equality",d1["NoContext"].behavioral_persistence_score-d0["NoContext"].behavioral_persistence_score,"zero"),
    ]
    rows=[]
    for name,series,direction in definitions:
        for (family,seed),value in series.items():
            predicted=(value<0 if direction=="lower" else value>0 if direction=="higher" else abs(value)<1e-10)
            rows.append({"family":family,"seed":seed,"comparison":name,"value":value,"predicted_direction":direction,"predicted_direction_observed":predicted})
    return pd.DataFrame(rows)


def bootstrap_summary(comparisons,config):
    rng=np.random.default_rng(config.bootstrap_seed);rows=[]
    for (family,name),group in comparisons.groupby(["family","comparison"]):
        values=group.value.to_numpy();samples=np.stack([rng.choice(values,len(values),replace=True) for _ in range(config.bootstrap_samples)])
        boot_mean=samples.mean(axis=1);boot_median=np.median(samples,axis=1)
        rows.append({"family":family,"comparison":name,"n_seeds":len(values),"mean":values.mean(),"median":np.median(values),
                     "mean_ci_low":np.quantile(boot_mean,.025),"mean_ci_high":np.quantile(boot_mean,.975),
                     "median_ci_low":np.quantile(boot_median,.025),"median_ci_high":np.quantile(boot_median,.975),
                     "fraction_predicted_direction":group.predicted_direction_observed.mean(),"resampling_unit":"seed"})
    return pd.DataFrame(rows)


def classify(summary):
    high=summary[summary.family.eq(HIGH)]
    def row(probe,intervention="single_midpoint",clamp="D0_natural"):
        return high[high.probe.eq(probe)&high.intervention.eq(intervention)&high.context_clamp.eq(clamp)].mean(numeric_only=True)
    mixed=row("Mixed");novel=row("NovelMean");no=row("NoContext")
    novel_d1=row("NovelMean",clamp="D1_zero");novel_d2=row("NovelMean",clamp="D2_composite")
    no_d2=row("NoContext",clamp="D2_composite");mixed_d2=row("Mixed",clamp="D2_composite");no_d1=row("NoContext",clamp="D1_zero")
    wrong=row("Mixed","wrong_position");orth=row("Mixed","orthogonal")
    controls_specific=mixed.remaining_sequence_rescue>max(wrong.remaining_sequence_rescue,orth.remaining_sequence_rescue)
    immediate=mixed.immediate_rescue>0 and novel.immediate_rescue>0 and no.immediate_rescue>0
    mixed_better=(mixed.mean_post_patch_hidden_residual<novel.mean_post_patch_hidden_residual and
                  mixed.mean_post_patch_hidden_residual<no.mean_post_patch_hidden_residual and
                  mixed.remaining_sequence_rescue>novel.remaining_sequence_rescue and mixed.remaining_sequence_rescue>no.remaining_sequence_rescue)
    equalities=(abs(mixed_d2.behavioral_persistence_score-mixed.behavioral_persistence_score)<1e-10 and
                abs(no_d1.behavioral_persistence_score-no.behavioral_persistence_score)<1e-10)
    cue_localized=(novel_d1.behavioral_persistence_score>novel.behavioral_persistence_score or
                   novel_d2.behavioral_persistence_score>novel.behavioral_persistence_score) and no_d2.behavioral_persistence_score>no.behavioral_persistence_score
    if not controls_specific:return "E","Directional controls rescue as well as the midpoint"
    if max(mixed.behavioral_persistence_score,novel.behavioral_persistence_score,no.behavioral_persistence_score)<=0:
        return "D","No probe retains the injection"
    scale=max(abs(mixed.behavioral_persistence_score),abs(novel.behavioral_persistence_score),abs(no.behavioral_persistence_score),1e-12)
    similar=(abs(mixed.behavioral_persistence_score-novel.behavioral_persistence_score)<.25*scale and
             abs(mixed.behavioral_persistence_score-no.behavioral_persistence_score)<.25*scale)
    if similar:return "C","All probes retain the injection similarly"
    if immediate and mixed_better and equalities and cue_localized:return "A","Full contextual stabilization account"
    return "B","Mixed-specific persistence without clean cue decomposition"
