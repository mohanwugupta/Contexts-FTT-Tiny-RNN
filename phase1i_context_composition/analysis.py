from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HIGH,LOW,ExperimentConfig


def _value(frame,probe,intervention,column="raw_sequence_loss"):
    subset=frame[(frame.probe.eq(probe))&(frame.intervention.eq(intervention))]
    return float(subset[column].mean())


def build_seed_summary(transition,timeline,functional,gradients,provenance,config):
    rows=[];final_t=config.training_sequences
    for seed in config.seeds:
        for family in config.families:
            tr=transition[(transition.seed==seed)&transition.family.eq(family)&transition.training_sequence_index.eq(final_t)]
            matched=tr[tr.intervention.eq("matched_state_branch")];free=tr[tr.intervention.eq("free_rollout")]
            patches=tr[(tr.patch_position.fillna(-1)==0)&tr.intervention.isin([
                "no_patch","composite_patch","wrong_position_patch","orthogonal_patch","A_endpoint_patch","B_endpoint_patch","midpoint_patch"])]
            fn=functional[(functional.seed==seed)&functional.family.eq(family)]
            grad=gradients[(gradients.seed==seed)&gradients.family.eq(family)].iloc[0]
            prov=provenance[(provenance.seed==seed)&provenance.family.eq(family)].iloc[0]
            mixed_base=_value(patches,"Mixed","no_patch");mixed_comp=_value(patches,"Mixed","composite_patch")
            no_base=_value(patches,"NoContext","no_patch");no_comp=_value(patches,"NoContext","composite_patch")
            novel=patches[patches.probe.str.startswith("Novel")]
            novel_base=novel[novel.intervention.eq("no_patch")].raw_sequence_loss.mean()
            novel_comp=novel[novel.intervention.eq("composite_patch")].raw_sequence_loss.mean()
            rows.append({"seed":seed,"family":family,"checkpoint_hash":prov.checkpoint_hash,
                         "gradient_cosine_AB":grad.gradient_cosine_AB,"within_transfer":fn.within_transfer.mean(),
                         "cross_transfer":fn.cross_transfer.mean(),
                         "mixed_matched_hidden_residual":matched[matched.probe.eq("Mixed")].hidden_composition_residual_normalized.mean(),
                         "mixed_matched_task_residual":matched[matched.probe.eq("Mixed")].task_composition_residual_normalized.mean(),
                         "nocontext_matched_hidden_residual":matched[matched.probe.eq("NoContext")].hidden_composition_residual_normalized.mean(),
                         "novel_matched_hidden_residual":matched[matched.probe.str.startswith("Novel")].hidden_composition_residual_normalized.mean(),
                         "mixed_free_hidden_residual":free[free.probe.eq("Mixed")].hidden_composition_residual_normalized.mean(),
                         "mixed_free_task_residual":free[free.probe.eq("Mixed")].task_composition_residual_normalized.mean(),
                         "AB_hidden_separation":free[free.probe.eq("Mixed")].AB_hidden_separation.mean(),
                         "AB_task_separation":free[free.probe.eq("Mixed")].AB_task_separation.mean(),
                         "nocontext_free_hidden_residual":free[free.probe.eq("NoContext")].hidden_composition_residual_normalized.mean(),
                         "novel_free_hidden_residual":free[free.probe.str.startswith("Novel")].hidden_composition_residual_normalized.mean(),
                         "mixed_baseline_raw_loss":mixed_base,"mixed_composite_patch_raw_loss":mixed_comp,
                         "mixed_wrong_position_raw_loss":_value(patches,"Mixed","wrong_position_patch"),
                         "mixed_orthogonal_raw_loss":_value(patches,"Mixed","orthogonal_patch"),
                         "mixed_A_endpoint_raw_loss":_value(patches,"Mixed","A_endpoint_patch"),
                         "mixed_B_endpoint_raw_loss":_value(patches,"Mixed","B_endpoint_patch"),
                         "mixed_midpoint_raw_loss":_value(patches,"Mixed","midpoint_patch"),
                         "mixed_rescue":mixed_base-mixed_comp,
                         "nocontext_baseline_raw_loss":no_base,"nocontext_composite_patch_raw_loss":no_comp,
                         "nocontext_wrong_position_raw_loss":_value(patches,"NoContext","wrong_position_patch"),
                         "nocontext_orthogonal_raw_loss":_value(patches,"NoContext","orthogonal_patch"),
                         "nocontext_rescue":no_base-no_comp,
                         "novel_mean_baseline_raw_loss":novel_base,"novel_mean_composite_patch_raw_loss":novel_comp,
                         "novel_mean_wrong_position_raw_loss":novel[novel.intervention.eq("wrong_position_patch")].raw_sequence_loss.mean(),
                         "novel_mean_orthogonal_raw_loss":novel[novel.intervention.eq("orthogonal_patch")].raw_sequence_loss.mean(),
                         "novel_mean_rescue":novel_base-novel_comp})
    return pd.DataFrame(rows)


def contrast_values(summary):
    rows=[]
    definitions={
        "Mixed matched residual minus Novel":("mixed_matched_hidden_residual","novel_matched_hidden_residual","lower"),
        "Mixed matched residual minus NoContext":("mixed_matched_hidden_residual","nocontext_matched_hidden_residual","lower"),
        "Mixed free residual minus Novel":("mixed_free_hidden_residual","novel_free_hidden_residual","lower"),
        "Mixed correct patch minus baseline":("mixed_composite_patch_raw_loss","mixed_baseline_raw_loss","lower"),
        "Mixed correct patch minus wrong":("mixed_composite_patch_raw_loss","mixed_wrong_position_raw_loss","lower"),
        "Mixed correct patch minus orthogonal":("mixed_composite_patch_raw_loss","mixed_orthogonal_raw_loss","lower"),
        "Novel correct patch minus baseline":("novel_mean_composite_patch_raw_loss","novel_mean_baseline_raw_loss","lower"),
        "NoContext correct patch minus baseline":("nocontext_composite_patch_raw_loss","nocontext_baseline_raw_loss","lower"),
    }
    for row in summary.itertuples():
        for contrast,(left,right,direction) in definitions.items():
            value=getattr(row,left)-getattr(row,right)
            rows.append({"seed":row.seed,"family":row.family,"contrast":contrast,"value":value,
                         "predicted_direction":direction,"predicted_direction_observed":value<0})
    return pd.DataFrame(rows)


def bootstrap_contrasts(values,config):
    rng=np.random.default_rng(config.bootstrap_seed);rows=[]
    for (family,contrast),group in values.groupby(["family","contrast"]):
        x=group.value.to_numpy();boot=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(config.bootstrap_samples)])
        rows.append({"family":family,"contrast":contrast,"n_seeds":len(x),"mean":x.mean(),"median":np.median(x),
                     "ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),
                     "fraction_predicted_direction":group.predicted_direction_observed.mean(),"resampling_unit":"seed"})
    return pd.DataFrame(rows)


def decision(summary,cue_exact):
    means=summary.groupby("family").mean(numeric_only=True);high=means.loc[HIGH]
    matched_good=high.mixed_matched_hidden_residual<min(high.novel_matched_hidden_residual,high.nocontext_matched_hidden_residual)
    rollout_good=high.mixed_free_hidden_residual<min(high.novel_free_hidden_residual,high.nocontext_free_hidden_residual)
    selective=(high.mixed_rescue>0 and high.mixed_composite_patch_raw_loss<high.mixed_wrong_position_raw_loss and
               high.mixed_composite_patch_raw_loss<high.mixed_orthogonal_raw_loss)
    low_alignment=abs(high.gradient_cosine_AB)<.20
    if cue_exact and matched_good and rollout_good and selective and low_alignment:
        primary=("A","Full compositional-context support")
    elif cue_exact and matched_good and high.mixed_free_hidden_residual>2*high.mixed_matched_hidden_residual and selective:
        primary=("B","Local composition with recurrent failure")
    elif selective:
        primary=("C","A useful composite target exists but is not naturally generated")
    else:primary=("D","Mixed composition is not supported")
    synthetic=(high.novel_mean_rescue+high.nocontext_rescue)/2
    controls=(high.novel_mean_baseline_raw_loss-high.novel_mean_wrong_position_raw_loss+
              high.nocontext_baseline_raw_loss-high.nocontext_wrong_position_raw_loss)/2
    secondary=("E","Synthetic composite rescues Novel/NoContext") if synthetic>0 and synthetic>controls else (
        "F","Synthetic composite does not selectively rescue Novel/NoContext")
    return primary,secondary
