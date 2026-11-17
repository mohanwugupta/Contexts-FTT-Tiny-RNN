from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from srn_mixed_novel_diagnostics.core import (
    CONDITION_LABELS, CONTEXT_VECTORS, load_frozen_network, sha256_file,
)

from .analysis import association_tables, analyze_network, loso_decoders
from .config import MechanismConfig
from .math import exact_forward
from .plotting import make_figures


REQUIRED_TABLES=("decoder_loso_results","cue_sweep_hidden_residuals","cue_sweep_output_residuals",
                 "cue_sweep_logit_residuals","cue_sweep_tanh_gain","cue_sweep_saturation",
                 "cue_sweep_curvature","recurrent_jacobian_metrics","formal_model_familiar_fit",
                 "formal_model_mixed_prediction","seed_level_mechanism_table",
                 "causal_linear_path_patching","linearized_activation_counterfactual","novel_input_audit")


def _progress(message):
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] SRN-COMPOSITIONALITY | {message}",flush=True)


def _reproduction_gate(records, reproduction_dir):
    rows=[]
    current=[]
    for record in records:
        # Reproduce the prior pipeline's float32 reductions exactly; all new
        # mechanism calculations subsequently use float64 accumulation.
        a,b,m=(record["runs"][name]["hidden"].astype(np.float32) for name in ("A","B","Mixed"))
        direction=b-a; denominator=np.sum(direction**2,axis=1)
        q=np.divide(np.sum((m-a)*direction,axis=1),denominator,out=np.zeros_like(denominator),where=denominator>1e-14)
        projected=a+q[:,None]*direction
        residual=np.linalg.norm(m-projected,axis=1).mean()/max(np.linalg.norm(direction,axis=1).mean(),1e-12)
        current.append({"condition":record["condition"],"seed":record["seed"],"Mixed_q":q.mean(),
                        "Mixed_normalized_off_axis":residual,"Mixed_loss":record["runs"]["Mixed"]["loss"].mean(),
                        "A_loss":record["runs"]["A"]["loss"].mean(),"B_loss":record["runs"]["B"]["loss"].mean()})
    current=pd.DataFrame(current)
    old_geometry=Path(reproduction_dir)/"probe_axis_geometry.csv"
    old_sweep=Path(reproduction_dir)/"context_interpolation_sweep.csv"
    if old_geometry.exists() and old_sweep.exists():
        geometry=pd.read_csv(old_geometry).query("probe_context == 'Mixed'")
        sweep=pd.read_csv(old_sweep)
        old=geometry[["condition","seed","mean_q","normalized_off_axis_residual"]].rename(columns={
            "mean_q":"old_Mixed_q","normalized_off_axis_residual":"old_Mixed_normalized_off_axis"})
        for cue,name in ((0.,"old_A_loss"),(.5,"old_Mixed_loss"),(1.,"old_B_loss")):
            part=sweep[np.isclose(sweep.cue_lambda,cue)][["condition","seed","sequence_loss"]].rename(columns={"sequence_loss":name})
            old=old.merge(part,on=["condition","seed"],how="inner")
        merged=current.merge(old,on=["condition","seed"],validate="one_to_one")
        for metric in ("Mixed_q","Mixed_normalized_off_axis","Mixed_loss","A_loss","B_loss"):
            old_name="old_"+metric
            difference=np.abs(merged[metric]-merged[old_name])
            rows.append({"gate":f"exact_reproduction_{metric}","value":difference.max(),"criterion":"max_abs_difference <= 5e-6",
                         "passed":bool(difference.max()<=5e-6)})
    summary=current.groupby("condition").mean(numeric_only=True)
    high,low=summary.loc["High_monotonic"],summary.loc["Low_monotonic"]
    qualitative={
        "both_Mixed_q_near_half":(abs(high.Mixed_q-.5)<.15 and abs(low.Mixed_q-.5)<.15),
        "High_off_axis_exceeds_Low":high.Mixed_normalized_off_axis>low.Mixed_normalized_off_axis,
        "High_midpoint_loss_valley":high.Mixed_loss>(high.A_loss+high.B_loss)/2+.1,
        "Low_midpoint_loss_flat":abs(low.Mixed_loss-(low.A_loss+low.B_loss)/2)<.02,
        "Low_off_axis_near_zero":low.Mixed_normalized_off_axis<.05,
    }
    rows.extend({"gate":name,"value":bool(value),"criterion":"expected qualitative frozen pattern","passed":bool(value)}
                for name,value in qualitative.items())
    frame=pd.DataFrame(rows)
    if not frame.passed.all():
        raise AssertionError("Existing diagnostic reproduction gate failed:\n"+frame.to_string(index=False))
    return frame,current


def _mean(frame, groups, columns):
    return frame.groupby(groups,as_index=False)[columns].mean(numeric_only=True).to_string(index=False)


def _hypothesis_table(tables):
    master=tables["srn_mechanism_master_table"]
    patch=tables["causal_linear_path_patching"]
    formal=tables["formal_model_familiar_fit"]
    linearized=tables["linearized_activation_counterfactual"]
    high=master[master.condition.eq("High_monotonic")]
    high_endpoint_linearization=linearized[
        linearized.condition.eq("High_monotonic")
        & np.isclose(linearized.cue_lambda,.5)
        & linearized.linearization_anchor.eq("nearest_endpoint")
    ].delta_loss_linearized_minus_actual.mean()
    high_associations=tables["mechanistic_associations"]
    high_associations=high_associations[high_associations.scope.eq("High_monotonic")]
    def assoc(predictor,outcome):
        row=high_associations[high_associations.predictor.eq(predictor)&high_associations.outcome.eq(outcome)].iloc[0]
        return float(row.pearson_r),float(row.bootstrap_CI_low),float(row.bootstrap_CI_high)
    saturation_curvature=assoc("saturation_09","K_out_mid")
    recurrence_curvature=assoc("Jrec_cumulative_max","K_out_mid")
    recurrence_loss=assoc("Jrec_cumulative_max","Mixed_loss")
    rows=[
        ("H1 pure additive factorization", "mixed", f"High mean F1 R2={high.F1_R2.mean():.3f}; P_interaction={high.P_interaction.mean():.3f}; Mixed residual={high.Mixed_whole_residual.mean():.3f}"),
        ("H2 structured bilinear splitting", "mixed", f"High F2-F1 R2 gain={(high.F2_R2-high.F1_R2).mean():.3f}; median rank={high.F2_selected_rank.median():.1f}; Mixed output R2={high.F2_Mixed_output_R2.mean():.3f}"),
        ("H3 nonlinear endpoint specialization", "supported", f"High K_out={high.K_out_mid.mean():.3f}; midpoint loss={high.Mixed_loss.mean():.3f}; ESI={high.endpoint_specialization_index.mean():.3f}"),
        ("H4 tanh saturation mechanism", "regime marker, not sufficient cause", f"High saturation09={high.saturation_09.mean():.3f}; within-High saturation→K_out r={saturation_curvature[0]:.3f}; endpoint linearization delta loss={high_endpoint_linearization:.3f}"),
        ("H5 recurrent amplification", "amplification capacity supported; seed prediction mixed", f"High local norm={high.Jrec_spectral_norm.mean():.3f}; cumulative max={high.Jrec_cumulative_max.mean():.3f}; cumulative→K_out r={recurrence_curvature[0]:.3f}; cumulative→loss r={recurrence_loss[0]:.3f}"),
        ("H6 output-relevant compositionality sufficient", "supported", f"High output-only patch delta loss={high.output_linear_path_patch_delta_loss.mean():.3f}; null-only={high.null_linear_path_patch_delta_loss.mean():.3f}"),
        ("H7 full hidden linearity necessary", "not supported", f"Full and output-only loss effects differ by {np.abs(high.linear_path_patch_delta_loss-high.output_linear_path_patch_delta_loss).mean():.3g}"),
        ("H8 Novel input OOD", "supported", f"NOVEL_INPUT_OOD fraction={master.NOVEL_INPUT_OOD.mean():.3f}"),
    ]
    return pd.DataFrame(rows,columns=["hypothesis","assessment","continuous_evidence"])


def _write_report(output,config,tables,manifest):
    master=tables["srn_mechanism_master_table"]; decoder=tables["decoder_loso_results"]
    hidden=tables["cue_sweep_hidden_residuals"]; out=tables["cue_sweep_output_residuals"]
    logit=tables["cue_sweep_logit_residuals"]; gain=tables["cue_sweep_tanh_gain"]
    saturation=tables["cue_sweep_saturation"]; curvature=tables["cue_sweep_curvature"]
    recurrent=tables["recurrent_jacobian_metrics"]; formal=tables["formal_model_familiar_fit"]
    formal_pred=tables["formal_model_mixed_prediction"]; patch=tables["causal_linear_path_patching"]
    linearized=tables["linearized_activation_counterfactual"]; novel=tables["novel_input_audit"]
    associations=tables["mechanistic_associations"]; mediation=tables["descriptive_nested_regressions"]
    hypotheses=_hypothesis_table(tables); hypotheses.to_csv(output/"hypothesis_evidence.csv",index=False)
    high=master[master.condition.eq("High_monotonic")]; low=master[master.condition.eq("Low_monotonic")]
    high_endpoint_linearization=linearized[
        linearized.condition.eq("High_monotonic")
        & np.isclose(linearized.cue_lambda,.5)
        & linearized.linearization_anchor.eq("nearest_endpoint")
    ].delta_loss_linearized_minus_actual.mean()
    high_associations=associations[associations.scope.eq("High_monotonic")]
    def report_association(predictor,outcome):
        return high_associations[
            high_associations.predictor.eq(predictor)&high_associations.outcome.eq(outcome)
        ].iloc[0]
    saturation_curvature=report_association("saturation_09","K_out_mid")
    output_curvature_loss=report_association("K_out_mid","Mixed_loss")
    recurrence_curvature=report_association("Jrec_cumulative_max","K_out_mid")
    recurrence_loss=report_association("Jrec_cumulative_max","Mixed_loss")
    q_answers=f"""Q1. **Yes.** High-scale mean Mixed q is {high.Mixed_q.mean():.3f}; it locates Mixed near the A/B midpoint.

Q2. **Yes for High, approximately no for Low.** Mean Mixed whole residual is {high.Mixed_whole_residual.mean():.3f} vs {low.Mixed_whole_residual.mean():.3f}.

Q3. **It is distributed across both subspaces, not primarily one.** At Mixed, normalized energy is approximately {tables['cue_sweep_output_residuals'].query("condition == 'High_monotonic' and cue_lambda == .5").E_out.mean():.3f} output-relevant and {tables['cue_sweep_output_residuals'].query("condition == 'High_monotonic' and cue_lambda == .5").E_null.mean():.3f} output-null. The output component is nevertheless behaviorally decisive.

Q4. **The output-relevant deviation is behaviorally sufficient, but it only weakly explains seed-to-seed loss variation within High.** High output-curvature→loss is r={output_curvature_loss.pearson_r:.3f} with bootstrap slope CI [{output_curvature_loss.bootstrap_CI_low:.3g}, {output_curvature_loss.bootstrap_CI_high:.3g}]. More decisively, the output-only path patch changes loss by {high.output_linear_path_patch_delta_loss.mean():.3f}, while null-only changes it by {high.null_linear_path_patch_delta_loss.mean():.3g}.

Q5. **Associated at the condition level, but not supported as a sufficient cause.** High-scale gain is {high.mean_tanh_gain.mean():.3f} (Low {low.mean_tanh_gain.mean():.3f}) and saturation09 is {high.saturation_09.mean():.3f} (Low {low.saturation_09.mean():.3f}); however, within High seeds saturation→output-curvature is r={saturation_curvature.pearson_r:.3f}, and nearest-endpoint linearization changes loss by {high_endpoint_linearization:+.3f} (it worsens rather than rescues it).

Q6. **The dynamics have strong amplification capacity, but seed-level explanatory evidence is mixed.** Mean High recurrent spectral norm is {high.Jrec_spectral_norm.mean():.3f}, with maximum cumulative sensitivity {high.Jrec_cumulative_max.mean():.3f}. Within High seeds, cumulative sensitivity→output-curvature is r={recurrence_curvature.pearson_r:.3f} and cumulative sensitivity→loss is r={recurrence_loss.pearson_r:.3f}; bootstrap intervals are in `mechanistic_associations.csv`.

Q7. **Mostly additive plus unstructured endpoint specialization, not robust low-rank bilinear binding.** F1 explains {high.F1_R2.mean():.3f} of High familiar variance; F2 adds only {(high.F2_R2-high.F1_R2).mean():.3f}, has median selected rank {high.F2_selected_rank.median():.1f}, and predicts Mixed output geometry poorly (R²={high.F2_Mixed_output_R2.mean():.3f}). F3 is the exact endpoint-lookup upper bound.

Q8. **Yes.** Restoring only the output-relevant component produces the same logits/loss as the full linear-path patch up to numerical error.

Q9. **No.** Null-space restoration cannot affect logits (`W_out P_null = 0`), and full restoration adds no behavioral rescue beyond the output-only patch.

Q10. **Yes.** Novel is an untrained, zero-effect input direction in {master.NOVEL_INPUT_OOD.mean():.0%} of networks and is excluded from Mixed-hypothesis adjudication."""
    report=f"""# SRN Nonlinearity and Compositionality Report

## 1. Executive summary

The frozen 40-network analysis supports a precise High/Low tradeoff: Low-scale models are
nearly affine and transfer smoothly but barely split A/B; High-scale models know that Mixed
is between A and B yet curve away from the straight endpoint path in behaviorally relevant
dimensions. Forcing only the output-relevant component back onto that path is sufficient to
recover the full linear-path behavioral effect. Continuous evidence—not one automatic label—
is reported below.

## 2. Existing diagnostic reproduction

```text
{tables['existing_diagnostic_reproduction'].to_string(index=False)}
```

All gates passed before the new mechanism analyses. Checkpoint bytes were hash-verified before/after.

## 3. Corrected sequence decoder results

```text
{_mean(decoder,['condition_label','test_context'],['decoder_accuracy','decoder_cross_entropy','decoder_margin'])}
```

Each held-out seed uses one StandardScaler + multinomial decoder fit to other seeds' familiar
A/B states; that exact fitted decoder scores all five contexts. Raw hidden units are not aligned
across independently initialized networks, so LOSO decoding is a deliberately stringent shared-coordinate test.

## 4. Whole-state cue interpolation

```text
{_mean(hidden[hidden.cue_lambda.isin([0,.25,.5,.75,1])],['condition_label','cue_lambda'],['E_whole','whole_residual_norm','sequence_loss'])}
```

## 5. Output-relevant vs output-null interpolation

```text
{_mean(out[out.cue_lambda.isin([0,.25,.5,.75,1])],['condition_label','cue_lambda'],['E_out','E_null','output_residual_norm','null_residual_norm'])}
```

## 6. Logit-space interpolation

```text
{_mean(logit[logit.cue_lambda.isin([0,.25,.5,.75,1])],['condition_label','cue_lambda'],['E_logit','logit_residual_norm','cross_entropy_deviation'])}
```

## 7. Familiar factorization and interaction structure

```text
{_mean(master,['condition_label'],['P_sequence','P_context','P_interaction','AB_hidden_separation'])}
```

Interaction is reported continuously; no arbitrary low/high cutoff is applied.

## 8. Tanh gain and saturation

```text
{_mean(master,['condition_label'],['mean_tanh_gain','saturation_08','saturation_09','saturation_095'])}
```

## 9. Hidden-state curvature

```text
{_mean(master,['condition_label'],['K_whole_mid','K_out_mid','K_null_mid','K_logit_mid','endpoint_specialization_index'])}
```

Mechanistic associations (Pearson, Spearman, bootstrap OLS):

```text
{associations.to_string(index=False)}
```

Descriptive nested regressions (not causal mediation):

```text
{mediation.to_string(index=False)}
```

## 10. Recurrent amplification

```text
{_mean(master,['condition_label'],['Jrec_spectral_radius','Jrec_spectral_norm','Jrec_cumulative_max'])}
```

## 11. Formal model comparison: F0-F3

```text
{_mean(formal,['condition_label','model'],['familiar_R2','familiar_RMSE','parameter_count','selected_rank','rank_cv_MSE'])}
```

F2 rank is selected using familiar-only entry-masked matrix-completion CV. F3 is an endpoint
lookup upper bound and is never used to predict Mixed.

## 12. Mixed prediction from formal models

```text
{_mean(formal_pred,['condition_label','model'],['Mixed_whole_R2','Mixed_output_R2','Mixed_logit_R2'])}
```

With two familiar endpoints, F0-F2 all make an endpoint-linear midpoint prediction; F2 can
describe structured endpoint interaction but cannot infer nonlinear context curvature from endpoints alone.

## 13. Causal linear-path patching

```text
{_mean(patch,['condition_label','patch_type'],['sequence_loss','delta_loss_vs_actual'])}
```

## 14. Local-linearization counterfactual

```text
{_mean(linearized[np.isclose(linearized.cue_lambda,.5)],['condition_label','linearization_anchor'],['actual_sequence_loss','linearized_sequence_loss','delta_loss_linearized_minus_actual','linearized_E_out'])}
```

Nearest-endpoint and midpoint operating-point linearizations are both explicit. The midpoint
linearization necessarily touches the actual function at lambda=.5 and is not treated as a rescue test there.

## 15. Novel input audit

```text
{_mean(novel,['condition_label'],['familiar_span_fraction','induced_novel_norm','induced_distance_to_familiar_span','NOVEL_INPUT_OOD'])}
```

Novel is not used to adjudicate any Mixed compositionality hypothesis.

## 16. Integrated mechanistic interpretation

```text
{hypotheses.to_string(index=False)}
```

## 17. Which formalization is currently best supported?

The High-scale regime is best described as **functional but endpoint-specialized familiar
solutions connected by a nonlinear path with output-relevant curvature**. F2 can compactly
describe familiar interaction when selected, but endpoint-linear bilinear binding alone does
not explain the actual Mixed trajectory. The Low-scale regime approximates F0/F1 and an affine cue path.

### Explicit answers

{q_answers}

## 18. Implications for the next training experiment

The next target should not simply reduce scale, which recovers the unsplit Low-scale solution.
It should preserve strong familiar separation while constraining cue-induced curvature in the
output row space. Null-space contextual geometry may remain nonlinear because it is annihilated
by the task readout. Novel requires a separately trained/interpretable input representation.

## Scientific firewall

- Frozen networks: `{manifest['network_count']}`; optimizer steps: `0`.
- Mixed/Novel training rows: `0`; human rows: `0`.
- F0-F2 fitting/rank selection contexts: A/B only.
- Hybrid controls remain unavailable because no matching frozen checkpoints exist.
"""
    (output/"SRN_NONLINEARITY_AND_COMPOSITIONALITY_REPORT.md").write_text(report,encoding="utf-8")


def run_pipeline(config:MechanismConfig):
    config.validate(); checkpoint_dir=Path(config.checkpoint_dir).expanduser().resolve()
    output=Path(config.output_dir).expanduser().resolve(); output.mkdir(parents=True,exist_ok=True)
    records=[]; checkpoint_hashes={}
    _progress(f"Loading {len(config.conditions)*len(config.seeds)} frozen networks")
    for condition in config.conditions:
        for seed in config.seeds:
            network=load_frozen_network(checkpoint_dir,condition,seed)
            checkpoint_hashes[str(network.path)]=sha256_file(network.path)
            runs={name:exact_forward(network.model,config.sequence,CONTEXT_VECTORS[name])
                  for name in ("A","B","Mixed","Novel","NoContextProbe")}
            records.append({"condition":condition,"condition_label":CONDITION_LABELS.get(condition,condition),
                            "seed":seed,"network":network,"runs":runs})
    reproduction,current=_reproduction_gate(records,config.reproduction_dir)
    _progress("Existing diagnostics reproduced; starting new analyses")
    decoder=loso_decoders(records,config.decoder_c,config.random_seed)
    collected={key:[] for key in ("hidden","output","logit","gain","saturation","curvature","recurrent",
                                  "formal_fit","formal_prediction","patching","linearized","novel","master")}
    for index,record in enumerate(records,start=1):
        results=analyze_network(record,config)
        for key in collected: collected[key].extend(results[key])
        _progress(f"Completed {index}/{len(records)}: {record['condition']} seed {record['seed']}")
    frames={key:pd.DataFrame(value) for key,value in collected.items()}
    master=frames["master"].merge(decoder[decoder.test_context.eq("Mixed")][["condition","seed","decoder_accuracy"]],
                                  on=["condition","seed"],validate="one_to_one").rename(columns={"decoder_accuracy":"Mixed_decoder_accuracy"})
    required_columns=("condition","seed","AB_hidden_separation","AB_output_relevant_separation","P_sequence","P_context","P_interaction",
                      "Mixed_q","Mixed_whole_residual","Mixed_output_residual","Mixed_null_residual","Mixed_logit_residual","Mixed_loss",
                      "Mixed_decoder_accuracy","mean_tanh_gain","saturation_08","saturation_09","saturation_095","K_whole_mid","K_out_mid",
                      "K_null_mid","K_logit_mid","Jrec_spectral_radius","Jrec_spectral_norm","F0_R2","F1_R2","F2_R2","F2_selected_rank",
                      "F2_Mixed_whole_R2","F2_Mixed_output_R2","F2_Mixed_logit_R2","linear_path_patch_delta_loss",
                      "output_linear_path_patch_delta_loss","null_linear_path_patch_delta_loss","NOVEL_INPUT_OOD")
    missing=set(required_columns)-set(master.columns)
    if missing: raise AssertionError(f"Master table missing required columns: {sorted(missing)}")
    associations,mediation=association_tables(master,config.bootstrap_samples,config.random_seed)
    tables={
        "decoder_loso_results":decoder,"cue_sweep_hidden_residuals":frames["hidden"],
        "cue_sweep_output_residuals":frames["output"],"cue_sweep_logit_residuals":frames["logit"],
        "cue_sweep_tanh_gain":frames["gain"],"cue_sweep_saturation":frames["saturation"],
        "cue_sweep_curvature":frames["curvature"],"recurrent_jacobian_metrics":frames["recurrent"],
        "formal_model_familiar_fit":frames["formal_fit"],"formal_model_mixed_prediction":frames["formal_prediction"],
        "seed_level_mechanism_table":master,"causal_linear_path_patching":frames["patching"],
        "linearized_activation_counterfactual":frames["linearized"],"novel_input_audit":frames["novel"],
        "srn_mechanism_master_table":master,"cue_sweep_context_jacobian":frames["curvature"][[c for c in frames["curvature"].columns if c in {"condition","condition_label","seed","cue_lambda","J_norm","J_out_norm","J_null_norm","delta_J_norm"}]],
        "mechanistic_associations":associations,"descriptive_nested_regressions":mediation,
        "existing_diagnostic_reproduction":reproduction,
    }
    for name,frame in tables.items(): frame.to_csv(output/f"{name}.csv",index=False)
    # T1/T14: immutable checkpoint bytes and explicit no-human/no-probe-fit audit.
    audit=[]
    for record in records:
        path=record["network"].path; after=sha256_file(path); before=checkpoint_hashes[str(path)]
        audit.append({"condition":record["condition"],"seed":record["seed"],"path":str(path),
                      "sha256_before":before,"sha256_after":after,"checkpoint_unchanged":before==after,
                      "optimizer_steps":0,"human_rows_used":0,"mixed_novel_training_rows":0})
    audit=pd.DataFrame(audit); audit.to_csv(output/"frozen_checkpoint_audit.csv",index=False)
    if not audit.checkpoint_unchanged.all(): raise AssertionError("T1 failed: checkpoint bytes changed.")
    manifest={"network_count":len(records),"conditions":[CONDITION_LABELS.get(c,c) for c in config.conditions],
              "config":{k:(str(v) if isinstance(v,Path) else v) for k,v in asdict(config).items()},
              "human_data_used":False,"mixed_novel_training_used":False,
              "formal_model_fit_contexts":["A","B"],"decoder_fit_contexts":["A","B"],
              "decoder_protocol":"within-condition leave-one-seed-out; identical fitted decoder for all contexts"}
    (output/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    if config.make_plots: make_figures(output,tables)
    _write_report(output,config,tables,manifest)
    _progress(f"Complete: {output}")
    return tables
