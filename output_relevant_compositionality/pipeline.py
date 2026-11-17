from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .phase1 import (
    CONDITIONS,baseline_separation_target,evaluate_model,selection_metrics,train_model,
)
from .phase2 import (
    ECOLOGIES,FAMILIES,evaluate_pretraining,exposure_table,human_finetune,train_pretraining,
)
from .plotting import make_phase1_figures,make_phase2_figures


PHASE1_TABLES=("phase1_training_summary","phase1_seed_metrics","phase1_cue_sweep",
               "phase1_curvature_metrics","phase1_patching","phase1_hyperparameter_selection")
PHASE2_TABLES=("phase2_pretraining_generalization","phase2_pretraining_compositionality",
               "phase2_human_transfer","phase2_seed_metrics","phase2_formal_model_fit")


def _progress(message):
    stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] OUTPUT-COMPOSITIONALITY | {message}",flush=True)


def _select_hyperparameters(config,separation_target):
    baseline=[]
    for seed in config.selection_seeds:
        result=train_model(config,"C0_high_baseline",seed)
        baseline.append(selection_metrics(config,result.model))
    baseline=pd.DataFrame(baseline).mean(numeric_only=True)
    rows=[];selected={}
    targets={"C1_output_curvature":"K_logit_mid","C2_whole_curvature":"K_whole_mid","C3_null_curvature":"K_null_mid"}
    for condition,target_metric in targets.items():
        _progress(f"Hyperparameter grid: {condition}")
        for regularizer_weight in config.regularizer_grid:
            for separation_weight in config.separation_grid:
                seed_metrics=[]
                for seed in config.selection_seeds:
                    result=train_model(config,condition,seed,regularizer_weight,separation_weight,separation_target)
                    seed_metrics.append(selection_metrics(config,result.model))
                mean=pd.DataFrame(seed_metrics).mean(numeric_only=True)
                separation_ratio=mean.AB_hidden_separation/separation_target
                familiar_ratio=mean.familiar_loss/max(baseline.familiar_loss,1e-12)
                curvature_ratio=mean[target_metric]/max(baseline[target_metric],1e-12)
                rows.append({"condition":condition,"regularizer_weight":regularizer_weight,
                             "separation_weight":separation_weight,"selection_seed_count":len(config.selection_seeds),
                             "familiar_loss":mean.familiar_loss,"AB_hidden_separation":mean.AB_hidden_separation,
                             "K_whole_mid":mean.K_whole_mid,"K_out_mid":mean.K_out_mid,"K_null_mid":mean.K_null_mid,
                             "K_logit_mid":mean.K_logit_mid,"separation_ratio_to_frozen_target":separation_ratio,
                             "familiar_loss_ratio_to_C0":familiar_ratio,"target_curvature_metric":target_metric,
                             "curvature_ratio_to_C0":curvature_ratio,
                             "basic_eligible":separation_ratio>=config.separation_preservation and familiar_ratio<=1+config.familiar_loss_tolerance,
                             "target_reduction_met":curvature_ratio<=config.curvature_ratio_gate,
                             "selection_used_mixed_loss":False,"selection_used_novel":False})
        frame=pd.DataFrame([row for row in rows if row["condition"]==condition])
        qualified=frame[frame.basic_eligible&frame.target_reduction_met]
        if len(qualified): choice=qualified.sort_values(["regularizer_weight","separation_weight"]).iloc[0]
        else:
            eligible=frame[frame.basic_eligible]
            choice=(eligible if len(eligible) else frame).sort_values(["curvature_ratio_to_C0","regularizer_weight","separation_weight"]).iloc[0]
        selected[condition]=(float(choice.regularizer_weight),float(choice.separation_weight))
        for row in rows:
            if row["condition"]==condition: row["selected"]=(row["regularizer_weight"],row["separation_weight"])==selected[condition]
    return pd.DataFrame(rows),selected


def _phase1_gate(config,metrics):
    means=metrics.groupby("condition").mean(numeric_only=True)
    c0,c1,c4=(means.loc[name] for name in ("C0_high_baseline","C1_output_curvature","C4_low_baseline"))
    rows=[
        ("AB separation >= 90% C0",c1.AB_hidden_separation/c0.AB_hidden_separation,config.separation_preservation,"gte"),
        ("familiar loss <= 110% C0",c1.familiar_loss/c0.familiar_loss,1+config.familiar_loss_tolerance,"lte"),
        ("Mixed loss <= 75% C0",c1.Mixed_loss/c0.Mixed_loss,config.mixed_loss_ratio_gate,"lte"),
        ("K_out <= 50% C0",c1.K_out_mid/c0.K_out_mid,config.curvature_ratio_gate,"lte"),
        ("K_logit <= 50% C0",c1.K_logit_mid/c0.K_logit_mid,config.curvature_ratio_gate,"lte"),
        ("Mixed q near .5",abs(c1.Mixed_q-.5),.15,"lte"),
        ("separation not Low-like",c1.AB_hidden_separation/c4.AB_hidden_separation,config.low_separation_multiplier,"gte"),
    ]
    frame=pd.DataFrame(rows,columns=["criterion","observed","threshold","direction"])
    frame["passed"]=[row.observed>=row.threshold if row.direction=="gte" else row.observed<=row.threshold for row in frame.itertuples()]
    return frame,bool(frame.passed.all())


def _phase1_report(output,metrics,gate,passed,selection):
    means=metrics.groupby("condition")[["AB_hidden_separation","familiar_loss","Mixed_loss","Mixed_q","K_out_mid","K_logit_mid"]].mean()
    c0,c1,c2,c3,c4=(means.loc[name] for name in CONDITIONS)
    text=f"""# Phase 1: Output-Relevant Compositionality Sufficiency

## Executive result

The prespecified gate **{'PASSED' if passed else 'FAILED'}**. Hyperparameters were selected
using familiar A/B loss, familiar separation, and synthetic interpolation curvature only.

## Condition means

```text
{means.to_string()}
```

## Gate

```text
{gate.to_string(index=False)}
```

## Required questions

1. **Can output curvature be reduced without collapsing A/B separation?** C1 separation is
   {c1.AB_hidden_separation/c0.AB_hidden_separation:.1%} of C0 and output curvature is
   {c1.K_out_mid/c0.K_out_mid:.1%} of C0.
2. **Does reducing output curvature reduce Mixed loss?** C1 Mixed loss is
   {c1.Mixed_loss/c0.Mixed_loss:.1%} of C0.
3. **Is output-specific regularization sufficient?** {'Yes under every prespecified gate.' if passed else 'No under the complete prespecified gate.'}
4. **Does whole-state regularization add anything?** C2 Mixed loss={c2.Mixed_loss:.4f},
   C1={c1.Mixed_loss:.4f}; separation must be considered jointly.
5. **Does null-space regularization fail as predicted?** C3 Mixed loss={c3.Mixed_loss:.4f}
   versus C0={c0.Mixed_loss:.4f}.
6. **Does Mixed remain near q≈.5?** C1 q={c1.Mixed_q:.3f}.
7. **Does post-hoc patching become unnecessary?** See `phase1_patching.csv`; successful
   learning should make the C1 output-only patch delta approach zero.

No human, Novel, or Mixed behavioral outcome entered hyperparameter selection.

`PHASE1_GATE = {'PASS' if passed else 'FAIL'}`
"""
    (output/"PHASE1_OUTPUT_RELEVANT_COMPOSITIONALITY_SUFFICIENCY.md").write_text(text,encoding="utf-8")


def _phase2_internal_gate(general,formal):
    rows=[]
    for family in FAMILIES:
        means=general[general.family.eq(family)].groupby("ecology").mean(numeric_only=True)
        structured,shuffled=means.loc["structured"],means.loc["shuffled"]
        f2=formal[(formal.family==family)&(formal.model=="F2")].groupby("ecology").mean(numeric_only=True)
        criteria={"heldout_accuracy":structured.heldout_accuracy>shuffled.heldout_accuracy,
                  "heldout_loss":structured.heldout_loss<shuffled.heldout_loss,
                  "K_out":structured.K_out<shuffled.K_out,"K_logit":structured.K_logit<shuffled.K_logit,
                  "F2_heldout_output_R2":f2.loc["structured"].heldout_output_R2>f2.loc["shuffled"].heldout_output_R2}
        rows.extend({"family":family,"criterion":name,"passed":value} for name,value in criteria.items())
    frame=pd.DataFrame(rows);family_pass=frame.groupby("family").passed.all()
    return frame,bool(family_pass.any()),list(family_pass[family_pass].index)


def _phase2_report(output,general,formal,human,gate,passing_families):
    pre=general.groupby(["family","ecology"])[["heldout_accuracy","heldout_loss","K_out","K_logit"]].mean()
    hum=human.groupby("source")[["AB_hidden_separation","Mixed_loss","K_out_mid","K_logit_mid"]].mean()
    text=f"""# Phase 2: Structured Experience Emergence

## Internal gate

Passing structured families: `{passing_families}`.

```text
{gate.to_string(index=False)}
```

## Pretraining

```text
{pre.to_string()}
```

## Human A/B transfer

```text
{hum.to_string()}
```

## Required questions

1. Held-out sequence×context generalization is reported against the exposure-matched shuffle.
2. No explicit curvature or separation regularizer was active during pretraining.
3. Low-rank F2 held-out predictions are in `phase2_formal_model_fit.csv`.
4. Structured and shuffled conditions are paired within family and seed.
5. Every transfer model received exactly 170 familiar A/B updates with a reset optimizer.
6. Mixed received zero optimizer steps and was evaluated only after freezing.
7. Separation and curvature are reported jointly, preventing weaker splitting from counting as success.
"""
    (output/"PHASE2_STRUCTURED_EXPERIENCE_EMERGENCE.md").write_text(text,encoding="utf-8")


def run_pipeline(config:ExperimentConfig):
    config.validate();torch.set_num_threads(1)
    output=Path(config.output_dir).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    checkpoints=output/"phase1_checkpoints";checkpoints.mkdir(exist_ok=True)
    separation_target,target_table=baseline_separation_target(config);target_table.to_csv(output/"phase1_frozen_separation_target.csv",index=False)
    _progress(f"Frozen familiar separation target = {separation_target:.4f}")
    selection,selected=_select_hyperparameters(config,separation_target);selection.to_csv(output/"phase1_hyperparameter_selection.csv",index=False)
    _progress(f"Selected Phase 1 hyperparameters: {selected}")
    training=[];metrics=[];sweeps=[];curves=[];patches=[]
    for condition in CONDITIONS:
        rw,sw=selected.get(condition,(0.,0.))
        for seed in config.seeds:
            result=train_model(config,condition,seed,rw,sw,separation_target)
            training.append(result.summary);torch.save(result.checkpoint,checkpoints/f"{condition}_seed_{seed}.pt")
            metric,sweep,curve,patch=evaluate_model(config,condition,seed,result.model)
            metrics.append(metric);sweeps.extend(sweep);curves.extend(curve);patches.extend(patch)
            _progress(f"Phase 1 {condition} seed {seed} complete")
    phase1={"phase1_training_summary":pd.DataFrame(training),"phase1_seed_metrics":pd.DataFrame(metrics),
            "phase1_cue_sweep":pd.DataFrame(sweeps),"phase1_curvature_metrics":pd.DataFrame(curves),
            "phase1_patching":pd.DataFrame(patches),"phase1_hyperparameter_selection":selection}
    gate,phase1_pass=_phase1_gate(config,phase1["phase1_seed_metrics"]);gate.to_csv(output/"phase1_gate.csv",index=False)
    for name,frame in phase1.items():frame.to_csv(output/f"{name}.csv",index=False)
    _phase1_report(output,phase1["phase1_seed_metrics"],gate,phase1_pass,selection)
    if config.make_plots:make_phase1_figures(output,phase1)

    phase2_status="not_run_phase1_failed";phase2={}
    if phase1_pass:
        _progress("PHASE1_GATE=PASS; beginning structured-experience emergence")
        pretrained={};pre_summary=[];general=[];formal=[];pair_rows=[]
        for seed in config.seeds:
            reference_exposure=None
            for family in FAMILIES:
                for ecology in ECOLOGIES:
                    model,spec,summary=train_pretraining(config,seed,family,ecology)
                    exposure=exposure_table(spec).drop(columns="transform_index")
                    if reference_exposure is None:reference_exposure=exposure
                    elif not reference_exposure.equals(exposure):raise AssertionError("T9 failed: structured/shuffled exposures differ.")
                    if exposure.query("heldout").empty or exposure.query("not heldout").empty:raise AssertionError("T10 failed: invalid heldout pairs.")
                    pretrained[(seed,family,ecology)]=model;pre_summary.append(summary)
                    result,formal_rows,pairs=evaluate_pretraining(config,seed,family,ecology,model,spec)
                    general.append(result);formal.extend(formal_rows);pair_rows.extend(pairs)
                    _progress(f"Phase 2 pretraining {family}/{ecology} seed {seed} complete")
        general=pd.DataFrame(general);formal=pd.DataFrame(formal)
        internal_gate,internal_pass,passing_families=_phase2_internal_gate(general,formal)
        internal_gate.to_csv(output/"phase2_internal_gate.csv",index=False)
        phase2_status="internal_gate_pass" if internal_pass else "internal_gate_fail"
        human=[]
        if internal_pass:
            for seed in config.seeds:
                _,metric,*_=human_finetune(config,seed,"random_high",None);human.append(metric)
                for family in FAMILIES:
                    for ecology in ECOLOGIES:
                        source=f"{ecology}_{family}"
                        _,metric,*_=human_finetune(config,seed,source,pretrained[(seed,family,ecology)]);human.append(metric)
            human=pd.DataFrame(human)
            _phase2_report(output,general,formal,human,internal_gate,passing_families)
        else:human=pd.DataFrame([{"status":"not_run_internal_gate_failed"}])
        phase2={"phase2_pretraining_generalization":general,
                "phase2_pretraining_compositionality":pd.DataFrame(pair_rows),
                "phase2_human_transfer":human,"phase2_seed_metrics":human.copy(),
                "phase2_formal_model_fit":formal}
        for name,frame in phase2.items():frame.to_csv(output/f"{name}.csv",index=False)
        pd.DataFrame(pre_summary).to_csv(output/"phase2_pretraining_summary.csv",index=False)
        if config.make_plots and internal_pass:make_phase2_figures(output,phase2)
    else:
        _progress("PHASE1_GATE=FAIL; Phase 2 is blocked by design")
        for name in PHASE2_TABLES:
            frame=pd.DataFrame([{"status":phase2_status,"PHASE1_GATE":"FAIL"}]);frame.to_csv(output/f"{name}.csv",index=False);phase2[name]=frame

    master=phase1["phase1_seed_metrics"].assign(phase="phase1")
    if phase1_pass and "phase2_human_transfer" in phase2 and "condition" in phase2["phase2_human_transfer"]:
        master=pd.concat([master,phase2["phase2_human_transfer"].assign(phase="phase2_human_transfer")],ignore_index=True,sort=False)
    master.to_csv(output/"output_relevant_compositionality_master.csv",index=False)
    manifest={"config":{k:(str(v) if isinstance(v,Path) else v) for k,v in asdict(config).items()},
              "PHASE1_GATE":"PASS" if phase1_pass else "FAIL","phase2_status":phase2_status,
              "human_data_used":False,"mixed_novel_task_training_rows":0,
              "hyperparameter_selection_signals":["familiar_A_B_loss","familiar_A_B_separation","synthetic_cue_curvature"]}
    (output/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    _progress(f"Complete: PHASE1_GATE={manifest['PHASE1_GATE']}; phase2={phase2_status}")
    return {**phase1,**phase2,"phase1_gate":gate,"master":master}
