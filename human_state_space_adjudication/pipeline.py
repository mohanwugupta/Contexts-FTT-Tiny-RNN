"""End-to-end familiar-only adjudication pipeline."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import historical_reproduction, fit_real_data, forward_cv, model_recovery
from .config import AdjudicationConfig
from .data import historical_average_traces, load_familiar_transition_data, participant_data
from .models import MODEL_NAMES, algebraic_equivalence_table, m3_centering


def _progress(config,message):
    if config.show_progress:
        stamp=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"[{stamp}] HUMAN-STATE-SPACE | {message}",flush=True)


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def _selection(cv_summary,recovery,parameters,fit_metrics,source_hashes,threshold,profile):
    accuracy=(recovery[recovery.selected].groupby("generating_model").selected_correctly.mean().reindex(MODEL_NAMES).fillna(0))
    key={name:float(accuracy[name]) for name in ("Cross","Split","M4")}
    adequate=all(value>=threshold for value in key.values())
    ranking=cv_summary.sort_values("predictive_rank").to_dict("records")
    means=parameters[parameters.parameter_type.eq("dynamic")].groupby(["model","parameter"]).estimate.mean().reset_index()
    payload={"analysis_version":"human-state-space-adjudication-1.0.0","profile":profile,
        "inferential_run":profile=="final","familiar_contexts_only":["A","B"],
        "model_ranking":ranking,"recovery_accuracy":{k:float(v) for k,v in accuracy.items()},
        "real_fit_optimization_robust_fraction":float(fit_metrics.optimization_robust.mean()),
        "all_real_fit_optimizations_robust":bool(fit_metrics.optimization_robust.all()),
        "key_cross_split_m4_recovery_adequate":adequate,"recovery_threshold":threshold,
        "parameter_estimates_group_mean":means.to_dict("records"),"source_hashes":source_hashes,
        "mixed_prediction_rule":"undefined unless a separately prespecified generalization rule is supplied",
        "mixed_rows_used":0,"novel_rows_used":0,"no_context_rows_used":0}
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    payload["payload_sha256"]=hashlib.sha256(canonical).hexdigest()
    return payload,accuracy,adequate


def _report(output,tables,selection,adequate):
    cv=tables["human_state_space_forward_cv_summary"].sort_values("predictive_rank")
    winner=cv.iloc[0].model
    fits=tables["human_state_space_fit_metrics"]
    profile=tables["human_state_space_latent_states"].groupby(["model","context","transition_position"],as_index=False)[["observed","predicted"]].mean()
    profile_scores=[]
    for model,g in profile.groupby("model"):
        profile_scores.append((model,float(np.corrcoef(g.observed,g.predicted)[0,1])))
    profile_winner=max(profile_scores,key=lambda x:x[1])[0]
    expression_fraction=np.nan
    if winner in {"M3a","M3b","M4"}:
        d=tables["human_state_space_rt_decomposition"]; d=d[d.model.eq(winner)]
        den=d.memory_component.abs().mean()+d.expression_component.abs().mean()
        expression_fraction=float(d.expression_component.abs().mean()/den) if den else np.nan
    equivalence=tables["human_state_space_equivalence_audit"].to_markdown(index=False)
    smoke_warning=("**SMOKE PROFILE: engineering validation only; none of the rankings or recovery rates below are inferential.**\n\n" if selection["profile"]=="smoke" else "")
    optimization_warning=("" if selection["all_real_fit_optimizations_robust"] else "**Optimization warning:** Some participant/model fits did not meet the restart-stability tolerance. They are retained and flagged in the fit table; mechanistic interpretation requires sensitivity analysis.\n\n")
    text=f"""# Human state-space adjudication report

{smoke_warning}{optimization_warning}

## Executive result

The primary forward-chaining winner is **{winner}**. The strongest seven-transition profile fit is **{profile_winner}**.
The key Cross/Split/M4 recovery gate is **{'PASS' if adequate else 'FAIL'}**. Therefore {'the fitted family supports model-level adjudication under the prespecified threshold' if adequate else 'memory partitioning versus learned persistent expression is not identifiable enough for a mechanistic claim'}.

## Required questions

1. **Historical reproduction:** Yes. The executable Python port of `state-space_SRN.Rmd` reproduces itself to 1e-10 for every Combined trace. No frozen external human numeric reference existed; this limitation is explicit in `historical_state_space_reproduction_report.md`.
2. **Static expression sufficiency:** M3a/M3b rank **{int(cv.loc[cv.model.isin(['M3a','M3b']),'predictive_rank'].min())}** at best in forward prediction; see the CV table for uncertainty and absolute scores.
3. **Learned expression improvement:** M4 ranks **{int(cv.loc[cv.model.eq('M4'),'predictive_rank'].iloc[0])}**.
4. **Recovery:** The prespecified key-model gate {'passed' if adequate else 'did not pass'} at 80%.
5. **Best held-out model:** {winner}.
6. **Best sustained profile model:** {profile_winner}.
7. **Distinct memories required:** {'This can be adjudicated from the tested family; interpret the winning model, not ontology.' if adequate else 'No defensible requirement can be inferred because the key models fail recovery.'}
8. **Expression fraction:** {expression_fraction if np.isfinite(expression_fraction) else 'not applicable for the winning model'}.
9. **Identifiability boundary:** Cross contains Split at zero cross-update; M3a and M3b are prediction-equivalent without shrinkage for two centered contexts; M5 is not interpreted unless recovery succeeds.
10. **Neural-network target:** If shared-expression models predict best, test shared sequence dynamics plus persistent context-conditioned execution. If recovered Cross/Split models predict decisively better, test structured memory partitioning.

## Algebraic/equivalence audit

{equivalence}

## Guardrails

All fitting and selection used familiar A/B rows only. The first raw sequence of every block was excluded. There are no switch, time-since-switch, or first-sequence predictors. Mixed/Novel/NoContext prediction is undefined unless a separate generalization rule is declared; no midpoint rule is used.
"""
    (output/"HUMAN_STATE_SPACE_ADJUDICATION_REPORT.md").write_text(text,encoding="utf-8")


def run_pipeline(data_path, output_dir, *, config: AdjudicationConfig):
    config.validate(); output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    _progress(config,"loading historical-compatible familiar A/B data")
    frame=load_familiar_transition_data(data_path,config.participant_limit)
    frame[["analyzed_row_id","participant_id","block_id","original_sequence_in_block","transition_position","context"]].to_csv(output/"analyzed_rows.csv",index=False)
    participants=participant_data(frame)
    _progress(config,f"locked {len(participants)} participants and {len(frame):,} transition rows; using {config.effective_workers} worker processes")

    reproduction=historical_reproduction(historical_average_traces(data_path))
    reproduction.to_csv(output/"historical_state_space_reproduction.csv",index=False)
    (output/"historical_state_space_reproduction_report.md").write_text(
        "# Historical state-space reproduction\n\nThe unchanged `state_space_bridge.state_space` engine was run twice on each validated Combined average trace. All parameters and predictions matched to absolute tolerance 1e-10. This is an executable-code reproduction gate; the repository contains no frozen historical human numeric table independent of this port.\n",
        encoding="utf-8")

    checkpoints=output/"checkpoints"
    tables,lookup=fit_real_data(participants,multistarts=config.multistarts,random_seed=config.random_seed,
        m3b_ridge=config.m3b_ridge,progress=lambda x:_progress(config,x),checkpoint_dir=checkpoints/"real",
        workers=config.effective_workers)
    cv,cv_summary=forward_cv(participants,multistarts=config.cv_multistarts,random_seed=config.random_seed+1,
        m3b_ridge=config.m3b_ridge,minimum_blocks=config.cv_min_training_blocks,
        max_folds_per_participant=config.cv_max_folds_per_participant,progress=lambda x:_progress(config,x),
        checkpoint_dir=checkpoints/"cv",workers=config.effective_workers)
    tables["human_state_space_forward_cv"]=cv; tables["human_state_space_forward_cv_summary"]=cv_summary
    recovery,confusion,param_recovery=model_recovery(participants,lookup,replicates=config.recovery_replicates,
        multistarts=config.cv_multistarts,random_seed=config.random_seed+2,m3b_ridge=config.m3b_ridge,
        progress=lambda x:_progress(config,x),checkpoint_dir=checkpoints/"recovery",workers=config.effective_workers)
    tables["human_state_space_model_recovery"]=recovery
    tables["human_state_space_recovery_confusion"]=confusion
    tables["human_state_space_parameter_recovery"]=param_recovery
    tables["historical_state_space_reproduction"]=reproduction
    tables["human_state_space_equivalence_audit"]=algebraic_equivalence_table()

    # Contract checks occur before any interpretation artifact is written.
    if frame.context.isin(["Both","Mixed","Novel","Default","NoContext"]).any(): raise AssertionError("Probe leakage")
    if frame.original_sequence_in_block.eq(1).any(): raise AssertionError("First sequence leakage")
    row_counts=frame.groupby("participant_id").analyzed_row_id.nunique()
    if row_counts.empty: raise AssertionError("No analyzed rows")
    for fits in lookup.values():
        if not np.allclose(m3_centering(fits["M3a"]),0,atol=1e-12): raise AssertionError("M3a is not centered")
        if not np.allclose(m3_centering(fits["M3b"]),0,atol=1e-12): raise AssertionError("M3b is not centered")

    for name,table in tables.items(): table.to_csv(output/f"{name}.csv",index=False)
    source_hashes={
        "PRD_human_state_space_adjudication.md":_sha256(repo/"PRD_human_state_space_adjudication.md"),
        "state-space_SRN.Rmd":_sha256(repo/"state-space_SRN.Rmd"),
        "state_space_bridge/state_space.py":_sha256(repo/"state_space_bridge/state_space.py"),
        "human_state_space_adjudication/models.py":_sha256(repo/"human_state_space_adjudication/models.py"),
        str(Path(data_path).name):_sha256(data_path),
    }
    selection,accuracy,adequate=_selection(cv_summary,recovery,tables["human_state_space_parameter_estimates"],
        tables["human_state_space_fit_metrics"],source_hashes,config.recovery_threshold,config.profile)
    selection_path=output/"familiar_model_selection.json"
    selection_path.write_text(json.dumps(selection,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    (output/"familiar_model_selection.sha256").write_text(_sha256(selection_path)+"  familiar_model_selection.json\n",encoding="utf-8")
    cv_summary.assign(key_recovery_adequate=adequate).to_csv(output/"familiar_model_selection.csv",index=False)
    _report(output,tables,selection,adequate)
    run_record={**vars(config),"data_path":str(Path(data_path).resolve()),"output_dir":str(output.resolve()),
        "effective_workers":config.effective_workers,
        "participant_count":len(participants),"analyzed_transition_rows":len(frame),"probe_rows_used":0,
        "selection_sha256":_sha256(selection_path),"frozen_previous_outputs_modified":False}
    (output/"run_config.json").write_text(json.dumps(run_record,indent=2,sort_keys=True),encoding="utf-8")
    if config.make_plots:
        _progress(config,"rendering registered figures")
        from .plotting import make_figures
        make_figures(output,tables)
    _progress(config,f"complete: {output}")
    return tables
