from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import hashlib,json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from srn_training_timeline_audit.core import sha256_file
from .analysis import bootstrap_summary,classify,primary_comparisons,seed_summary
from .config import ExperimentConfig
from .core import (add_novel_mean,all_intervention_rows,assert_mixed_cue,load_final_model,
                   natural_reference_rows,replay_acquisition)
from .plotting import enforce_figures,make_figures


def _stamp():return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
def _log(message):print(f"[{_stamp()}] PHASE1J | {message}",flush=True)
def _source_hash():
    digest=hashlib.sha256()
    for name in ("config.py","core.py","analysis.py","plotting.py","pipeline.py"):digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()
def _git_commit(repo):
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:return "unavailable:not-a-git-worktree"


def _run_or_load(config,output,family,seed):
    partial=output/"partials";partial.mkdir(parents=True,exist_ok=True);stem=f"{family}_seed_{seed}"
    metric_path=partial/f"{stem}_transition.csv";acq_path=partial/f"{stem}_acquisition.csv";meta_path=partial/f"{stem}_metadata.json"
    if metric_path.exists() and acq_path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta.get("config_sha256")!=config.config_sha256 or meta.get("source_sha256")!=_source_hash():raise RuntimeError(f"Partial Phase 1J result uses another frozen specification: {stem}")
        _log(f"resuming completed {family} seed {seed}")
        return pd.read_csv(metric_path),pd.read_csv(acq_path),meta["provenance"]
    model,policy,provenance=load_final_model(config,family,seed);parameter_hash=provenance["parameter_hash"]
    natural,rolls,midpoint,midpoint_task=natural_reference_rows(model,policy,config,family,seed,provenance["source_checkpoint_hash"],parameter_hash)
    interventions=all_intervention_rows(model,policy,config,family,seed,provenance["source_checkpoint_hash"],parameter_hash,rolls,midpoint,midpoint_task)
    metrics=add_novel_mean(pd.concat([natural,interventions],ignore_index=True))
    acquisition=replay_acquisition(config,family,seed)
    metrics.to_csv(metric_path,index=False);acquisition.to_csv(acq_path,index=False)
    meta_path.write_text(json.dumps({"config_sha256":config.config_sha256,"source_sha256":_source_hash(),"provenance":provenance},indent=2,sort_keys=True))
    return metrics,acquisition,provenance


def _assert_clamp_equalities(metrics):
    keys=["family","seed","probe","injection_position","transition","intervention"]
    def compare(probe,left,right):
        subset=metrics[metrics.probe.eq(probe)&metrics.intervention.eq("single_midpoint")]
        a=subset[subset.context_clamp.eq(left)].set_index(keys).sort_index();b=subset[subset.context_clamp.eq(right)].set_index(keys).sort_index()
        for column in ("raw_ce","hidden_midpoint_residual","task_midpoint_residual"):
            if not np.array_equal(a[column].to_numpy(),b[column].to_numpy()):raise AssertionError("FAIL: CONTEXT_CLAMP_IDENTITY")
    compare("Mixed","D0_natural","D2_composite");compare("NoContext","D0_natural","D1_zero")


def _write_report(output,summary,statistics,decision):
    code,label=decision;means=summary.groupby(["family","probe","intervention","context_clamp"]).mean(numeric_only=True)
    focus=means.loc[(slice(None),["Mixed","NoContext","NovelMean"],"single_midpoint",slice(None)),
                    ["immediate_rescue","remaining_sequence_rescue","mean_remaining_ce_difference","behavioral_persistence_score","mean_post_patch_hidden_residual","fraction_upper_bound_retained"]]
    text=f"""# Phase 1J Single-Injection Context Stability Report

## 1. Executive summary

**Outcome {code} — {label}.** This classification follows the frozen qualitative decision table.

## 2. Provenance audit

All final High/Low checkpoints are registered canonical 8-unit tanh SRNs at scales 2.0/.02. Source file hashes,
cue/sequence hashes, and output-layer hashes are in `phase1j_provenance.csv`. Canonical acquisition was replayed
independently and matched every final checkpoint exactly.

## 3. Exact midpoint/composite verification

The target is exactly `m=(h_A+h_B)/2`, and the Mixed cue is exactly `(c_A+c_B)/2`. Targets use hidden states,
then pass through the actual output layer; probabilities are never averaged.

## 4. Natural probe trajectories

Natural A, B, Mixed, NoContext, Novel1--4, and within-seed NovelMean raw transition CE and distances are retained.

## 5. Immediate single-injection rescue

Each run replaces the hidden state once at one of seven positions, computes that transition's output, and passes
the replacement into the next recurrent step. Patch count and inheritance are asserted.

## 6. Post-injection hidden-state drift

FIG04 reports normalized hidden distance to the position-matched familiar midpoint at every available lag.

## 7. Post-injection task-space drift

FIG05 repeats the persistence analysis in centered-logit space.

## 8. Behavioral rescue persistence

Raw rescue is baseline CE minus intervention CE. Normalized persistence is exported only when immediate rescue
has a stable denominator. Remaining-sequence sums and means retain the number of available transitions.

## 9. Context-clamp decomposition

Mixed D0 equals D2 exactly; NoContext D0 equals D1 exactly. D1 removes the post-patch cue, while D2 supplies
the physical Mixed/composite cue. All comparisons use fixed patch strength 1.0.

## 10. Directional controls

Wrong-position derangement, deterministic norm-matched orthogonal, A endpoint, and B endpoint controls use one
patch under natural cue continuation.

## 11. Repeated-patch upper bound versus single-patch persistence

Repeated midpoint patching is an upper-bound state-sufficiency reference. Denominator instability is explicitly
flagged before calculating the fraction retained by a single patch.

## 12. High-versus-Low comparison

```text
{focus.to_string()}
```

## 13. Per-seed heterogeneity

All primary comparisons and bootstrap intervals resample seeds. Novel1--4 are nested within seed and averaged
before inference.

## 14. Decision-table outcome

**Outcome {code}: {label}.**

## 15. Implications for the human averaging account

Only Outcome A supports a full contextual-stabilization account. Outcome B leaves the stabilizing source
unresolved; C localizes failure to initial state generation; D implies local sufficiency without maintenance;
E indicates nonspecific perturbation rescue.

## 16. Limitations

Teacher-forced external tokens are used throughout. A one-time intervention tests persistence after state
placement, not natural state generation, and no attractor or stable-manifold language is warranted without
the corresponding lag-level evidence.
"""
    (output/"PHASE1J_SINGLE_INJECTION_CONTEXT_STABILITY_REPORT.md").write_text(text)


def run_pipeline(config:ExperimentConfig):
    config.validate();assert_mixed_cue();output=config.resolve(config.output_dir);output.mkdir(parents=True,exist_ok=True);repo=Path(config.repo_dir).resolve()
    _log(f"starting {config.profile} read-only stability audit")
    metrics=[];acquisition=[];provenance=[]
    for family in config.families:
        for seed in config.seeds:
            _log(f"analyzing {family} seed {seed}");m,a,p=_run_or_load(config,output,family,seed);metrics.append(m);acquisition.append(a);provenance.append(p)
    metrics=pd.concat(metrics,ignore_index=True);acquisition=pd.concat(acquisition,ignore_index=True);provenance=pd.DataFrame(provenance)
    _assert_clamp_equalities(metrics)
    if not acquisition.parameters_changed.all() or acquisition.groupby(["family","seed"]).parameter_hash.nunique().min()<=1:raise AssertionError("FAIL: ACQUISITION_PROVENANCE")
    summary=seed_summary(metrics,config);comparisons=primary_comparisons(summary,config);statistics=bootstrap_summary(comparisons,config);outcome=classify(summary)
    git_commit=_git_commit(repo)
    for frame in (metrics,acquisition,summary,comparisons,statistics,provenance):frame["git_commit"]=git_commit
    tables={"phase1j_transition_metrics.csv":metrics,"phase1j_seed_summary.csv":summary,
            "phase1j_statistical_summary.csv":statistics,"phase1j_provenance.csv":provenance,
            "phase1j_primary_comparisons.csv":comparisons,"phase1j_true_acquisition.csv":acquisition}
    for name,frame in tables.items():frame.to_csv(output/name,index=False)
    if config.make_plots:make_figures(output,acquisition,metrics,summary,config);enforce_figures(output)
    _write_report(output,summary,statistics,outcome)
    manifest={"phase":"Phase1J","profile":config.profile,"created_at":_stamp(),"read_only_sources":True,
              "source_artifacts_modified":False,"config":asdict(config),"config_sha256":config.config_sha256,
              "source_sha256":_source_hash(),"source_config_sha256":sha256_file(config.resolve(config.source_config)),
              "git_commit":git_commit,"patch_strength":1.0,"teacher_forced_tokens":True,"parameter_updates_during_interventions":0,
              "mixed_D0_equals_D2":True,"nocontext_D0_equals_D1":True,
              "novel_mean_members":["Novel1","Novel2","Novel3","Novel4"],
              "outcome":{"code":outcome[0],"label":outcome[1]}}
    (output/"PHASE1J_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str))
    _log(f"complete: Outcome {outcome[0]} — {outcome[1]}");return {"output":output,"outcome":outcome,"summary":summary}
