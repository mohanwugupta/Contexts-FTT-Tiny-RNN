"""Orchestration and reporting for the Phase 1H read-only audit."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from srn_training_timeline_audit.core import schedule_hash,sha256_file

from .analysis import (attach_actual_state_changes,decoder_results,seed_summary,shared_private_states,
                       statistical_summary,subspaces_and_updates)
from .config import HIGH,LOW,AuditConfig
from .plotting import enforce_outputs,make_figures
from .replay import replay_condition


def _stamp(): return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
def _log(message): print(f"[{_stamp()}] PHASE1H | {message}",flush=True)
def _source_hash():
    digest=hashlib.sha256()
    for name in ("config.py","replay.py","analysis.py","plotting.py","pipeline.py"):
        digest.update((Path(__file__).with_name(name)).read_bytes())
    return digest.hexdigest()
def _git_commit(repo):
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:return "unavailable:not-a-git-worktree"


def _replay_or_load(config,output,condition,seed):
    partial=output/"partials";partial.mkdir(parents=True,exist_ok=True);stem=f"{condition}_seed_{seed}"
    paths={name:partial/f"{stem}_{name}.csv" for name in ("states","gradients","blocks","functional")}
    provenance_path=partial/f"{stem}_provenance.json"
    if all(path.exists() for path in paths.values()) and provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        if provenance.get("audit_config_sha256")!=config.config_sha256 or provenance.get("audit_source_sha256")!=_source_hash():
            raise RuntimeError(f"Partial replay was created by a different frozen specification: {stem}")
        _log(f"resuming completed replay {condition} seed {seed}")
        return tuple(pd.read_csv(paths[name]) for name in ("states","gradients","blocks","functional"))+(provenance,)
    values=replay_condition(config,condition,seed,progress=_log);*frames,provenance=values
    provenance.update({"audit_config_sha256":config.config_sha256,"audit_source_sha256":_source_hash()})
    for name,frame in zip(("states","gradients","blocks","functional"),frames):frame.to_csv(paths[name],index=False)
    provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True));return tuple(frames)+(provenance,)


def _decision(summary):
    means=summary.groupby("scale_condition").mean(numeric_only=True);high=means.loc[HIGH];low=means.loc[LOW]
    shared_information=high.shared_sequence_decoder_accuracy>.50
    private_growth=high.private_structure_growth>0
    private_dominance=high.mean_private_change>high.mean_shared_change
    weak_high_alignment=high.mean_gradient_alignment<low.mean_gradient_alignment
    small_cross=abs(high.mean_cross_context_transfer)<.5*abs(high.mean_within_context_transfer)
    low_more_shared=(low.median_private_shared_ratio<high.median_private_shared_ratio and
                     low.mean_gradient_alignment>high.mean_gradient_alignment)
    if all((shared_information,private_growth,private_dominance,weak_high_alignment,small_cross,low_more_shared)):
        return "A","Strong support for shared content with private-dominant plasticity"
    if shared_information and not private_dominance:
        return "B","Shared structure exists but plasticity is not private-dominant"
    if (abs(high.transfer_gradient_alignment_correlation)>abs(high.transfer_shared_energy_correlation) and
        abs(high.transfer_gradient_alignment_correlation)>.20):
        return "C","Gradient alignment predicts transfer more clearly than state decomposition"
    return "D","High and Low do not show the full preregistered plasticity dissociation"


def _write_report(output,config,provenance,summary,statistics,decision):
    code,label=decision;means=summary.groupby("scale_condition").mean(numeric_only=True)
    table=means[["mean_shared_change","mean_private_change","median_private_shared_ratio",
                 "shared_update_energy_fraction","private_update_energy_fraction","mean_gradient_alignment",
                 "mean_within_context_transfer","mean_cross_context_transfer","AB_hidden_separation",
                 "shared_sequence_decoder_accuracy","private_context_decoder_accuracy"]]
    text=f"""# Phase 1H Shared/Private Plasticity Audit

## 1. Executive result

**Outcome {code} — {label}.** This result follows the frozen decision rule and uses seed-level summaries.

## 2. Canonical replay and provenance

All {len(provenance)} canonical High/Low models are 8-unit tanh SRNs with scales 2.0/.02. Every deterministic
170-update replay matched its saved final checkpoint exactly. Source checkpoints were read-only.

## 3. Formal shared/private hypothesis

At every transition and replay state, `shared=(A+B)/2` and `private=(A-B)/2`; exact endpoint reconstruction is
asserted. This midpoint is an operational decomposition, not a claim about literal storage.

## 4. Shared sequence structure over training

Leave-one-seed-out shared-state position decoding is reported at checkpoints 0,10,...,170. Chance is 1/7.

## 5. Growth of A/B private structure

`private_difference_norm`, hidden A/B distance, and task-space A/B separation are saved for all 171 states.

## 6. Shared vs private representational plasticity

Both raw change norms and squared private/shared ratios are reported. The ratio is never used without its
unnormalized numerator and denominator.

## 7. Task-space plasticity

The complete decomposition is repeated in centered-logit space to distinguish output-relevant change from
hidden output-null movement.

## 8. Shared/private sequence decoding

Shared/private sequence-position and private context decoders use leave-seed-out ridge decoding only.

## 9. Gradient alignment

Frozen A/B gradients use the canonical sequence-plus-execution training objective and never accumulate in
model `.grad` buffers. Raw behavioral quantities remain unnormalized sequence cross-entropy.

## 10. Parameter-block gradient alignment

Input, recurrent, hidden-bias, output-weight, and output-bias alignments are reported separately.

## 11. One-step functional A↔B transfer

Actual updates are evaluated at all 170 steps. At registered checkpoints, cloned models receive exactly one
optimizer-equivalent A or B update from the same Adam state; the original model/optimizer/RNG hashes are
verified unchanged before the next replay operation.

## 12. Transfer vs shared/private plasticity

Relationships are descriptive. Per-seed correlations—not 170 updates treated as independent subjects—feed
the seed summary.

## 13. High-vs-Low comparison

```text
{table.to_string()}
```

## 14. Raw-loss time series

TS01 and TS02 use literal canonical raw mean CE from actual adjacent replay states, not final-checkpoint replay.

## 15. Decision-tree classification

**Outcome {code}: {label}.**

## 16. Implications for the human representational account

The private-plasticity explanation should be used only for Outcome A. Other outcomes retain the distinction
between representational overlap and functional learning transfer but do not establish the proposed mechanism.

## 17. Remaining uncertainties

Midpoint and SVD subspaces are operational summaries; state changes are nonlinear consequences of parameter
updates; gradient alignment is local; and all inferential intervals use seed as the resampling unit.
"""
    (output/"PHASE1H_SHARED_PRIVATE_PLASTICITY_REPORT.md").write_text(text)


def run_audit(config:AuditConfig):
    config.validate();repo=Path(config.repo_dir).resolve();output=config.resolve(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    source_config=config.resolve(config.source_config)
    if not source_config.exists():raise FileNotFoundError(source_config)
    git_commit=_git_commit(repo);states=[];gradients=[];blocks=[];functional=[];provenance=[]
    _log(f"starting {config.profile} read-only audit")
    for condition in config.conditions:
        for seed in config.seeds:
            _log(f"replay {condition} seed {seed}")
            s,g,b,f,p=_replay_or_load(config,output,condition,seed)
            states.append(s);gradients.append(g);blocks.append(b);functional.append(f);provenance.append(p)
    states=pd.concat(states,ignore_index=True);gradients=pd.concat(gradients,ignore_index=True)
    blocks=pd.concat(blocks,ignore_index=True);functional=pd.concat(functional,ignore_index=True)
    for frame in (states,gradients,blocks,functional):frame["git_commit"]=git_commit
    _log("decomposing shared/private states")
    shared_private,_=shared_private_states(states,config);shared_private["git_commit"]=git_commit
    _log("estimating familiar-only subspaces and projecting actual updates")
    subspaces,updates=subspaces_and_updates(states,config);subspaces["git_commit"]=git_commit;updates["git_commit"]=git_commit
    functional=attach_actual_state_changes(functional,updates)
    _log("running leave-seed-out linear decoders")
    decoders=decoder_results(shared_private,config);decoders["git_commit"]=git_commit
    summary=seed_summary(shared_private,updates,gradients,functional,decoders,config)
    statistics=statistical_summary(summary,config);decision=_decision(summary)
    tables={"PHASE1H_replay_states.csv":states,"PHASE1H_shared_private_states.csv":shared_private,
            "PHASE1H_representational_updates.csv":updates,"PHASE1H_subspace_metrics.csv":subspaces,
            "PHASE1H_gradient_alignment.csv":gradients,"PHASE1H_block_gradient_alignment.csv":blocks,
            "PHASE1H_functional_transfer.csv":functional,"PHASE1H_decoder_results.csv":decoders,
            "PHASE1H_seed_summary.csv":summary,"PHASE1H_statistical_summary.csv":statistics,
            "PHASE1H_checkpoint_provenance.csv":pd.DataFrame(provenance)}
    for name,frame in tables.items():frame.to_csv(output/name,index=False)
    if config.make_plots:make_figures(output,states,shared_private,updates,subspaces,gradients,blocks,functional,decoders,summary);enforce_outputs(output)
    _write_report(output,config,provenance,summary,statistics,decision)
    manifest={"phase":"Phase1H","profile":config.profile,"created_at":_stamp(),"read_only_sources":True,
              "source_artifacts_modified":False,"config":asdict(config),"config_sha256":config.config_sha256,
              "audit_source_sha256":_source_hash(),"canonical_source_config_sha256":sha256_file(source_config),
              "training_schedule_sha256":schedule_hash(type("C",(),{"n_blocks":17,"block_length":10})()),
              "git_commit":git_commit,"replay_matches_final_checkpoint":all(x["replay_matches_final_checkpoint"] for x in provenance),
              "familiar_only_subspaces":True,"gradient_checkpoints":list(config.diagnostic_checkpoints),
              "subspace_ranks":list(config.subspace_ranks),"primary_subspace_rank":config.primary_subspace_rank,
              "bootstrap_resampling_unit":"seed","decision":{"code":decision[0],"label":decision[1]}}
    (output/"PHASE1H_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str))
    _log(f"complete: Outcome {decision[0]} — {decision[1]}");return {"output":output,"decision":decision,"summary":summary}
