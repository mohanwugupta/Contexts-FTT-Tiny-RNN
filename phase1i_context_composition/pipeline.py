from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib,json
from pathlib import Path
import subprocess

import pandas as pd

from srn_training_timeline_audit.core import sha256_file

from .analysis import bootstrap_contrasts,build_seed_summary,contrast_values,decision
from .config import ExperimentConfig
from .cue_encoding import PROBES,assert_coefficient_firewall,cue_audit_table,cue_coefficients
from .plotting import enforce_figures,make_figures
from .replay import replay_seed


def _stamp():return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
def _log(message):print(f"[{_stamp()}] PHASE1I | {message}",flush=True)
def _source_hash():
    digest=hashlib.sha256()
    for name in ("config.py","cue_encoding.py","mechanics.py","replay.py","analysis.py","plotting.py","pipeline.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()
def _git_commit(repo):
    try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,stderr=subprocess.DEVNULL,text=True).strip()
    except Exception:return "unavailable:not-a-git-worktree"


def _replay_or_load(config,output,family,seed):
    partial=output/"partials";partial.mkdir(parents=True,exist_ok=True);stem=f"{family}_seed_{seed}"
    names=("timeline","transition","acquisition","functional")
    paths={name:partial/f"{stem}_{name}.csv" for name in names};meta=partial/f"{stem}_metadata.json"
    if all(path.exists() for path in paths.values()) and meta.exists():
        payload=json.loads(meta.read_text())
        if payload.get("config_sha256")!=config.config_sha256 or payload.get("source_sha256")!=_source_hash():
            raise RuntimeError(f"Partial Phase 1I replay uses another frozen specification: {stem}")
        _log(f"resuming completed {family} seed {seed}")
        frames=[pd.read_csv(paths[name]) for name in names]
        return (*frames,payload["gradient"],payload["provenance"])
    timeline,transition,acquisition,functional,gradient,provenance=replay_seed(config,family,seed,progress=_log)
    for name,frame in zip(names,(timeline,transition,acquisition,functional)):frame.to_csv(paths[name],index=False)
    meta.write_text(json.dumps({"config_sha256":config.config_sha256,"source_sha256":_source_hash(),
                                "gradient":gradient,"provenance":provenance},indent=2,sort_keys=True))
    return timeline,transition,acquisition,functional,gradient,provenance


def _write_report(output,summary,statistics,cue,primary,secondary):
    code,label=primary;scode,slabel=secondary;means=summary.groupby("family").mean(numeric_only=True)
    text=f"""# Phase 1I Context Composition Report

## 1. Executive summary

**Primary Outcome {code} — {label}.** **Novel/NoContext Outcome {scode} — {slabel}.**

## 2. Cue encoding audit

The encoding-only solution is `alpha_A={cue.alpha_A.iloc[0]:.6g}`, `alpha_B={cue.alpha_B.iloc[0]:.6g}` with
cue residual `{cue.AB_span_residual.iloc[0]:.3g}`. Coefficients were computed before behavioral evaluation
by a function whose signature accepts cue vectors only. Maximum pre-tanh preactivation residual was
`{cue.preactivation_composition_residual_max.max():.3g}`.

## 3. Matched-state one-step composition

Every branch uses a bit-identical prestate. Hidden and centered-logit residuals are normalized by the
same-state A/B branch distance, with unnormalized values retained in `phase1i_transition_metrics.csv`.

## 4. Free recurrent composition

Natural A, B, zero, Mixed, and all four Novel trajectories start from the same zero state. The cue-derived
target uses only A/B/zero and the encoding coefficients; no coefficient is fit to Mixed behavior.

## 5. Causal Mixed rescue

Correct trajectory, wrong-position derangement, seed-specific norm-matched orthogonal, A endpoint, B endpoint,
no-patch, and diagnostic midpoint interventions use fixed strength 1.0. Seven single-position acute variants
are also exported with next-token, remaining-sequence, and full-sequence raw CE.

## 6. Novel/NoContext synthetic composite rescue

The identical familiar A+B/zero target is applied without refitting to NoContext and every preregistered Novel
cue. NovelMean includes all four variants, never a selected best cue.

## 7. Forward composition versus backward credit assignment

Final A/B gradient cosine and actual within/cross update effects are calculated from the canonical replay and
joined to forward residual and causal-rescue metrics only after all targets are frozen.

## 8. Raw acquisition and probe-loss time series

FIG01 uses actual pre-update raw mean CE with changing parameter hashes. FIG02 evaluates genuine registered
replay states; FIG04 preserves every external update index. Sequence loss is always mean CE across exactly
seven predicted transitions.

## 9. Controls

Wrong-position targets are a fixed derangement. Orthogonal movements are deterministic, direction-orthogonal,
and norm matched. The midpoint is explicitly secondary because cue-derived composition includes the neutral
baseline and is not assumed equal to a hidden-state midpoint.

## 10. Per-seed heterogeneity

```text
{means.to_string()}
```

Bootstrap intervals in `phase1i_statistical_summary.csv` resample seeds, not transitions or checkpoints.

## 11. Decision-table outcome

**{code}: {label}.** **{scode}: {slabel}.** These are mechanical classifications, not significance-only calls.

## 12. Limitations

Composition is evaluated in the existing canonical SRN and cannot establish that humans use the same
transformation. Matched-state tests are local, causal patches establish state sufficiency rather than natural
generation, and gradient alignment is a local property.

## 13. Implications for the human averaging account

Only Outcome A supports the full claim that familiar cue transformations compose naturally while backward
credit remains insulated. Outcome B localizes failure to recurrence; Outcome C supports state sufficiency but
not natural construction; Outcome D rejects this mechanism for the current SRN.
"""
    (output/"PHASE1I_CONTEXT_COMPOSITION_REPORT.md").write_text(text)


def run_pipeline(config:ExperimentConfig):
    config.validate();assert_coefficient_firewall();alpha_a,alpha_b,cue_residual=cue_coefficients(config.composition_tolerance)
    output=config.resolve(config.output_dir);output.mkdir(parents=True,exist_ok=True);repo=Path(config.repo_dir).resolve()
    _log(f"starting {config.profile} read-only composition audit")
    timelines=[];transitions=[];acquisitions=[];functional=[];gradients=[];provenance=[]
    for family in config.families:
        for seed in config.seeds:
            _log(f"replay {family} seed {seed}")
            timeline,transition,acq,func,gradient,prov=_replay_or_load(config,output,family,seed)
            timelines.append(timeline);transitions.append(transition);acquisitions.append(acq);functional.append(func)
            gradients.append(gradient);provenance.append(prov)
    timeline=pd.concat(timelines,ignore_index=True);transition=pd.concat(transitions,ignore_index=True)
    acquisition=pd.concat(acquisitions,ignore_index=True);functional=pd.concat(functional,ignore_index=True)
    gradients=pd.DataFrame(gradients);provenance=pd.DataFrame(provenance);git_commit=_git_commit(repo)
    for frame in (timeline,transition,acquisition,functional,gradients,provenance):frame["git_commit"]=git_commit
    preactivation_max=float(provenance.preactivation_composition_residual_max.max())
    if preactivation_max>=config.composition_tolerance:raise AssertionError("FAIL: PREACTIVATION_COMPOSITION")
    cue=cue_audit_table(preactivation_max)
    if not cue.is_exact_AB_composition.all():raise AssertionError("FAIL: MIXED_NOT_IN_FAMILIAR_CUE_SPAN")
    if set(transition[transition.probe.str.startswith("Novel")].probe.unique())!=set(PROBES[2:]):
        raise AssertionError("FAIL: NOVEL_VARIANTS_INCOMPLETE")
    summary=build_seed_summary(transition,timeline,functional,gradients,provenance,config)
    values=contrast_values(summary);statistics=bootstrap_contrasts(values,config)
    primary,secondary=decision(summary,bool(cue.is_exact_AB_composition.all()))
    tables={"phase1i_seed_summary.csv":summary,"phase1i_transition_metrics.csv":transition,
            "phase1i_cue_encoding.csv":cue,"phase1i_timeline_metrics.csv":timeline,
            "phase1i_true_acquisition.csv":acquisition,"phase1i_functional_transfer.csv":functional,
            "phase1i_gradient_bridge.csv":gradients,"phase1i_checkpoint_provenance.csv":provenance,
            "phase1i_seed_contrasts.csv":values,"phase1i_statistical_summary.csv":statistics}
    for name,frame in tables.items():frame.to_csv(output/name,index=False)
    if config.make_plots:make_figures(output,acquisition,timeline,transition,summary,config);enforce_figures(output)
    _write_report(output,summary,statistics,cue,primary,secondary)
    manifest={"phase":"Phase1I","profile":config.profile,"created_at":_stamp(),"read_only_sources":True,
              "source_artifacts_modified":False,"config":asdict(config),"config_sha256":config.config_sha256,
              "source_sha256":_source_hash(),"source_config_sha256":sha256_file(config.resolve(config.source_config)),
              "git_commit":git_commit,"alpha_A":alpha_a,"alpha_B":alpha_b,"cue_residual":cue_residual,
              "coefficient_source":"cue_encoding_only","probe_training_rows":0,"patch_strength":1.0,
              "checkpoint_selection":"registered canonical replay states only",
              "all_replays_exact":bool(provenance.replay_matches_final_checkpoint.all()),
              "novel_variants":list(PROBES[2:]),"primary_decision":{"code":primary[0],"label":primary[1]},
              "novel_nocontext_decision":{"code":secondary[0],"label":secondary[1]}}
    (output/"PHASE1I_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True,default=str))
    _log(f"complete: Outcome {primary[0]} / {secondary[0]}");return {"output":output,"primary":primary,"secondary":secondary,"summary":summary}
