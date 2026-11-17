"""Read-only pilot audits and fail-closed confirmatory freeze verification."""

from __future__ import annotations

from datetime import datetime
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .ecology import generate_world
from .models import build_model,model_hash


FROZEN_ARCHITECTURES=("A","B")
NEUTRAL_TRANSITIONS=(1,2)
NEUTRAL_MEAN_TOLERANCE=.05
NEUTRAL_WORLD_INTERVAL=(.4,.6)
NEUTRAL_REQUIRED_FRACTION=.95
CAUSAL_TOLERANCE=1e-7
SCIENTIFIC_SOURCE_FILES=(
    "configs/context_controlled_recurrent_architectures.json",
    "configs/context_controlled_recurrent_architectures_confirmatory.json",
    "context_controlled_recurrent_architectures/config.py",
    "context_controlled_recurrent_architectures/ecology.py",
    "context_controlled_recurrent_architectures/models.py",
    "context_controlled_recurrent_architectures/synthetic.py",
    "context_controlled_recurrent_architectures/analysis.py",
    "context_controlled_recurrent_architectures/plotting.py",
    "context_controlled_recurrent_architectures/preflight.py",
    "context_controlled_recurrent_architectures/pipeline.py",
    "scripts/run_context_controlled_preflight.py",
    "scripts/run_context_controlled_recurrent_architectures.py",
)
AUDIT_ARTIFACTS=(
    "synthetic_gate_results_pilot.csv",
    "architecture_selection_pilot.csv",
    "implementation_audit_neutral_history_raw.csv",
    "implementation_audit_neutral_history.csv",
    "implementation_audit_target_leakage.csv",
    "IMPLEMENTATION_PREFLIGHT_AUDIT.md",
)


def _sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def _true(series):
    return series.eq(True) if series.dtype==bool else series.astype(str).str.strip().str.lower().eq("true")


def load_pilot_selection(output):
    """Return the frozen pilot-selected architectures, never every passing model."""
    output=Path(output);gate_path=output/"synthetic_gate_results_pilot.csv";selection_path=output/"architecture_selection_pilot.csv"
    if not gate_path.exists(): raise FileNotFoundError("Confirmatory mode requires synthetic_gate_results_pilot.csv.")
    if not selection_path.exists(): raise FileNotFoundError("Run the implementation preflight to freeze architecture_selection_pilot.csv.")
    gates=pd.read_csv(gate_path);selection=pd.read_csv(selection_path)
    passing=set(gates.loc[_true(gates.gate_passed),"architecture"].astype(str))
    selected=selection.loc[_true(selection.selected_for_human_transfer)].copy()
    selected=selected.sort_values("selection_rank").architecture.astype(str).tolist()
    if not selected: raise RuntimeError("No pilot architecture was selected; confirmatory training is prohibited.")
    if not set(selected)<=passing: raise RuntimeError("Pilot selection contains an architecture that did not pass the pilot gates.")
    if tuple(selected)!=FROZEN_ARCHITECTURES: raise RuntimeError(f"The audited confirmatory architecture set must be {FROZEN_ARCHITECTURES}, observed {tuple(selected)}.")
    return selected


def _load_checkpoint(output,pilot_config,architecture,condition,seed):
    path=Path(output)/"checkpoints"/f"pilot_{architecture}_{condition}_seed_{seed}.pt"
    if not path.exists(): raise FileNotFoundError(path)
    payload=torch.load(path,map_location="cpu",weights_only=False)
    contract=(payload.get("architecture"),payload.get("condition"),payload.get("seed"),payload.get("worlds"),payload.get("phase"),payload.get("config_sha256"))
    expected=(architecture,condition,seed,pilot_config.worlds,"pilot",pilot_config.config_sha256)
    if contract!=expected: raise ValueError(f"Pilot checkpoint contract mismatch: {path}")
    model=build_model(architecture,seed);model.load_state_dict(payload["model_state_dict"]);model.eval()
    return path,model


def _neutral_history_audit(pilot_config,output,architectures):
    rows=[];offset=int(pilot_config.payload["synthetic"]["validation_seed_offset"])
    for architecture in architectures:
        for seed in pilot_config.seeds:
            _,model=_load_checkpoint(output,pilot_config,architecture,"structured",seed);before=model_hash(model)
            with torch.no_grad():
                for world_index in range(pilot_config.validation_worlds):
                    world=generate_world(seed+offset,world_index,pilot_config);regimes=np.asarray(world.evidence_regimes);causes=np.asarray(world.latent_causes);clear=np.flatnonzero(regimes!="ambiguous")
                    tokens=torch.stack([F.one_hot(torch.as_tensor(world.sequence_A if causes[i]=="A" else world.sequence_B)[:-1],4).float() for i in clear])
                    output_clear=model(tokens,torch.as_tensor(world.evidence[clear]));representations=(output_clear["context_state"] if output_clear["context_state"] is not None else output_clear["hidden"]).numpy()
                    clear_a=world.evidence[regimes=="clear_A"].mean(0);clear_b=world.evidence[regimes=="clear_B"].mean(0);midpoint=(clear_a+clear_b)/2
                    neutral_sequence=torch.as_tensor(world.shared_sequence);neutral_tokens=F.one_hot(neutral_sequence[:-1],4).float().unsqueeze(0)
                    output_neutral=model(neutral_tokens,torch.as_tensor(midpoint).unsqueeze(0));neutral_representation=(output_neutral["context_state"] if output_neutral["context_state"] is not None else output_neutral["hidden"])[0].numpy()
                    oracle=world.oracle_p_A[clear].astype(float)
                    for transition in NEUTRAL_TRANSITIONS:
                        x=representations[:,transition-1];xm=x.mean(0);xs=x.std(0);xs[xs<1e-8]=1.;z=(x-xm)/xs;centered=oracle-oracle.mean()
                        beta=z.T@np.linalg.solve(z@z.T+1e-3*np.eye(len(z)),centered)
                        decoded=float(np.clip(oracle.mean()+((neutral_representation[transition-1]-xm)/xs)@beta,0,1))
                        rows.append({"architecture":architecture,"seed":seed,"validation_world":world_index,"transition":transition,"decoded_p_A":decoded,"reset_state":True,"midpoint_evidence":True,"diagnostic_task_history":False})
            if model_hash(model)!=before: raise AssertionError("Neutral-history audit mutated a pilot checkpoint model.")
    raw=pd.DataFrame(rows)
    summary=raw.groupby(["architecture","seed","transition"],as_index=False).agg(worlds=("validation_world","nunique"),mean_decoded_p_A=("decoded_p_A","mean"),median_decoded_p_A=("decoded_p_A","median"),mean_absolute_deviation_from_half=("decoded_p_A",lambda x:float(np.mean(np.abs(x-.5)))),fraction_within_040_060=("decoded_p_A",lambda x:float(np.mean((x>=NEUTRAL_WORLD_INTERVAL[0])&(x<=NEUTRAL_WORLD_INTERVAL[1])))))
    summary["mean_centered_pass"]=(summary.mean_decoded_p_A-.5).abs()<=NEUTRAL_MEAN_TOLERANCE
    summary["world_fraction_pass"]=summary.fraction_within_040_060>=NEUTRAL_REQUIRED_FRACTION
    summary["audit_passed"]=summary.mean_centered_pass&summary.world_fraction_pass
    return raw,summary


def _target_leakage_audit(pilot_config,output,architectures):
    rows=[];offset=int(pilot_config.payload["synthetic"]["validation_seed_offset"])
    for architecture in architectures:
        for condition in ("structured","shuffled"):
            for seed in pilot_config.seeds:
                _,model=_load_checkpoint(output,pilot_config,architecture,condition,seed);before=model_hash(model)
                with torch.no_grad():
                    world=generate_world(seed+offset,0,pilot_config);sequence=torch.as_tensor(world.sequence_A);tokens=F.one_hot(sequence[:-1],4).float().unsqueeze(0);evidence=torch.as_tensor(world.evidence[0]).unsqueeze(0);cutoff=2
                    changed=tokens.clone();changed[:,cutoff+1:]=torch.roll(changed[:,cutoff+1:],1,dims=-1)
                    original_logits=model(tokens,evidence)["logits"];changed_logits=model(changed,evidence)["logits"]
                    prefix_effect=float((original_logits[:,:cutoff+1]-changed_logits[:,:cutoff+1]).abs().max());future_effect=float((original_logits[:,cutoff+1:]-changed_logits[:,cutoff+1:]).abs().max())
                unchanged=model_hash(model)==before;forward_has_target_argument="target" in inspect.signature(model.forward).parameters
                passed=prefix_effect<=CAUSAL_TOLERANCE and future_effect>CAUSAL_TOLERANCE and unchanged and not forward_has_target_argument
                rows.append({"architecture":architecture,"structure_condition":condition,"seed":seed,"changed_after_transition":cutoff+1,"max_preceding_logit_effect":prefix_effect,"max_future_logit_effect":future_effect,"forward_has_target_argument":forward_has_target_argument,"parameters_unchanged":unchanged,"audit_passed":passed})
    return pd.DataFrame(rows)


def _write_report(output,neutral,target,passed):
    grouped=neutral.groupby("architecture",as_index=False).agg(mean_decoded_p_A=("mean_decoded_p_A","mean"),minimum_fraction_within_040_060=("fraction_within_040_060","min"),all_seed_transition_checks_passed=("audit_passed","all"))
    target_summary=target.groupby("architecture",as_index=False).agg(checkpoints=("seed","size"),maximum_preceding_logit_effect=("max_preceding_logit_effect","max"),minimum_future_logit_effect=("max_future_logit_effect","min"),all_checks_passed=("audit_passed","all"))
    text=f"""# Implementation Preflight Audit

- Overall result: **{'PASS' if passed else 'FAIL'}**
- Architectures: A and B only
- Neutral-history definition: reset state, midpoint evidence, and transitions 1–2 before task history differentiates A from B
- Neutral criterion: mean decoded P(A) within {NEUTRAL_MEAN_TOLERANCE:.2f} of 0.5 and at least {NEUTRAL_REQUIRED_FRACTION:.0%} of worlds in [{NEUTRAL_WORLD_INTERVAL[0]:.1f}, {NEUTRAL_WORLD_INTERVAL[1]:.1f}], evaluated separately for every architecture/seed/transition
- Leakage criterion: changing future observable tokens has at most {CAUSAL_TOLERANCE:g} effect on preceding logits, changes future logits, exposes no target argument to `forward`, and mutates no parameters
- Optimizer updates: 0
- Human data accessed: no

## Neutral-history uncertainty

{grouped.to_markdown(index=False)}

## Target-information leakage

{target_summary.to_markdown(index=False)}

These checks are audits of frozen pilot checkpoints. They do not alter architecture, ecology, parameters, or gate thresholds.
"""
    (Path(output)/"IMPLEMENTATION_PREFLIGHT_AUDIT.md").write_text(text)


def _build_freeze_manifest(confirmatory_config,output,architectures,audit_passed):
    output=Path(output);repo=confirmatory_config.repo_root
    sources={name:_sha256(repo/name) for name in SCIENTIFIC_SOURCE_FILES}
    artifacts={name:_sha256(output/name) for name in AUDIT_ARTIFACTS}
    checkpoints={}
    for architecture in architectures:
        for condition in ("structured","shuffled"):
            for seed in range(5):
                name=f"checkpoints/pilot_{architecture}_{condition}_seed_{seed}.pt";checkpoints[name]=_sha256(output/name)
    manifest={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),"status":"frozen_after_passing_implementation_audits","audit_passed":bool(audit_passed),"selected_architectures":list(architectures),"confirmatory_seeds":list(confirmatory_config.seeds),"confirmatory_worlds_per_cell":confirmatory_config.worlds,"confirmatory_validation_worlds":confirmatory_config.validation_worlds,"confirmatory_config_sha256":confirmatory_config.config_sha256,"scientific_source_sha256":sources,"audit_artifact_sha256":artifacts,"audited_pilot_checkpoint_sha256":checkpoints,"human_firewall_default":"closed"}
    path=output/"confirmatory_freeze_manifest.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");return manifest


def verify_freeze_manifest(config,output):
    path=Path(output)/"confirmatory_freeze_manifest.json"
    if not path.exists(): raise FileNotFoundError("Confirmatory mode requires a passing preflight and confirmatory_freeze_manifest.json.")
    manifest=json.loads(path.read_text())
    if not manifest.get("audit_passed"): raise RuntimeError("The frozen implementation preflight did not pass.")
    if config.config_sha256!=manifest.get("confirmatory_config_sha256"): raise RuntimeError("Confirmatory configuration changed after the freeze.")
    if list(config.seeds)!=manifest.get("confirmatory_seeds"): raise RuntimeError("Confirmatory seed registry changed after the freeze.")
    for name,expected in manifest["scientific_source_sha256"].items():
        if _sha256(config.repo_root/name)!=expected: raise RuntimeError(f"Frozen scientific source changed: {name}")
    for name,expected in manifest["audit_artifact_sha256"].items():
        if _sha256(Path(output)/name)!=expected: raise RuntimeError(f"Frozen audit artifact changed: {name}")
    for name,expected in manifest["audited_pilot_checkpoint_sha256"].items():
        if _sha256(Path(output)/name)!=expected: raise RuntimeError(f"Audited pilot checkpoint changed: {name}")
    return manifest


def run_preflight(pilot_config,confirmatory_config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    pilot_selection=output/"architecture_selection_pilot.csv"
    if not pilot_selection.exists():
        current=output/"architecture_selection.csv"
        if not current.exists(): raise FileNotFoundError("The completed pilot architecture_selection.csv is missing.")
        pd.read_csv(current).to_csv(pilot_selection,index=False)
    architectures=load_pilot_selection(output)
    if set(pilot_config.seeds)&set(confirmatory_config.seeds): raise RuntimeError("Pilot and confirmatory seed registries overlap.")
    raw,neutral=_neutral_history_audit(pilot_config,output,architectures);target=_target_leakage_audit(pilot_config,output,architectures)
    raw.to_csv(output/"implementation_audit_neutral_history_raw.csv",index=False);neutral.to_csv(output/"implementation_audit_neutral_history.csv",index=False);target.to_csv(output/"implementation_audit_target_leakage.csv",index=False)
    passed=bool(neutral.audit_passed.all() and target.audit_passed.all());_write_report(output,neutral,target,passed)
    if not passed: raise RuntimeError("Implementation preflight failed; no freeze manifest was created.")
    return _build_freeze_manifest(confirmatory_config,output,architectures,passed)
