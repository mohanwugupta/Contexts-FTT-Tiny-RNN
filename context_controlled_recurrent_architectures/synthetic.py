"""Prediction-only synthetic training and immutable validation battery."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .ecology import generate_world, manifest_record, world_batch
from .models import build_model, model_hash


CHECKPOINT_INTERVAL=2500


@dataclass
class SyntheticRun:
    architecture: str
    condition: str
    seed: int
    phase: str
    model: object
    training: pd.DataFrame
    manifest: pd.DataFrame
    audit: dict


def _score(model,batch):
    tokens,evidence,targets=batch; output=model(tokens,evidence); logits=output["logits"]
    loss=F.cross_entropy(logits.reshape(-1,4),targets.reshape(-1)); accuracy=logits.argmax(-1).eq(targets).float().mean()
    confidence=logits.softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1).mean()
    return loss,accuracy,confidence,output


def _atomic_torch_save(payload,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload,temporary);os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _record_directory(checkpoint_path):
    checkpoint_path=Path(checkpoint_path);return checkpoint_path.parent/"records"/checkpoint_path.stem


def _atomic_csv(frame,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary,index=False);os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _write_record_chunk(checkpoint_path,kind,records,start,end):
    if end<=start:return
    path=_record_directory(checkpoint_path)/f"{kind}_{start:06d}_{end:06d}.csv"
    _atomic_csv(pd.DataFrame(records[start:end]),path)


def _chunk_records(checkpoint_path,kind,completed):
    directory=_record_directory(checkpoint_path);frames=[]
    for path in sorted(directory.glob(f"{kind}_*.csv")):
        frame=pd.read_csv(path)
        if "world_index" in frame:frame=frame[pd.to_numeric(frame.world_index,errors="coerce")<completed]
        if len(frame):frames.append(frame)
    if not frames:return []
    frame=pd.concat(frames,ignore_index=True);frame["world_index"]=pd.to_numeric(frame.world_index).astype(int)
    frame=frame.drop_duplicates("world_index",keep="last").sort_values("world_index")
    return frame.to_dict("records")


def _persist_legacy_records(checkpoint_path,rows,manifests,completed):
    directory=_record_directory(checkpoint_path)
    for start in range(0,completed,CHECKPOINT_INTERVAL):
        end=min(completed,start+CHECKPOINT_INTERVAL)
        if not (directory/f"training_{start:06d}_{end:06d}.csv").exists():_write_record_chunk(checkpoint_path,"training",rows,start,end)
        if not (directory/f"manifest_{start:06d}_{end:06d}.csv").exists():_write_record_chunk(checkpoint_path,"manifest",manifests,start,end)


def _checkpoint_contract(payload,config,architecture,condition,seed,worlds,phase,checkpoint_path):
    contract=(payload.get("architecture"),payload.get("condition"),payload.get("seed"),payload.get("worlds"),payload.get("phase"),payload.get("config_sha256"))
    expected=(architecture,condition,seed,worlds,phase,config.config_sha256)
    if contract!=expected:raise ValueError(f"Synthetic checkpoint contract mismatch: {checkpoint_path}")


def train_run(config,architecture,condition,seed,worlds,phase,progress=None,checkpoint_path=None):
    model=build_model(architecture,seed); initial_hash=model_hash(model); settings=config.payload["synthetic"]
    optimizer=torch.optim.Adam(model.parameters(),lr=float(settings["outer_learning_rate"])); rows=[];manifests=[];pulse=max(1,worlds//20);start=0;persisted=0
    if checkpoint_path is not None and checkpoint_path.exists():
        payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
        _checkpoint_contract(payload,config,architecture,condition,seed,worlds,phase,checkpoint_path)
        model.load_state_dict(payload["model_state_dict"]);optimizer.load_state_dict(payload["optimizer_state_dict"]);start=int(payload["next_world"])
        rows=list(payload.get("training_rows",[])) or _chunk_records(checkpoint_path,"training",start);manifests=list(payload.get("manifest_rows",[])) or _chunk_records(checkpoint_path,"manifest",start)
        if len(rows)!=start or len(manifests)!=start:raise RuntimeError(f"Checkpoint records are incomplete for {checkpoint_path}: expected {start}, observed training={len(rows)}, manifest={len(manifests)}")
        if payload.get("checkpoint_schema")!=2:
            _persist_legacy_records(checkpoint_path,rows,manifests,start)
            if start==worlds:
                _atomic_torch_save({"checkpoint_schema":2,"architecture":architecture,"condition":condition,"seed":seed,"worlds":worlds,"phase":phase,"config_sha256":config.config_sha256,"next_world":start,
                            "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"initial_state_sha256":initial_hash},checkpoint_path)
        persisted=start
        if progress: progress(f"resumed {phase} {architecture}_{condition} seed {seed} at world {start}/{worlds}")
    checkpoint_pulse=max(1,min(CHECKPOINT_INTERVAL,pulse))
    for index in range(start,worlds):
        world=generate_world(seed,index,config); batch=world_batch(world,condition); model.train(); optimizer.zero_grad(); loss,accuracy,confidence,_=_score(model,batch)
        loss.backward(); gradient=float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)))
        torch.nn.utils.clip_grad_norm_(model.parameters(),float(settings["gradient_clip"])); optimizer.step()
        rows.append({"phase":phase,"architecture":architecture,"structure_condition":condition,"seed":seed,"world_index":index,"world_sha256":world.world_sha256,
                     "prediction_NLL":float(loss.detach()),"token_accuracy":float(accuracy.detach()),"correct_token_confidence":float(confidence.detach()),
                     "outer_gradient_norm":gradient,"optimizer_updates":index+1,"RT_loss":0.0,"context_label_loss":0.0,"prediction_only_objective":True})
        manifests.append(manifest_record(world,condition,phase)|{"architecture":architecture,"seed":seed})
        if checkpoint_path is not None and ((index+1)%checkpoint_pulse==0 or index+1==worlds):
            end=index+1;_write_record_chunk(checkpoint_path,"training",rows,persisted,end);_write_record_chunk(checkpoint_path,"manifest",manifests,persisted,end)
            _atomic_torch_save({"checkpoint_schema":2,"architecture":architecture,"condition":condition,"seed":seed,"worlds":worlds,"phase":phase,"config_sha256":config.config_sha256,"next_world":end,
                        "model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"initial_state_sha256":initial_hash},checkpoint_path);persisted=end
        if progress and ((index+1)%pulse==0 or index+1==worlds): progress(f"{phase} {architecture}_{condition} seed {seed}: world {index+1}/{worlds}")
    return SyntheticRun(architecture,condition,seed,phase,model,pd.DataFrame(rows),pd.DataFrame(manifests),
                        {"phase":phase,"architecture":architecture,"structure_condition":condition,"seed":seed,"initial_state_sha256":initial_hash,
                         "final_state_sha256":model_hash(model),"optimizer_updates":worlds,"RT_supervision":False,"latent_label_input":False,"mixed_class_training_rows":0})


def load_completed_run(config,architecture,condition,seed,worlds,phase,checkpoint_path,include_records=False):
    checkpoint_path=Path(checkpoint_path);payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False);_checkpoint_contract(payload,config,architecture,condition,seed,worlds,phase,checkpoint_path)
    if int(payload["next_world"])!=worlds:raise RuntimeError(f"Training is incomplete for {checkpoint_path}: {payload['next_world']}/{worlds}")
    model=build_model(architecture,seed);initial_hash=model_hash(model);model.load_state_dict(payload["model_state_dict"])
    rows=_chunk_records(checkpoint_path,"training",worlds) if include_records else [];manifests=_chunk_records(checkpoint_path,"manifest",worlds) if include_records else []
    return SyntheticRun(architecture,condition,seed,phase,model,pd.DataFrame(rows),pd.DataFrame(manifests),{"phase":phase,"architecture":architecture,"structure_condition":condition,"seed":seed,"initial_state_sha256":payload.get("initial_state_sha256",initial_hash),"final_state_sha256":model_hash(model),"optimizer_updates":worlds,"RT_supervision":False,"latent_label_input":False,"mixed_class_training_rows":0})


def record_chunks(checkpoint_path,kind):
    directory=_record_directory(checkpoint_path)
    for path in sorted(directory.glob(f"{kind}_*.csv")):yield pd.read_csv(path)


def _cosine(a,b):
    denominator=np.linalg.norm(a)*np.linalg.norm(b); return float(a.dot(b)/denominator) if denominator>1e-12 else np.nan


def _sequence_metrics(output,targets):
    logits=output["logits"][0].reshape(36,7,4); target=targets[0].reshape(36,7); hidden=output["hidden"][0].reshape(36,7,-1)
    probability=logits.softmax(-1); confidence=probability.gather(-1,target.unsqueeze(-1)).squeeze(-1)
    nll=F.cross_entropy(logits.reshape(-1,4),target.reshape(-1),reduction="none").reshape(36,7)
    accuracy=logits.argmax(-1).eq(target)
    context=output["context_state"][0].reshape(36,7,-1) if output["context_state"] is not None else None
    gate=output["gate"][0].reshape(36,7,-1) if output["gate"] is not None else None
    return logits.detach(),hidden.detach(),context.detach() if context is not None else None,gate.detach() if gate is not None else None,nll.detach(),accuracy.detach(),confidence.detach()


def _split_shared_rows(run,world,output,targets,validation_index):
    logits,hidden,context,gate,nll,accuracy,confidence=_sequence_metrics(output,targets); causes=np.asarray(world.latent_causes); regimes=np.asarray(world.evidence_regimes)
    clear_a=np.where((causes=="A")&(regimes!="ambiguous"))[0]; clear_b=np.where((causes=="B")&(regimes!="ambiguous"))[0]; ambiguous=np.where(regimes=="ambiguous")[0]
    ha=hidden[clear_a].mean(0).numpy(); hb=hidden[clear_b].mean(0).numpy(); pa=logits[clear_a].softmax(-1).mean(0).numpy(); pb=logits[clear_b].softmax(-1).mean(0).numpy()
    split={"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"validation_world":validation_index,
           "AB_hidden_distance":float(np.linalg.norm(ha-hb,axis=1).mean()),"AB_prediction_profile_distance":float(np.linalg.norm(pa-pb,axis=1).mean()),
           "AB_context_state_distance":float(np.linalg.norm(context[clear_a].mean(0).numpy()-context[clear_b].mean(0).numpy(),axis=1).mean()) if context is not None else np.nan,
           "AB_gate_distance":float(np.linalg.norm(gate[clear_a].mean(0).numpy()-gate[clear_b].mean(0).numpy(),axis=1).mean()) if gate is not None else np.nan,
           "clear_accuracy":float(accuracy[np.r_[clear_a,clear_b]].float().mean()),"clear_NLL":float(nll[np.r_[clear_a,clear_b]].mean()),"numerically_stable":bool(torch.isfinite(logits).all())}
    shared_positions=[0,3,6]; cos=[_cosine(ha[position],hb[position]) for position in shared_positions]
    stack=np.vstack([ha,hb]); singular=np.linalg.svd(stack-stack.mean(0),compute_uv=False); variance=singular**2
    a0=ha-ha.mean(0); b0=hb-hb.mean(0); u,_,vt=np.linalg.svd(a0.T@b0,full_matrices=False); aligned=a0@(u@vt)
    shared={"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"validation_world":validation_index,
            "shared_token_hidden_cosine":float(np.nanmean(cos)),"shared_subspace_variance_explained_rank8":float(variance[:8].sum()/max(variance.sum(),1e-12)),
            "procrustes_alignment_RMSE":float(np.sqrt(np.mean((aligned-b0)**2))),"shared_transition_count":3}
    ambiguity={"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"validation_world":validation_index,
               "ambiguous_query_NLL":float(nll[ambiguous].mean()),"ambiguous_query_accuracy":float(accuracy[ambiguous].float().mean()),
               "ambiguous_correct_token_confidence":float(confidence[ambiguous].mean()),"clear_AB_hidden_distance_retained":split["AB_hidden_distance"]}
    representation=context[:,-1] if context is not None else hidden[:,-1]
    belief=[{"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"validation_world":validation_index,
             "sequence_in_world":i+1,"evidence_regime":world.evidence_regimes[i],"oracle_p_A":float(world.oracle_p_A[i]),"true_latent_cause":world.latent_causes[i],
             **{f"representation_{j+1}":float(value) for j,value in enumerate(representation[i])}} for i in range(36)]
    return split,shared,ambiguity,belief


def _custom_stream(model,world,causes,surfaces):
    tokens=[];targets=[];evidence=[]
    for cause,surface in zip(causes,surfaces):
        sequence=torch.as_tensor(world.sequence_A if cause=="A" else world.sequence_B); tokens.append(F.one_hot(sequence[:-1],4).float()); targets.append(sequence[1:])
        evidence.append(torch.as_tensor(surface,dtype=torch.float32))
    batch=(torch.cat(tokens).unsqueeze(0),torch.cat(evidence).unsqueeze(0),torch.cat(targets).unsqueeze(0)); return batch,model(batch[0],batch[1])


def _switch_rows(run,world):
    clear_a=world.evidence[np.asarray(world.evidence_regimes)=="clear_A"].mean(0); clear_b=world.evidence[np.asarray(world.evidence_regimes)=="clear_B"].mean(0); rows=[]
    for direction,causes in (("A_to_B",list("AAAABBBB")),("B_to_A",list("BBBBAAAA"))):
        surfaces=[clear_a if cause=="A" else clear_b for cause in causes]; batch,output=_custom_stream(run.model,world,causes,surfaces); logits=output["logits"][0].reshape(8,7,4); target=batch[2][0].reshape(8,7)
        accuracies=logits.argmax(-1).eq(target).float().mean(1).detach().numpy(); baseline=float(accuracies[:4].mean()); recovery=5
        for offset,value in enumerate(accuracies[4:],1):
            if value>=baseline-.1: recovery=offset; break
        representation=(output["context_state"] if output["context_state"] is not None else output["hidden"])[0].reshape(8,7,-1)[:,-1].detach().numpy()
        old=representation[:4].mean(0); new=representation[6:].mean(0); latency=5
        for offset,value in enumerate(representation[4:],1):
            if np.linalg.norm(value-new)<np.linalg.norm(value-old): latency=offset; break
        gate_latency=np.nan
        if output["gate"] is not None:
            values=output["gate"][0].reshape(8,7,-1)[:,-1].detach().numpy(); old_g=values[:4].mean(0); new_g=values[6:].mean(0); gate_latency=5
            for offset,value in enumerate(values[4:],1):
                if np.linalg.norm(value-new_g)<np.linalg.norm(value-old_g): gate_latency=offset; break
        rows.append({"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"switch_direction":direction,
                     "prediction_recovery_sequences":recovery,"context_transition_latency":latency,"gate_transition_latency":gate_latency,
                     "pre_switch_accuracy":baseline,"post_switch_accuracy":float(accuracies[4:].mean())})
    return rows


def _continuum_rows(run,world,reliabilities):
    clear_a=world.evidence[np.asarray(world.evidence_regimes)=="clear_A"].mean(0); clear_b=world.evidence[np.asarray(world.evidence_regimes)=="clear_B"].mean(0); midpoint=(clear_a+clear_b)/2; rows=[]
    for reliability in reliabilities:
        for cause,endpoint in (("A",clear_a),("B",clear_b)):
            surface=midpoint+float(reliability)*(endpoint-midpoint); batch,output=_custom_stream(run.model,world,[cause],[surface]); logits=output["logits"]; target=batch[2]
            representation=(output["context_state"] if output["context_state"] is not None else output["hidden"])[0,-1].detach().numpy(); gate=output["gate"][0,-1].detach().numpy() if output["gate"] is not None else None
            rows.append({"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"reliability":float(reliability),"evidence_lean":cause,
                         "prediction_NLL":float(F.cross_entropy(logits.reshape(-1,4),target.reshape(-1)).detach()),"token_accuracy":float(logits.argmax(-1).eq(target).float().mean()),
                         "representation_norm":float(np.linalg.norm(representation)),"representation_mean":float(representation.mean()),
                         "gate_mean":float(gate.mean()) if gate is not None else np.nan,"gate_entropy":float(-(np.clip(gate,1e-8,1-1e-8)*np.log(np.clip(gate,1e-8,1-1e-8))+(1-np.clip(gate,1e-8,1-1e-8))*np.log(1-np.clip(gate,1e-8,1-1e-8))).mean()) if gate is not None else np.nan,
                         **{f"representation_{j+1}":float(value) for j,value in enumerate(representation)}})
    return rows


def _conflict_rows(run,world):
    a=world.evidence[np.asarray(world.evidence_regimes)=="clear_A"].mean(0); b=world.evidence[np.asarray(world.evidence_regimes)=="clear_B"].mean(0); rows=[]
    for surface_cause,task_cause,surface in (("A","B",a),("B","A",b)):
        batch,output=_custom_stream(run.model,world,[task_cause],[surface]); rows.append({"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,
            "surface_evidence":surface_cause,"task_history_regime":task_cause,"conflicting":True,"prediction_NLL":float(F.cross_entropy(output["logits"].reshape(-1,4),batch[2].reshape(-1)).detach()),
            "token_accuracy":float(output["logits"].argmax(-1).eq(batch[2]).float().mean())})
    return rows


def validate_runs(config,runs,worlds,progress=None):
    split=[];shared=[];ambiguity=[];belief=[];switch=[];continuum=[];conflict=[];audit=[]
    for run in runs:
        before=model_hash(run.model); run.model.eval()
        with torch.no_grad():
            for index in range(worlds):
                world=generate_world(run.seed+int(config.payload["synthetic"]["validation_seed_offset"]),index,config); batch=world_batch(world,run.condition); output=run.model(batch[0],batch[1])
                a,b,c,d=_split_shared_rows(run,world,output,batch[2],index); split.append(a);shared.append(b);ambiguity.append(c);belief.extend(d)
                if index==0:
                    switch.extend(_switch_rows(run,world)); continuum.extend(_continuum_rows(run,world,config.payload["synthetic"]["reliability_grid"])); conflict.extend(_conflict_rows(run,world))
        after=model_hash(run.model); audit.append({"phase":run.phase,"architecture":run.architecture,"structure_condition":run.condition,"seed":run.seed,"hash_before":before,"hash_after":after,"parameters_unchanged":before==after,"optimizer_updates":0})
        if before!=after: raise AssertionError("T8: frozen synthetic validation mutated parameters.")
        if progress: progress(f"validated {run.phase} {run.architecture}_{run.condition} seed {run.seed}")
    return {"split":pd.DataFrame(split),"shared":pd.DataFrame(shared),"ambiguity":pd.DataFrame(ambiguity),"belief_raw":pd.DataFrame(belief),"switch":pd.DataFrame(switch),
            "continuum":pd.DataFrame(continuum),"conflict":pd.DataFrame(conflict),"audit":pd.DataFrame(audit)}
