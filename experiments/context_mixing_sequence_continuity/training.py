"""D3 only: source-objective continuation, then a sealed held-out evaluation."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from confirmatory_srn.config import ConfirmatoryConfig
from confirmatory_srn.training import state_hash
from execution_objective_search.objectives import execution_loss,predictive_statistics
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from context_sequence.provenance.hashing import sha256,object_hash
from .config import CONTRACT,ROOT,SEQUENCE
from .curriculum import schedule,validate_schedule
from .source_adapter import records,load_torch,frozen_from_state,cues
from .replay import natural_trace,separation
from .statistics import seed_bootstrap,novel_means
from .outcomes import paired_effects
from .io import dump,save,report,table,codes


def _parameters(model,policy):
    names=[*dict(model.named_parameters()),*['policy.'+n for n,_ in policy.named_parameters()]]
    expected=['recurrent.weight_ih_l0','recurrent.weight_hh_l0','recurrent.bias_ih_l0','recurrent.bias_hh_l0','sequence_head.weight','sequence_head.bias','policy.raw_slope','policy.intercept']
    if names!=expected:raise ValueError('blocked_training_objective_fidelity: unexpected source parameters')
    return list(model.parameters())+list(policy.parameters())


def source_loss(model,policy,context_schedule,cfg):
    if cfg.learning_rate!=.005 or cfg.gradient_clip!=1. or cfg.lambda_execution!=.2 or cfg.normative_c_time!=.5 or cfg.normative_c_risk!=1. or cfg.normative_gamma!=2.:raise ValueError('Wrong source objective/optimizer coefficients')
    if tuple(cfg.fixed_sequence)!=tuple(SEQUENCE):raise ValueError('Wrong source tokens/labels')
    context_schedule=torch.as_tensor(context_schedule,dtype=torch.float32)
    isA=(context_schedule==torch.tensor([1.,0,0])).all(1);isB=(context_schedule==torch.tensor([0.,1,0])).all(1)
    if context_schedule.shape!=(7,3) or not (isA|isB).all():raise ValueError('Protected cue leaked into training loss')
    tokens=encode_sequence(cfg.fixed_sequence[:-1],as_tensor=True).unsqueeze(0);targets=torch.as_tensor(cfg.fixed_sequence[1:],dtype=torch.long)
    # Thin time-varying-input adapter: same source RNN and sequence head, one full call.
    hidden,_=model.recurrent(torch.cat([tokens,(context_schedule*model.context_alpha).unsqueeze(0)],dim=-1),None)
    logits=model.sequence_head(hidden);ce=torch.nn.functional.cross_entropy(logits.squeeze(0),targets)
    confidence,_,_=predictive_statistics(logits,targets);speed=policy(confidence)
    execution=execution_loss(logits,targets,speed,cfg.execution_candidate)[0].mean()
    return ce+cfg.lambda_execution*execution,ce,execution,logits


def source_step(model,policy,optimizer,context_schedule,cfg):
    params=_parameters(model,policy)
    if [id(p) for g in optimizer.param_groups for p in g['params']]!=[id(p) for p in params]:raise ValueError('Optimizer parameter contract mismatch')
    settings={'lr':.005,'betas':(.9,.999),'eps':1e-8,'weight_decay':0,'amsgrad':False,'maximize':False,'foreach':None,'capturable':False,'differentiable':False,'fused':None}
    if not isinstance(optimizer,torch.optim.Adam) or len(optimizer.param_groups)!=1 or any(optimizer.param_groups[0].get(k)!=v for k,v in settings.items()):raise ValueError('Optimizer source-default contract mismatch')
    if optimizer.param_groups[0].get('decoupled_weight_decay',False):raise ValueError('Optimizer decoupled weight decay forbidden')
    optimizer.zero_grad();total,ce,execution,logits=source_loss(model,policy,context_schedule,cfg)
    result={'total_loss':float(total.detach()),'sequence_ce':float(ce.detach()),'execution_loss':float(execution.detach()),'optimizer_step_executed':False,'gradient_norm':np.nan,
      'sequence_accuracy':float((logits.argmax(-1)==torch.tensor(SEQUENCE[1:])[None]).float().mean()),'policy_raw_slope':float(policy.raw_slope.detach()),'policy_intercept':float(policy.intercept.detach())}
    if not torch.isfinite(total):return {**result,'status':'nonfinite_loss'}
    total.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):return {**result,'status':'nonfinite_or_missing_gradient'}
    norm=torch.nn.utils.clip_grad_norm_(params,cfg.gradient_clip);result['gradient_norm']=float(norm)
    if not torch.isfinite(norm):return {**result,'status':'nonfinite_joint_gradient_norm'}
    optimizer.step();model.enforce_novel_neutral();result['optimizer_step_executed']=True
    if not all(torch.isfinite(p).all() for p in params):return {**result,'status':'nonfinite_postupdate_weight'}
    return {**result,'status':'completed'}


def continuation(model,policy,contexts,cfg=None):
    cfg=ConfirmatoryConfig() if cfg is None else cfg;params=_parameters(model,policy)
    optimizer=torch.optim.Adam(params,lr=cfg.learning_rate);model.train();policy.train();rows=[];failed=False;failure='completed'
    if len(optimizer.state)!=0:raise ValueError('Expected identical fresh Adam moments')
    for i,c in enumerate(contexts):
        if failed:q={'status':'not_executed_after_numerical_failure','optimizer_step_executed':False,'total_loss':np.nan,'sequence_ce':np.nan,'execution_loss':np.nan,'gradient_norm':np.nan}
        else:
            q=source_step(model,policy,optimizer,c,cfg)
            if q['status']!='completed':failed=True;failure=q['status']
        q.update(continuation_update=i+1,absolute_update=170+i+1,context_schedule=''.join('A' if x[0]==1 else 'B' for x in c),token_count=7,target_sequence=str(tuple(SEQUENCE[1:])))
        rows.append(q)
    return pd.DataFrame(rows),optimizer,failure


def train_arm(root,out,r,arm):
    torch.manual_seed(r['seed']);torch.use_deterministic_algorithms(True)
    model,policy,payload=load_torch(root,r);before=state_hash(model,policy);cfg=ConfirmatoryConfig(**payload['config']);contexts=schedule(arm);validate_schedule(contexts,arm)
    history,optimizer,status=continuation(model,policy,contexts,cfg)
    path=out/'checkpoints_new'/f'High_monotonic_seed_{r["seed"]}_{arm}_step330.pt'
    checkpoint={'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'policy_state_dict':{k:v.detach().cpu().clone() for k,v in policy.state_dict().items()},
       'optimizer_state_dict':optimizer.state_dict(),'original_checkpoint_sha':r['checkpoint_sha256'],'original_update':170,'continuation_updates_planned':160,'updates_executed':int(history.optimizer_step_executed.sum()),'final_update':330,'arm':arm,'seed':r['seed'],'condition':'High_monotonic','config':payload['config'],'continuation_config_sha':CONTRACT['sha256'],'training_status':status,'optimizer_mode':'fine_tune_from_weights_optimizer_reset'}
    with path.open('xb') as f:torch.save(checkpoint,f)
    for key,value in {'seed':r['seed'],'arm':arm,'original_checkpoint_sha':r['checkpoint_sha256']}.items():history[key]=value
    row={'seed':r['seed'],'arm':arm,'status':status,'updates_planned':160,'updates_executed':int(history.optimizer_step_executed.sum()),'initial_model_policy_sha':before,'final_model_policy_sha':state_hash(model,policy),
       'original_checkpoint_sha':r['checkpoint_sha256'],'checkpoint_path':str(path.relative_to(out)),'checkpoint_sha':sha256(path),'optimizer_mode':'fine_tune_from_weights_optimizer_reset','config_sha':CONTRACT['sha256'],'fixed_endpoint':330}
    return row,history


def familiar_validation(out,registry,baseline):
    rows=[];position=[]
    for r in registry.to_dict('records'):
        base={'seed':r['seed'],'arm':r['arm'],'status':r['status'],'checkpoint_sha':r['checkpoint_sha']}
        if r['status']!='completed':rows.append({**base,**{k:np.nan for k in ['A_ce','B_ce','familiar_mean_ce','worst_endpoint_ce','hidden_separation','task_separation']}});continue
        payload=torch.load(out/r['checkpoint_path'],map_location='cpu',weights_only=False);m=frozen_from_state(payload['model_state_dict']);a=natural_trace(m,[1,0,0]);b=natural_trace(m,[0,1,0]);h,z=separation(a,b,m)
        original=baseline[baseline.seed.eq(r['seed'])];ca=float(a.raw_ce.mean());cb=float(b.raw_ce.mean())
        originalmean=float(original[original.cue.isin(['A','B'])].raw_ce.mean())
        rows.append({**base,'A_ce':ca,'B_ce':cb,'familiar_mean_ce':(ca+cb)/2,'worst_endpoint_ce':max(ca,cb),'change_from_original_familiar':(ca+cb)/2-originalmean,
           'A_accuracy':float((a[[f'logit_{i}' for i in range(4)]].to_numpy().argmax(1)==np.array(SEQUENCE[1:])).mean()),'B_accuracy':float((b[[f'logit_{i}' for i in range(4)]].to_numpy().argmax(1)==np.array(SEQUENCE[1:])).mean()),'hidden_separation':h.mean(),'task_separation':z.mean()})
        for t in range(7):position.append({**base,'transition':t+1,'hidden_separation':h[t],'task_separation':z[t]})
    return pd.DataFrame(rows),pd.DataFrame(position)


def protected_evaluation(root,out,registry):
    authorization=json.loads((out/'manifests/d3_protected_evaluation_authorization.json').read_text())
    if len(registry)!=60 or not authorization['all_60_attempts_registered'] or authorization['final_endpoint']!=330:raise ValueError('Protected evaluation attempted before final training freeze')
    for r in registry.to_dict('records'):
        if sha256(out/r['checkpoint_path'])!=authorization['checkpoint_hashes'][r['checkpoint_path']]:raise ValueError('Training endpoint changed before protected evaluation')
    rows=[];signatures=[];histories=[]
    from .mixing_decomposition import decompose
    from .history_diagnostic import signature
    for r in registry.to_dict('records'):
        if r['status']=='completed':
            payload=torch.load(out/r['checkpoint_path'],map_location='cpu',weights_only=False);m=frozen_from_state(payload['model_state_dict'])
        else:m=None
        for cue,vector in cues().items():
            if m is not None:f=natural_trace(m,vector)
            else:f=pd.DataFrame({'transition':np.arange(1,8),'token':SEQUENCE[:-1],'target':SEQUENCE[1:],'raw_ce':np.nan})
            for key,value in {'seed':r['seed'],'arm':r['arm'],'cue':cue,'status':r['status'],'checkpoint_sha':r['checkpoint_sha'],'original_checkpoint_sha':r['original_checkpoint_sha'],'config_sha':CONTRACT['sha256']}.items():f[key]=value
            rows.append(f)
        if m is not None and r['arm'] in ['I','S']:
            d=decompose(m);d['seed']=r['seed'];d['arm']=r['arm'];signatures.append(d)
            history,s=signature(m);history['seed']=r['seed'];history['arm']=r['arm'];histories.append(history)
    return pd.concat(rows,ignore_index=True),pd.concat(signatures,ignore_index=True) if signatures else pd.DataFrame([{'status':'no_valid_final_models'}]),pd.concat(histories,ignore_index=True) if histories else pd.DataFrame([{'status':'no_valid_final_models'}])


def run(root,out,smoke=False):
    objective=json.loads((out/'manifests/d3_objective_contract.json').read_text())
    if objective['status']!='verified':
        rows=pd.DataFrame([{'seed':s,'arm':a,'status':'blocked_training_objective_fidelity'} for s in range(20) for a in ['B','I','S']]);save(out,'d3_model_registry_blocked',rows)
        report(out,'D3_CONTINUITY_TRAINING.md','# D3 blocked\n\nThe original training objective could not be faithfully reconstructed. No additional training or protected evaluation was performed. D1/D2 remain valid.');return {'scientific_status':'blocked_training_objective_fidelity','models_planned':60,'optimizer_steps':0}
    if smoke:
        dump(out/'manifests/d3_training_contract.json',{'contract':CONTRACT,'objective':objective,'code_hashes':codes(),'smoke_reused_not_retrained':True,'checkpoint_rule':'fixed final330, no protected selection'})
        plans=[]
        for arm in ['B','I','S']:
            c=schedule(arm)
            for i in range(160):
                for t in range(7):plans.append({'arm':arm,'continuation_update':i+1,'transition':t+1,'token':SEQUENCE[t],'target':SEQUENCE[t+1],'cue':'A' if c[i,t,0] else 'B','within_sequence_switch':bool(t>0 and np.any(c[i,t]!=c[i,t-1]))})
        save(out,'d3_training_schedule',pd.DataFrame(plans))
        todo=[r for r in records() if r['seed'] in [0,1]];models=[];logs=[]
    else:
        frozen=json.loads((out/'manifests/d3_training_contract.json').read_text())
        for path,want in frozen['code_hashes'].items():
            if sha256(ROOT/path)!=want:raise ValueError('D3 implementation changed after smoke')
        todo=[r for r in records() if r['seed'] not in [0,1]];models=pd.read_csv(out/'tables/d3_smoke_model_registry.csv').to_dict('records');logs=[pd.read_csv(out/'tables/d3_smoke_training_history.csv')]
    for r in todo:
        for arm in ['B','I','S']:
            row,history=train_arm(root,out,r,arm);models.append(row);logs.append(history);print(f'D3 {"smoke" if smoke else "full"}: seed{r["seed"]} arm{arm}; {row["status"]}; updates{row["updates_executed"]}',flush=True)
    registry=pd.DataFrame(models);history=pd.concat(logs,ignore_index=True);prefix='d3_smoke' if smoke else 'd3'
    if len(registry)!=(6 if smoke else 60) or len(history)!=(960 if smoke else 9600):raise ValueError('D3 registry/budget mismatch')
    if registry.groupby('seed').initial_model_policy_sha.nunique().ne(1).any():raise ValueError('Paired arms started from different model/policy weights')
    save(out,prefix+'_model_registry',registry);save(out,prefix+'_training_history',history)
    native=pd.read_csv(out/'tables/d0_native_replay.csv');fam,sep=familiar_validation(out,registry,native);save(out,prefix+'_familiar_validation',fam);save(out,prefix+'_separation_by_position',sep)
    if smoke:return {'models':6,'scientific_updates':int(history.optimizer_step_executed.sum()),'protected_evaluations':0,'smoke_fixed_config_passed':bool(registry.status.eq('completed').all()),'models_reused_in_full':True}
    # All endpoints and familiar-only checks are sealed before any protected rollout.
    hashes=dict(zip(registry.checkpoint_path,registry.checkpoint_sha))
    dump(out/'manifests/d3_training_endpoints_sealed.json',{'all_60_attempts_registered':True,'checkpoint_hashes':hashes,'training_history_sha':sha256(out/'tables/d3_training_history.csv'),'objective_sha':sha256(out/'manifests/d3_objective_contract.json'),'training_contract_sha':sha256(out/'manifests/d3_training_contract.json'),'source_config_sha':CONTRACT['sha256'],'scientific_updates':int(history.optimizer_step_executed.sum())})
    dump(out/'manifests/d3_protected_evaluation_authorization.json',{'all_60_attempts_registered':True,'final_endpoint':330,'checkpoint_hashes':hashes,'familiar_checks_performed':True,'familiar_thresholds_gate_evaluation':False,'training_endpoints_manifest_sha':sha256(out/'manifests/d3_training_endpoints_sealed.json')})
    raw,d1,d2=protected_evaluation(root,out,registry);save(out,'d3_natural_transition',raw);save(out,'d3_d1_signatures',d1);save(out,'d3_history_signatures',d2)
    means=raw.groupby(['seed','arm','cue'],as_index=False).raw_ce.agg(lambda x:x.mean() if x.notna().all() else np.nan);save(out,'d3_natural_seed',means)
    effects=paired_effects(means);save(out,'d3_paired_effects',effects)
    effects=novel_means(effects,[],['S_minus_I','S_minus_B','I_minus_B']);boot=seed_bootstrap(effects,['S_minus_I','S_minus_B','I_minus_B'],['cue']);save(out,'d3_effect_intervals',boot)
    primary=boot[boot.cue.eq('Mixed')&boot.metric.eq('S_minus_I')]
    report(out,'D3_CONTINUITY_TRAINING.md','# D3 — familiar continuity learning\n\nAllthree arms strictly loaded each original High model and policy, then used identical fresh Adam state, the source full loss, joint clipping, sequence reset and160 updates. Smoke seeds0/1 count among the60 final models, so exactly9600 scientific updates were planned. B uses10-sequence blocks; I alternates whole sequences; S alternates A→B/B→A within each sequence at3→4. Every scored position sees80 A and80 B exposures. Total context-switch events are not matched. No protected input, target loss or checkpoint selection entered training.\n\nAll60 endpoints were registered and training hashes sealed before protected evaluation. Primary final-step330 Mixed S−I CE:\n\n'+table(primary[['mean','ci_low','ci_high','n_seeds','status']])+'\n\nFamiliar A/B, accuracy and raw/task separation are saved for every arm and seed, with all failures retained. The .20-nat familiar comparability and .80 separation-retention checks qualify interpretation; they never gate evaluation or remove seeds. D1/D2 changes on final I/S models are converging diagnostics, not formal mediation.\n')
    return {'models':60,'scientific_updates':int(history.optimizer_step_executed.sum()),'planned_updates':9600,'protected_evaluation_rows':len(raw),'original_weights_unchanged':True,'smoke_models_not_retrained':True}
