"""80 fixed-budget new models; protected evaluation is a separate sealed stage."""
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from confirmatory_srn.model import MonotonicExecutionPolicy
from confirmatory_srn.config import ConfirmatoryConfig
from execution_objective_search.objectives import execution_loss,predictive_statistics
from .protocol import *
from .model import make_models,tensor_hash


def make_policy():return MonotonicExecutionPolicy()

def validate_training_cue(cue):
 if cue not in ['A','B']:raise ValueError('Protected/non-familiar input forbidden in training')
 return True

def full_loss(logits,policy):
 y=torch.tensor(TARGET,dtype=torch.long);ce=torch.nn.functional.cross_entropy(logits,y);confidence=predictive_statistics(logits.unsqueeze(0),y)[0];speed=policy(confidence);ex=execution_loss(logits.unsqueeze(0),y,speed,ConfirmatoryConfig().execution_candidate)[0].mean()
 return ce+.2*ex,ce,ex

def make_optimizer(model,policy):return torch.optim.Adam(list(model.parameters())+list(policy.parameters()),lr=.005)

def update(model,policy,optimizer,cue):
 validate_training_cue(cue);params=list(model.parameters())+list(policy.parameters());optimizer.zero_grad();z,_=model(torch.tensor(windows()),torch.tensor(CUES[cue]));total,ce,ex=full_loss(z,policy)
 row=dict(sequence_ce=float(ce.detach()),execution_loss=float(ex.detach()),total_loss=float(total.detach()),sequence_accuracy=float((z.argmax(1)==torch.tensor(TARGET)).float().mean()),gradient_norm=np.nan,optimizer_step_executed=False,status='nonfinite_loss')
 if not torch.isfinite(total) or not torch.isfinite(z).all():return row
 total.backward()
 if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):row['status']='nonfinite_or_missing_gradient';return row
 norm=torch.nn.utils.clip_grad_norm_(params,1.);row['gradient_norm']=float(norm)
 if not torch.isfinite(norm):row['status']='nonfinite_joint_norm';return row
 optimizer.step();model.enforce_novel_neutral();row['optimizer_step_executed']=True
 row['status']='completed' if all(torch.isfinite(p).all() for p in params) else 'nonfinite_postupdate_weight'
 return row

def _raw_hash(x):return hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()

def run():
 if not (OUT/'audit/pretraining_gate.json').is_file():raise ValueError('Pretraining GREEN gate required')
 gate=json.loads((OUT/'audit/pretraining_gate.json').read_text())
 if not gate['passed']:raise ValueError('Failed pretraining gate')
 for name,h in {**gate['inputs'],**gate['code']}.items():
  if hash_file(ROOT/name)!=h:raise ValueError('Pretraining source/code changed: '+name)
 if (OUT/'audit/training_started.json').exists():raise FileExistsError('Scientific training already started; no duplicate runs')
 code=codes();dump(OUT/'audit/training_started.json',dict(config=CONFIG,code=code));torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 rows=[];histories=[];init=[];start=time.monotonic()
 for seed in range(20):
  models=make_models(seed)
  for architecture,model in models.items():
   policy=make_policy();optimizer=make_optimizer(model,policy);failed=False;history=[];executed=0;last_finite={k:v.detach().clone() for k,v in model.state_dict().items()}
   init.append(dict(seed=seed,architecture=architecture,input_sha=_raw_hash(model.wi),output_sha=_raw_hash(model.wo),recurrent_sha=_raw_hash(model.wh) if model.wh is not None else None,recurrent_spectral_norm=float(torch.linalg.matrix_norm(model.wh,2)) if model.wh is not None else None,initial_model_hash=tensor_hash(model),initial_policy_slope=float(policy.raw_slope.detach()),initial_policy_intercept=float(policy.intercept.detach()),model_parameters=sum(p.numel() for p in model.parameters()),policy_parameters=2))
   for step in range(1,1001):
    cue=training_cue(step)
    if failed:q=dict(status='not_executed_after_failure',optimizer_step_executed=False,sequence_ce=np.nan,execution_loss=np.nan,total_loss=np.nan,sequence_accuracy=np.nan,gradient_norm=np.nan)
    else:
     q=update(model,policy,optimizer,cue);executed+=int(q['optimizer_step_executed'])
     if q['status']!='completed':failed=True
     else:last_finite={k:v.detach().clone() for k,v in model.state_dict().items()}
    history.append(dict(seed=seed,architecture=architecture,update=step,training_cue=cue,**q))
    if step in STEPS:
     path=None;checkpoint_sha=None
     if not failed:
      path=f'checkpoints/seed_{seed}_{architecture}_step_{step}.pt'
      payload=dict(seed=seed,architecture=architecture,step=step,status='completed',new_factorial_cohort=True,config=CONFIG,model_state={k:v.detach().clone() for k,v in model.state_dict().items()},policy_state=policy.state_dict(),optimizer_state=optimizer.state_dict(),training_code=code)
      with (OUT/path).open('xb') as f:torch.save(payload,f)
      checkpoint_sha=hash_file(OUT/path)
     rows.append(dict(seed=seed,architecture=architecture,step=step,status='completed' if not failed else 'unavailable_after_numerical_failure',updates_executed=executed,checkpoint_path=path,checkpoint_sha=checkpoint_sha))
   if failed:
    with (OUT/'checkpoints'/f'seed_{seed}_{architecture}_last_finite_diagnostic.pt').open('xb') as f:torch.save(dict(seed=seed,architecture=architecture,last_finite_model_state=last_finite,scientific_status='failed_not_an_endpoint'),f)
   histories.extend(history);print(f'seed{seed} {architecture}: '+('FAILED' if failed else 'completed')+f'; executed{executed}/1000; elapsed{time.monotonic()-start:.1f}s',flush=True)
   pd.DataFrame(history).to_csv(OUT/'audit'/f'training_seed_{seed}_{architecture}.csv',index=False)
 registry=pd.DataFrame(rows);initial=pd.DataFrame(init)
 if len(registry)!=320 or len(histories)!=80000 or len(initial)!=80:raise ValueError('Budget/coverage ledger mismatch')
 for seed,g in initial.groupby('seed'):
  if g.input_sha.nunique()!=1 or g.output_sha.nunique()!=1 or g[g.architecture.str.startswith('RNN')].recurrent_sha.nunique()!=1:raise ValueError('Paired initialization failed')
 registry.to_csv(OUT/'tables/checkpoint_registry.csv',index=False);initial.to_csv(OUT/'tables/initialization.csv',index=False);pd.DataFrame(histories).to_csv(OUT/'tables/training_history.csv',index=False)
 for name,h in gate['inputs'].items():
  if hash_file(ROOT/name)!=h:raise ValueError('Historical/source bytes changed')
 if codes()!=code:raise ValueError('Implementation changed during scientific training')
 available=registry[registry.status=='completed'];seal=dict(all_80_attempts_registered=True,all_320_checkpoint_slots_registered=True,planned_updates=80000,executed_updates=int(pd.DataFrame(histories).optimizer_step_executed.sum()),checkpoint_hashes=dict(zip(available.checkpoint_path,available.checkpoint_sha)),registry_sha=hash_file(OUT/'tables/checkpoint_registry.csv'),history_sha=hash_file(OUT/'tables/training_history.csv'),code=code,config=CONFIG,protected_evaluations=0)
 dump(OUT/'audit/training_seal.json',seal);append('## Scientific training sealed\n'+json.dumps({k:v for k,v in seal.items() if k not in ['checkpoint_hashes','code','config']}));print('TRAINING SEALED; protected evaluations0',flush=True)

if __name__=='__main__':run()
