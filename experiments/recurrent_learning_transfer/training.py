"""Registered40 scientific trajectories and640 isolated diagnostic updates."""
import json,time
import numpy as np
import pandas as pd
import torch
from .design import *
from .network import pair
from .learning import policy,optimizer,step,diagnostic,familiar,state_hash


def run():
 gate=json.loads((OUT/'audit/pretraining_gate.json').read_text())
 if not gate['passed']:raise ValueError('Mandatory pretraining gate failed')
 for name,h in {**gate['inputs'],**gate['code']}.items():
  if hash_file(ROOT/name)!=h:raise ValueError('Pretraining input/code changed')
 if (OUT/'audit/training_started.json').exists():raise FileExistsError('Registered training already started; no duplicate/replacement runs')
 code=codes();dump(OUT/'audit/training_started.json',dict(config=CONFIG,code=code));torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();learning=[];transfers=[];updates=[];registry=[];initial=[]
 for seed in range(20):
  for arch,m in pair(seed).items():
   p=policy();o=optimizer(m,p);initial_hash=state_hash(m,p,o);initial.append(dict(seed=seed,architecture=arch,initial_model_policy_adam_hash=initial_hash,initial_recurrent_spectral_norm=float(torch.linalg.matrix_norm(m.wh.detach(),2)),model_parameters=172,policy_parameters=2));failed=False;executed=0;last_finite={k:v.detach().clone() for k,v in m.state_dict().items()};failure=None;history=[]
   for k in range(331):
    if k:
     cue=training_cue(k)
     if failed:r=dict(status='not_executed_after_failure',optimizer_step_executed=False,ce_before=np.nan,execution_loss=np.nan,total_loss=np.nan,gradient_norm=np.nan)
     else:
      r=step(m,p,o,cue);executed+=int(r['optimizer_step_executed'])
      if r['status']!='completed':failed=True;failure=r['status']
      else:last_finite={key:v.detach().clone() for key,v in m.state_dict().items()}
     row=dict(seed=seed,architecture=arch,update=k,training_cue=cue,**r);updates.append(row);history.append(row)
    state=familiar(m) if not failed else None
    for cue in ['A','B']:
     learning.append(dict(seed=seed,architecture=arch,update=k,cue=cue,ce=float(state[cue]['ce'].mean()) if state else np.nan,accuracy=float(state[cue]['accuracy'].mean()) if state else np.nan,status='completed' if not failed else 'unavailable_after_failure',last_training_context=training_cue(k) if k else 'none',is_block_boundary=bool(k>0 and k%10==0),before_next_scientific_update=k+1 if k<330 else None))
    if k in DIAG:
     path=None;h=None
     if not failed:
      before=state_hash(m,p,o);rows=diagnostic(m,p,o)
      if state_hash(m,p,o)!=before:raise ValueError('Clones contaminated scientific trajectory')
      for r in rows:transfers.append(dict(seed=seed,architecture=arch,checkpoint=k,**r))
      path=f'checkpoints/seed_{seed}_{arch}_step_{k}.pt';payload=dict(seed=seed,architecture=arch,update=k,config=CONFIG,model_state={key:v.detach().clone() for key,v in m.state_dict().items()},policy_state=p.state_dict(),optimizer_state=o.state_dict(),scientific_state_hash=before,training_code=code,new_cohort_not_original=True)
      with (OUT/path).open('xb') as f:torch.save(payload,f)
      h=hash_file(OUT/path)
     else:
      for train_cue in ['A','B']:
       for eval_cue in ['A','B']:transfers.append(dict(seed=seed,architecture=arch,checkpoint=k,train_context=train_cue,evaluation_context=eval_cue,within_context=train_cue==eval_cue,loss_before=np.nan,loss_after=np.nan,improvement=np.nan,diagnostic_status='unavailable_after_scientific_failure',diagnostic_update_executed=False,optimizer_restored=False,scientific_state_unchanged=True))
     registry.append(dict(seed=seed,architecture=arch,checkpoint=k,status='completed' if not failed else 'unavailable_after_failure',updates_executed=executed,checkpoint_path=path,checkpoint_sha=h))
   if failed:
    with (OUT/'checkpoints'/f'seed_{seed}_{arch}_failure_diagnostic.pt').open('xb') as f:torch.save(dict(seed=seed,architecture=arch,failure=failure,last_finite_model_only=last_finite,last_finite_not_an_endpoint=True,updates_executed=executed),f)
   pd.DataFrame(history).to_csv(OUT/'audit'/f'training_seed_{seed}_{arch}.csv',index=False)
   print(f'seed{seed} {arch}: '+('FAILED '+str(failure) if failed else 'completed')+f'; scientific{executed}/330; elapsed{time.monotonic()-start:.1f}s',flush=True)
 learning=pd.DataFrame(learning);transfers=pd.DataFrame(transfers);updates=pd.DataFrame(updates);registry=pd.DataFrame(registry);initial=pd.DataFrame(initial)
 if len(learning)!=26480 or len(transfers)!=1280 or len(updates)!=13200 or len(registry)!=320 or len(initial)!=40:raise ValueError('Registered coverage/budget discrepancy')
 if not initial.groupby('seed').initial_model_policy_adam_hash.nunique().eq(1).all():raise ValueError('Paired arms not identically initialized')
 learning.to_csv(OUT/'learning_trajectories.csv',index=False);transfers.to_csv(OUT/'learning_transfer.csv',index=False);updates.to_csv(OUT/'tables/training_history.csv',index=False);registry.to_csv(OUT/'tables/checkpoint_registry.csv',index=False);initial.to_csv(OUT/'tables/initialization.csv',index=False)
 for name,h in gate['inputs'].items():
  if hash_file(ROOT/name)!=h:raise ValueError('Historical/source input modified during training')
 if code!=codes():raise ValueError('Training implementation changed during run')
 valid=registry[registry.status=='completed'];n_diag=int(transfers.drop_duplicates(['seed','architecture','checkpoint','train_context']).diagnostic_update_executed.eq(True).sum());seal=dict(all40_attempts_registered=True,all320_checkpoint_slots_registered=True,scientific_updates_planned=13200,scientific_updates_executed=int(updates.optimizer_step_executed.eq(True).sum()),diagnostic_updates_planned=640,diagnostic_updates_executed=n_diag,checkpoint_hashes=dict(zip(valid.checkpoint_path,valid.checkpoint_sha)),checkpoint_registry_sha=hash_file(OUT/'tables/checkpoint_registry.csv'),learning_sha=hash_file(OUT/'learning_trajectories.csv'),diagnostics_sha=hash_file(OUT/'learning_transfer.csv'),protected_evaluations=0,code=code)
 dump(OUT/'audit/training_seal.json',seal);append('## All scientific trajectories and diagnostic states sealed\n'+json.dumps({k:v for k,v in seal.items() if k not in ['checkpoint_hashes','code']}));print('TRAINING AND DIAGNOSTICS SEALED; protected evaluations0',flush=True)

if __name__=='__main__':run()
