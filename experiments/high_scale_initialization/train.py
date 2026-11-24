"""One fixed40-run experiment, source-exact Tanh parity and sealed checkpoints."""
import time
import pandas as pd
from .core import *

def verify_inputs(gate):
 for n,h in {**gate['inputs'],**gate['code']}.items():
  if sha(ROOT/n)!=h:raise ValueError('Frozen input/code changed')

def run():
 gate=json.loads((OUT/'audit/pretraining_gate.json').read_text());assert gate['passed'];verify_inputs(gate)
 if (OUT/'audit/training_started.json').exists():raise FileExistsError('No duplicate or replacement scientific run')
 code={str(p.relative_to(ROOT)):sha(p) for folder in ['experiments','tests'] for p in (ROOT/folder/'high_scale_initialization').glob('*.py')};dump(OUT/'audit/training_started.json',dict(code=code,config=CONFIG));torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();learning=[];training=[];registry=[];parity=[];failures=[]
 historical=pd.read_csv(ROOT/'outputs/confirmatory_srn_final/confirmatory_training_sequence_level.csv',float_precision='round_trip');historical=historical[historical.condition=='High_monotonic']
 for seed in range(20):
  for a,m in pair(seed).items():
   p=policy();o=optimizer(m,p);failed=False;executed=0;failurepoint=None;hist=[];last_finite={n:v.detach().clone() for n,v in m.state_dict().items()}
   for k in range(331):
    if k:
     cue=training_cue(k)
     if failed:r=dict(status='not_executed_after_failure',optimizer_step_executed=False,ce_before=np.nan,execution_loss=np.nan,total_loss=np.nan,accuracy_before=np.nan,hidden_absmax=np.nan,saturation_95=np.nan,saturation_99=np.nan,gradient_norm=np.nan)
     else:
      r=step(m,p,o,cue);executed+=int(r['optimizer_step_executed'])
      if r['status']!='completed':failed=True;failurepoint=k;failures.append(dict(seed=seed,architecture=a,first_failure_update=k,failure_status=r['status'],**spectra(m),hidden_absmax=r['hidden_absmax'],gradient_norm=r['gradient_norm']))
      else:last_finite={n:v.detach().clone() for n,v in m.state_dict().items()}
     record=dict(seed=seed,architecture=a,update=k,training_cue=cue,**r,**spectra(m));training.append(record);hist.append(record)
    for c in ['A','B']:
     tr=None if failed else trace(m,c)
     learning.append(dict(seed=seed,architecture=a,update=k,cue=c,ce=float(tr['ce'].mean()) if tr else np.nan,accuracy=float(tr['accuracy'].mean()) if tr else np.nan,hidden_absmax=float(abs(tr['hidden']).max()) if tr else np.nan,status='completed' if tr else 'unavailable',scientific_updates_executed=executed))
    if k in STEPS:
     path=None;h=None
     if not failed:
      path=f'checkpoints/seed_{seed}_{a}_step_{k}.pt';payload=dict(seed=seed,architecture=a,update=k,config=CONFIG,model_state_dict={n:v.detach().clone() for n,v in m.state_dict().items()},policy_state_dict={n:v.detach().clone() for n,v in p.state_dict().items()},optimizer_state_dict=o.state_dict(),model_policy_hash=state_hash(m,p),training_code=code)
      with (OUT/path).open('xb') as f:torch.save(payload,f)
      h=sha(OUT/path)
     registry.append(dict(seed=seed,architecture=a,checkpoint=k,status='completed' if not failed else 'unavailable_after_failure',checkpoint_path=path,checkpoint_sha=h,updates_executed=executed))
    if k==170 and a=='RNN-Tanh':
     oldpath=ROOT/f'outputs/confirmatory_srn_final/confirmatory_checkpoints/High_monotonic_seed_{seed}.pt';old=torch.load(oldpath,map_location='cpu',weights_only=False);model_exact=not failed and all(torch.equal(v,old['model_state_dict'][n]) for n,v in m.state_dict().items());policy_exact=not failed and all(torch.equal(v,old['policy_state_dict'][n]) for n,v in p.state_dict().items());loss=np.array([r['ce_before'] for r in hist]);source_loss=historical[historical.seed==seed].sort_values('training_sequence').sequence_CE.to_numpy();maxloss=float(abs(loss-source_loss).max()) if np.isfinite(loss).all() else np.nan
     maxparam=max(float(abs(v.detach()-old['model_state_dict'][n]).max()) for n,v in m.state_dict().items()) if not failed else np.nan
     parity.append(dict(seed=seed,checkpoint=170,canonical_checkpoint_sha=sha(oldpath),model_tensors_bitwise_equal=model_exact,policy_tensors_bitwise_equal=policy_exact,preupdate_losses_bitwise_equal=bool(np.array_equal(loss,source_loss)),maximum_training_ce_error=maxloss,maximum_model_parameter_error=maxparam,independent_historical_arm=False))
   pd.DataFrame(hist).to_csv(OUT/'audit'/f'training_seed_{seed}_{a}.csv',index=False)
   if failed:
    with (OUT/'checkpoints'/f'seed_{seed}_{a}_last_finite_diagnostic.pt').open('xb') as f:torch.save(dict(seed=seed,architecture=a,first_failure_update=failurepoint,last_finite_model_only=last_finite,not_endpoint=True),f)
   print(f'seed{seed} {a}: '+('FAILED@'+str(failurepoint) if failed else 'completed330')+f'; updates{executed}; elapsed{time.monotonic()-start:.1f}s',flush=True)
 training=pd.DataFrame(training);learning=pd.DataFrame(learning);registry=pd.DataFrame(registry);parity=pd.DataFrame(parity)
 assert len(training)==13200 and len(learning)==26480 and len(registry)==240 and len(parity)==20
 training.to_csv(OUT/'training_measurements.csv',index=False);learning.to_csv(OUT/'learning_trajectories.csv',index=False);registry.to_csv(OUT/'tables/checkpoint_registry.csv',index=False);parity.to_csv(OUT/'tables/canonical_training_parity.csv',index=False);pd.DataFrame(failures,columns=['seed','architecture','first_failure_update','failure_status','recurrent_spectral_norm','recurrent_spectral_radius','hidden_absmax','gradient_norm']).to_csv(OUT/'tables/failures.csv',index=False)
 verify_inputs(gate)
 for n,h in code.items():
  if sha(ROOT/n)!=h:raise ValueError('Training source mutated')
 good=registry[registry.status=='completed'];seal=dict(all40_attempts_retained=True,all240_checkpoint_slots_retained=True,scientific_updates_planned=13200,scientific_updates_executed=int(training.optimizer_step_executed.sum()),diagnostic_updates=0,checkpoint_hashes=dict(zip(good.checkpoint_path,good.checkpoint_sha)),training_sha=sha(OUT/'training_measurements.csv'),learning_sha=sha(OUT/'learning_trajectories.csv'),protected_evaluations=0,canonical170_model_matches=int(parity.model_tensors_bitwise_equal.sum()),canonical170_policy_matches=int(parity.policy_tensors_bitwise_equal.sum()),canonical170_loss_matches=int(parity.preupdate_losses_bitwise_equal.sum()),failed_networks=len(failures),code=code);dump(OUT/'audit/training_seal.json',seal);append('## Registered training sealed\n'+json.dumps({k:v for k,v in seal.items() if k not in ['checkpoint_hashes','code']}));print('SEALED; no protected probes yet',flush=True)

if __name__=='__main__':run()
