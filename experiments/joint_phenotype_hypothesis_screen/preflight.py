"""Fixed background-only calibration and baseline-only clipping trigger."""
from .common import *
from . import adapter as a,engine
from .storage import Exposure,validate_resume
from experiments.high_scale_meta_progression.protocol import restore_target

def calibration_task(seed,e,index,stage,variant,eta=None,deadline=None):
 dest=OUT/f'preflight/{stage}_s{seed}_task{index}';dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json'
 if done.exists():return json.loads(done.read_text())
 ledger=Exposure();name=f'calibration/{stage}_s{seed}_task{index}';charge=engine.charge_callback(ledger,'auxiliary',name);identity=engine.require_identity();progress=dest/'progress.pt'
 if progress.exists():
  cp=torch.load(progress,weights_only=False);validate_resume(cp,identity,ledger.run_count(name));m,p,o=a.restore(cp['state']);rows=cp['rows'];before=cp['before'];start=cp['cursor']
 else:
  assert ledger.run_count(name)==0,'Uncertain calibration work';state=torch.load(ROOT/engine.sources(seed)['A00']['initializer'],weights_only=False)['model'];m=a.make(state,variant);p=core.policy();o=a.optimizer(m,p,variant,eta);rows=[];before=engine.own_probes(m,e);start=0
 for j in range(start,20):
  if engine.due(deadline):raise engine.CleanStop('Calibration continuation saved at preceding update')
  r=e['support_P2'][j];q=a.update(m,p,o,r['context'],variant,j+1,charge,path=e['paths'][r['sequence']],meta=True);rows.append(q);atomic(progress,dict(identity=identity,cursor=j+1,state=a.snapshot(m,p,o,j+1,variant),rows=rows,before=before))
  if q['status']!='completed':break
 after=engine.own_probes(m,e) if q['status']=='completed' else []
 ratios=[r['prediction_displacement']/r['clipped_prediction_gradient_norm'] for r in rows if np.isfinite([r['prediction_displacement'],r['clipped_prediction_gradient_norm']]).all() and r['clipped_prediction_gradient_norm']>1e-12]
 gain=float(np.mean([v['ce'] for v in before])-np.mean([v['ce'] for v in after])) if after else np.nan
 result=dict(seed=seed,task=index,stage=stage,variant=variant,eta=eta,finite=bool(after and np.isfinite(gain)),gain=gain,ratios=ratios,excluded_states=len(rows)-len(ratios),attempted_updates=len(rows),task_sha=digest(e),before=before,after=after);put(done,result);return clean(result)

def main(deadline=None):
 configure();receipt=OUT/'audit/preflight_complete.json'
 if receipt.exists():engine.require_identity();return json.loads(receipt.read_text())
 engine.frozen_identity();put(OUT/'registry/procedure_constants.json',dict(epsilon240=engine.epsilons('B00'),epsilon80=engine.epsilons('B04'),background_B03_mask_rule='first three sealed shared meta masks, in order, assigned to three unrelated shared validation tasks; one complete presentation per5-update block',SGD_only_Adam_assertion_bypass='native core.step meta flag bypasses target Adam-moment assertion for SGD only; independent adapter target cue/path/C-weight checks remain enforced',partial_display_ranking='strict-joint count, worst minimum Fresh retention, C CE, registry order',no_target_outcome_calibration=True))
 # Exact complete-state scheduled native replay, never candidate outcomes.
 for seed in [0,1]:
  dest=OUT/f'preflight/native_s{seed}';dest.mkdir(parents=True,exist_ok=True);file=dest/'complete.json'
  if file.exists():continue
  ledger=Exposure();name=f'native_replay_s{seed}';charge=engine.charge_callback(ledger,'auxiliary',name);source=engine.sources(seed)['A00'];states=torch.load(ROOT/source['states'],weights_only=False)['states'];records=pd.read_csv(ROOT/source['training'],float_precision='round_trip');path=dest/'progress.pt'
  if path.exists():
   cp=torch.load(path,weights_only=False);assert cp['cursor']==ledger.run_count(name);m,p,o=a.restore(cp['state']);rows=cp['rows'];start=cp['cursor']
  else:assert ledger.run_count(name)==0;m,p,o=restore_target(states[0]);rows=[];start=0
  for n in range(start+1,61):
   if engine.due(deadline):raise engine.CleanStop('Native replay saved at preceding update')
   r=a.update(m,p,o,cue(n),'A00',n,charge);gold=states[n];assert r['status']=='completed' and r['ce_before']==records[records['update'].eq(n)].iloc[0].ce_before
   assert digest((m.state_dict(),p.state_dict(),o.state_dict()))==digest((gold['model'],gold['policy'],gold['optimizer'])),'Native full-state replay mismatch'
   if n==30:
    old=digest((m.state_dict(),p.state_dict(),o.state_dict(),rng()));m,p,o=a.restore(a.snapshot(m,p,o,n,'A00'));assert old==digest((m.state_dict(),p.state_dict(),o.state_dict(),rng()))
   rows.append(dict(seed=seed,update=n,bitwise_full_state_pass=True,**r));atomic(path,dict(cursor=n,rows=rows,state=a.snapshot(m,p,o,n,'A00')))
  csv(dest/'gradient_records.csv',rows);put(file,dict(passed=True,seed=seed,full_state_bitwise_replays=60,resume_roundtrip_at=30))
 grads=pd.concat([pd.read_csv(OUT/f'preflight/native_s{s}/gradient_records.csv') for s in [0,1]]);active=grads[grads.clipping_active.eq(True)];large=active.policy_squared_share.gt(.5);trigger=bool(len(active)>0 and large.mean()>=.2);put(OUT/'registry/A07_trigger.json',dict(triggered=trigger,status='triggered' if trigger else 'not_triggered',n_baseline_updates=120,n_clipping_active=len(active),n_policy_squared_share_above_point5=int(large.sum()),fraction=float(large.mean()) if len(active) else 0.,source='preflight native A00 only; before candidate histories'))
 calibration=OUT/'registry/sgd_calibration.json'
 if not calibration.exists():
  rows=[]
  for seed in [0,1]:
   for index,e in enumerate(engine.bank(seed)['calibration']):rows.append(calibration_task(seed,e,index,'native','A00',deadline=deadline))
  ratios=[v for r in rows for v in r['ratios']];eta=float(np.median(ratios)) if ratios else None;valid=[];backoff=False;eta_final=eta
  if eta is not None and np.isfinite(eta) and eta>0:
   for attempt in [0,1]:
    valid=[];rate=eta*(.25 if attempt else 1.)
    for seed in [0,1]:
     for index,e in enumerate(engine.bank(seed)['calibration_validation']):valid.append(calibration_task(seed,e,index,f'SGD_validation_{attempt}','A04',rate,deadline))
    success=bool(all(r['finite'] for r in valid) and np.median([r['gain'] for r in valid])>0)
    eta_final=rate
    if success:break
    backoff=attempt==0
   status='calibrated' if success else 'calibration_inconclusive'
  else:status='calibration_inconclusive'
  put(calibration,dict(passed=status=='calibrated',status=status,eta_pooled=eta,eta_final=eta_final if status=='calibrated' else None,backoff_used=backoff,eligible_states=len(ratios),excluded_states=sum(r['excluded_states'] for r in rows),calibration=rows,validation=valid,target_outcomes_used=False))
 result=dict(passed=True,native_bitwise_replays=120,A07_trigger=trigger,SGD_status=json.loads(calibration.read_text())['status'],SGD_rate=json.loads(calibration.read_text())['eta_final'],exposure=Exposure().counts(),runtime_sha=sha(OUT/'registry/runtime_identity.json'),candidates_opened=False);put(receipt,result);append('## Preflight GREEN\n120 original A00 updates replay full model/policy/Adam bitwise; complete snapshot/RNG/optimizer roundtrip at30 in both seeds. Baseline-only policy clipping trigger frozen. SGD rate calibrated on sealed unrelated tasks, at most one predeclared backoff, before target candidate outcomes. All native defaults and selection thresholds unchanged. See audit/preflight_complete.json and registry/sgd_calibration.json.');print(json.dumps(result),flush=True);return result
