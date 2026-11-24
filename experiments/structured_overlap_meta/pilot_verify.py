"""Saved pilot states/distributions/criteria verification, no native rollout replay."""
import concurrent.futures,multiprocessing
from .common import *
from .capacity import criteria,decision
from experiments.context_experience_prior.verify import verify_position_rows
from experiments.heldout_context_meta_scaling_10k.verify import pack_state

def validate_steps(rows,roles):
 try:
  assert len(rows)==330
  for k,r in enumerate(rows,1):
   role='X' if (k-1)//10%2==0 else 'Y';assert r['update']==k and r['role']==role and r['cue']==roles[role] and r['optimizer_step_executed'];np.testing.assert_allclose(r['total_loss'],r['ce_before']+.2*r['execution_loss'],rtol=1e-6,atol=2e-6)
 except (AssertionError,KeyError,TypeError) as exc:raise ValueError('Pilot budget, cue schedule or objective mismatch') from exc
 return True

def one(row):
 initialize_process();d=OUT/row['directory'];assert sha(d/'states.pt')==row['states_sha'];b=torch.load(d/'states.pt',weights_only=False);e=b['episode'];state,opt,src=source(row['seed']);assert b['source_sha']==sha(src) and b['outer_updates_executed']==0;np.testing.assert_array_equal(pack_state(state,opt),pack_state(state,b['original_outer_optimizer']));states=b['states'];assert len(states)==331 and all(s['status']=='completed' for s in states);qcol=4+'ABC'.index(e['roles']['Q']);column=state['recurrent.weight_ih_l0'][:,qcol]
 for k,s in enumerate(states):
  assert s['update']==k and torch.equal(s['model']['recurrent.weight_ih_l0'][:,qcol],column) and all(torch.isfinite(v).all() for v in s['model'].values())
  if k==0:
   assert all(torch.equal(v,s['model'][n]) for n,v in state.items()) and not s['optimizer']['state'] and all(torch.count_nonzero(v)==0 for v in s['policy'].values())
  else:
   assert s['optimizer']['param_groups'][0]['lr']==.005
   for v in s['optimizer']['state'].values():
    assert float(v['step'])==k and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()
    if v['exp_avg'].shape==(8,7):assert torch.count_nonzero(v['exp_avg'][:,qcol])==torch.count_nonzero(v['exp_avg_sq'][:,qcol])==0
 pos=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip');assert len(pos)==331*28 and not pos.duplicated(['update','role','subtrial','position']).any();verify_position_rows(pos);paths=np.array(e['paths']['X']+e['paths']['Y']);ii=pos.subtrial.to_numpy(int)+np.where(pos.role.eq('Y'),2,0);tt=pos.position.to_numpy(int)-1;assert pos.position.between(1,7).all();np.testing.assert_array_equal(pos.token,paths[ii,tt]);np.testing.assert_array_equal(pos.target,paths[ii,tt+1]);np.testing.assert_array_equal(pos.critical,np.array(e['critical_mask'])[ii,tt]);np.testing.assert_array_equal(pos.shared_eligible,np.array(e['novel_mask'])[pos.subtrial.to_numpy(int),tt]);assert pos.cue.eq(pos.role.map(e['roles'])).all()
 curves=pd.read_csv(d/'learning.csv',float_precision='round_trip');assert len(curves)==331;first=None
 for k,g in pos.groupby('update',sort=True):
  g=g.sort_values(['role','subtrial','position']);r=criteria(g.ce.to_numpy().reshape(4,7),g.correct.to_numpy(bool).reshape(4,7),e['critical_mask']);saved=curves[curves['update'].eq(k)].iloc[0]
  for n,v in r.items():
   if isinstance(v,bool):assert bool(saved[n])==v
   else:np.testing.assert_allclose(saved[n],v,rtol=1e-12,atol=1e-12,equal_nan=True)
  if r['success'] and first is None:first=k
 assert bool(row['reached'])==(first is not None) and (pd.isna(row['first_pass']) if first is None else row['first_pass']==first) and bool(row['final_success'])==bool(curves.iloc[-1].success)
 t=pd.read_csv(d/'training.csv',float_precision='round_trip');validate_steps(t.to_dict('records'),e['roles']);loss=curves.set_index('update');np.testing.assert_allclose(t.ce_before,[loss.loc[u-1,'CE_X' if role=='X' else 'CE_Y'] for u,role in zip(t['update'],t.role)],rtol=1e-5,atol=1e-4)
 return dict(seed=int(row['seed']),task=int(row['task']),states=331,positions=len(pos),updates=330,optimizer_updates_in_audit=0,native_forwards=0)

def main():
 f=read('pilot_cases');assert f.status.eq('completed').all()
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,f.to_dict('records')))
 assert decision(f)==json.loads((OUT/'audit/capacity_decision.json').read_text());write_frame(OUT/'audit/pilot_verified.csv',rows);dump(OUT/'audit/pilot_verified.json',dict(passed=True,cases=len(rows),states=sum(r['states'] for r in rows),positions=sum(r['positions'] for r in rows),updates=sum(r['updates'] for r in rows),all_initial_models_and_reset_states=True,all_class_probabilities_labels_and_criteria=True,source_outer_moments_preserved_unused=True,optimizer_updates_in_audit=0,native_forwards=0));append('## Pilot integrity verification complete\nAll72×331model/policy/Adamstatesand667,296positionprobability/logit/CE/margin/labelrowschecked. Allsourceinitialweights/resets,330updatecadence,withheldQcolumns, originaloutermomentcopiesandfirst/finalcapacitycriteriareproducewithoutnativeforward/optimizerreplay. Capacitygate24/24passes; scientificmeta-trainingmaystartafterprospectivestreamseal.')
 print('Pilot saved-data verification PASSED',flush=True)
if __name__=='__main__':main()
