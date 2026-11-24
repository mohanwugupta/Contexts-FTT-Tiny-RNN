"""Read-only diagnostic reset/index/distribution audit; never replays a branch."""
import argparse
from .common import *
from experiments.heldout_context_meta import protocol as old
from experiments.context_experience_prior.verify import verify_position_rows
from experiments.heldout_context_meta.verify import flat

def validate_labels(f,eps,wrong=False):
 try:
  assert f.position.between(1,7).all() and f.position.mod(1).eq(0).all()
  for r in f.itertuples():
   ep=eps[int(r.task)];sequence=int(r.sequence);position=int(r.position);cells={(v['context'],v['sequence']) for v in old.support(ep,'P2','Transfer')};context=r.correct_context if wrong else r.context;assert (context,sequence) in cells;assert r.target==ep['paths'][sequence][position]
   if wrong:assert r.evaluation_context==next(c for c,j in cells if c!=context)
  verify_position_rows(f)
 except (AssertionError,KeyError,IndexError,TypeError) as e:raise ValueError('Invalid diagnostic cue, target index or distribution') from e
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);a=p.parse_args();rows=[]
 for stage in [a.stage]:
  if not (OUT/f'registry/{stage}_heldout.csv').exists():continue
  reg=pd.read_csv(OUT/f'registry/{stage}_heldout.csv')
  for r in reg.itertuples():
   d=OUT/r.directory;eps=json.loads((OUT/f'registry/heldout_v2_seed_{r.seed}.json').read_text())['episodes'];pos=pd.read_csv(d/'positions.csv',float_precision='round_trip');tasks=pd.read_csv(d/'tasks.csv',float_precision='round_trip');blocks=pd.read_csv(d/'blocks.csv',float_precision='round_trip');h=np.load(d/'inner_history.npz');states=torch.load(d/'adapted_states.pt',weights_only=False);model,_=initial(r.seed,r.prior,r.organization,r.arm,r.dose)
   assert len(tasks)==12 and tasks.status.eq('completed').all() and tasks.ce.notna().all() and len(pos)==168 and len(blocks)==48 and len(states)==12;validate_labels(pos,eps)
   assert h['inner'].shape==(12,21,3,174) and h['steps'].shape==(12,21,8);assert np.isfinite(h['inner']).all();np.testing.assert_array_equal(h['steps'],np.broadcast_to(np.arange(21)[None,:,None],(12,21,8)));assert np.all(h['inner'][:,0,1:]==0) and np.all(h['inner'][:,0,0,172:]==0)
   for task in range(12):
    np.testing.assert_array_equal(h['inner'][task,0,0,:172],flat(model));st=states[task];assert st['task']==task;np.testing.assert_array_equal(h['inner'][task,-1,0,:172],flat(st['after']));np.testing.assert_array_equal(h['inner'][task,-1,0,172:],flat(st['policy']));assert all(float(v['step'])==20 for v in st['optimizer']['state'].values())
    p=pos[pos.task.eq(task)];score=tasks[tasks.task.eq(task)].iloc[0];np.testing.assert_allclose(score.ce,p.ce.mean(),atol=1e-12,rtol=1e-12);np.testing.assert_allclose(score.conflict_accuracy,p[p.position.eq(3)].correct.astype(float).mean(),atol=1e-12)
    schedule=old.support(eps[task],'P2','Transfer');bb=blocks[blocks.task.eq(task)].sort_values('end');assert bb.end.tolist()==[5,10,15,20] and bb.practiced.tolist()==[schedule[k-1]['context'] for k in [5,10,15,20]]
   w=OUT/f'conflict_wrong_cue/{r.run_id}';wp=pd.read_csv(w/'positions.csv.gz',float_precision='round_trip');assert len(wp)==336 and set(wp.stage)=={'before','after'};validate_labels(wp,eps,wrong=True);cells=pd.read_csv(w/'cells.csv',float_precision='round_trip');assert len(cells)==24 and cells.status.eq('completed').all() and cells.correct_CE_after.notna().all()
   rows.append(dict(seed=r.seed,arm=r.arm,dose=r.dose,tasks=12,correct_position_rows=168,wrong_position_rows=336,inner_updates=int(r.diagnostic_steps),optimizer_updates_in_audit=0))
 assert len(rows)==(18 if a.stage=='screen' else 56);write_frame(OUT/f'audit/{a.stage}_diagnostics_verified.csv',rows);dump(OUT/f'audit/{a.stage}_diagnostics_verified.json',dict(passed=True,runs=len(rows),tasks=12*len(rows),correct_position_rows=168*len(rows),wrong_position_rows=336*len(rows),all_four_probabilities_checked=True,all_inner_reset_and_step_counts=True,all_registered_context_target_labels=True,optimizer_updates=0,native_forwards=0));append('## Final diagnostic integrity audit\nPredecessor negative fixtures reject wrong context/target/position. Allnewinitializations×12tasks havecompletefinite20-stepinnerhistories, freshpolicy/Adam, exactsourceinitialweights/finaladaptedstates, registered5-stepcues, correctcriticalaccuracy/CE, andall4probabilities/margins/labelsforallcorrect/wrongpositionrows. Auditreadsarchivesonly,zerooptimizerupdates/zero nativeforwards; nomatchedbranchreplayed.')
 print('All saved diagnostic states/labels/probabilities PASSED',flush=True)
if __name__=='__main__':main()
