"""Read-only original target and corrected-v2 diagnostic state/probability checks."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .verify import one as target_one
from experiments.meta_objective_balance.diagnostic_checks import validate_labels
from experiments.heldout_context_meta import protocol as native
from experiments.heldout_context_meta.verify import flat

def diagnostic_one(row):
 r=type('Row',(),row);d=OUT/r.directory;eps=json.loads((OUT/f'registry/heldout_v2_seed_{r.seed}.json').read_text())['episodes'];p=pd.read_csv(d/'positions.csv',float_precision='round_trip');tasks=pd.read_csv(d/'tasks.csv',float_precision='round_trip');blocks=pd.read_csv(d/'blocks.csv',float_precision='round_trip');validate_labels(p,eps);h=np.load(d/'inner_history.npz');states=torch.load(d/'adapted_states.pt',weights_only=False);model,_=initial(r.seed,r.prior,r.organization,r.arm,r.dose);assert len(tasks)==12 and tasks.status.eq('completed').all() and len(p)==168 and len(blocks)==48;assert h['inner'].shape==(12,21,3,174) and np.isfinite(h['inner']).all();np.testing.assert_array_equal(h['steps'],np.broadcast_to(np.arange(21)[None,:,None],(12,21,8)));assert np.all(h['inner'][:,0,1:]==0) and np.all(h['inner'][:,0,0,172:]==0)
 for j in range(12):
  np.testing.assert_array_equal(h['inner'][j,0,0,:172],flat(model));np.testing.assert_array_equal(h['inner'][j,20,0,:172],flat(states[j]['after']));assert all(float(v['step'])==20 for v in states[j]['optimizer']['state'].values());g=p[p.task.eq(j)];rr=tasks[tasks.task.eq(j)].iloc[0];np.testing.assert_allclose(rr.ce,g.ce.mean(),atol=1e-12);np.testing.assert_allclose(rr.conflict_accuracy,g[g.position.eq(3)].correct.astype(float).mean(),atol=1e-12);schedule=native.support(eps[j],'P2','Transfer');b=blocks[blocks.task.eq(j)].sort_values('end');assert b.end.tolist()==[5,10,15,20] and b.practiced.tolist()==[schedule[k-1]['context'] for k in [5,10,15,20]]
 w=OUT/f'conflict_wrong_cue/{r.run_id}';wp=pd.read_csv(w/'positions.csv.gz',float_precision='round_trip');validate_labels(wp,eps,wrong=True);assert len(wp)==336;return dict(seed=r.seed,arm=r.arm,dose=r.dose,tasks=12,correct_positions=168,wrong_positions=336,optimizer_updates_in_audit=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostics',action='store_true');a=p.parse_args();label='heldout' if a.diagnostics else 'targets';reg=pd.read_csv(OUT/f'registry/{label}.csv')
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(diagnostic_one if a.diagnostics else target_one,reg.to_dict('records')))
 assert len(rows)==(78 if a.diagnostics else 72);write_frame(OUT/f'audit/{label}_verified.csv',rows);dump(OUT/f'audit/{label}_verified.json',dict(passed=True,runs=len(rows),all_four_probabilities=True,all_original_targets_and_contexts=True,optimizer_updates=0));print(label+' verification PASSED',flush=True)
if __name__=='__main__':main()
