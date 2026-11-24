"""Read-only source, gradient, Adam, fork and frozen-table evidence checks."""
import sqlite3,concurrent.futures,multiprocessing
from .common import *
from .train import verify_saved_checkpoint
from .target import validate_frozen_grid,validate_training_records
from .precision import bounded_forward
from experiments.heldout_context_meta_scaling_10k.storage import decode
from experiments.heldout_context_meta.verify import check_adam,analytic_gradient
from experiments.meta_objective_balance.verify import unflatten,analytic as balanced_analytic
from experiments.structured_overlap_meta.verify import analytic as overlap_analytic,query_weights,check_forward

def history(path):
 db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
 try:return [decode(b,h) for b,h in db.execute('SELECT data,hash FROM episodes ORDER BY episode')]
 finally:db.close()

def trace_gradient(b,e,arm,template):
 r=b['record'];error=0.;precision=[]
 if arm=='T':
  g=analytic_gradient(unflatten(r['adapted'],template),e,'P2',r['query_logits'],r['query_hidden']);np.testing.assert_allclose(g,r['gradient'],rtol=2e-4,atol=1e-4);error=float(abs(g-r['gradient']).max())
 elif arm=='B':
  for part in ['shared','conflict']:
   q=r[part];st=unflatten(q['adapted'],template)
   for critical in ([False,True] if part=='conflict' else [False]):
    g=balanced_analytic(st,e[part],q['query_logits'],q['query_hidden'],critical);saved=q['gradient_critical' if critical else 'gradient_all'];np.testing.assert_allclose(g,saved,rtol=2e-4,atol=1e-4);error=max(error,float(abs(g-saved).max()))
 elif arm.startswith('O'):
  for part in ['early','familiar','novel']:
   early=part=='early';st=unflatten(r['early_model' if early else 'final_model'],template);paths=e['paths']['X']+(e['paths']['Y'] if part=='familiar' else []);contexts=([e['roles']['X']]*2+[e['roles']['Y']]*2) if part=='familiar' else [e['roles']['X' if early else 'Q']]*2;certificate=bounded_forward(st,paths,contexts,r[part+'_logits'],r[part+'_hidden']);precision.append(dict(part=part,**certificate));g,ce=overlap_analytic(st,paths,contexts,r[part+'_logits'],r[part+'_hidden'],query_weights(e,part));np.testing.assert_allclose(g,r[part+'_gradient'],rtol=2e-4,atol=1e-4);np.testing.assert_allclose(ce,r[part+'_ce'],rtol=2e-5,atol=2e-5);error=max(error,float(abs(g-r[part+'_gradient']).max()))
 for j,h in enumerate(b['inner_complete_states']):
  step=j%20+1;assert len(h['optimizer']['state'])==8 and all(float(v['step'])==step for v in h['optimizer']['state'].values())
 return error,precision

def meta_one(row):
 configure();s,k,a=row['seed'],row['scale'],row['arm'];d=OUT/f'meta/s{s:02d}_scale{k}';b=verify_saved_checkpoint(d/f'{a}_e0.pt');src=ROOT/b['source'];assert sha(src)==b['source_sha'];template=torch.load(src,weights_only=False)['model'];assert all(torch.equal(v,b['model'][n]) for n,v in template.items());assert b['optimizer'] is None if a=='R' else not b['optimizer']['state'];prev=None;rs=history(d/f'{a}_journal.sqlite');assert len(rs)==(960 if a=='B' else 1920);eps=json.loads((OUT/f'registry/{a}_s{s:02d}.json').read_text())['records'];maxerror=0.;precision=[]
 for index,r in enumerate(rs,1):
  assert r['episode']==index and r['meta_episode']==index*(2 if a=='B' else 1) and np.isfinite(r['after']).all()
  if prev is not None:np.testing.assert_array_equal(r['before'],prev)
  if a!='R':check_adam(r['before'],r['after'],r['gradient'],r['combined_norm'],index)
  prev=r['after'];dose=r['meta_episode'];trace=d/f'{a}_trace_e{dose}.pt'
  if trace.exists():
   err,certificates=trace_gradient(torch.load(trace,weights_only=False),eps[index-1],a,template);maxerror=max(maxerror,err);precision.extend(dict(dose=dose,**q) for q in certificates)
  if dose in [240,960,1920]:
   cp=verify_saved_checkpoint(d/f'{a}_e{dose}.pt');np.testing.assert_array_equal(np.concatenate([v.numpy().ravel() for v in cp['model'].values()]),prev[0]);assert cp['meta_episode']==dose and cp['source_sha']==b['source_sha']
 return dict(seed=s,scale=k,arm=a,episodes=sum(r['episodes'] for r in rs),inner_updates=sum(r['inner_updates'] for r in rs),outer_updates=sum(r['outer_updates'] for r in rs),interpolations=sum(r['interpolations'] for r in rs),analytic_trace_gradient_max_error=maxerror,optimizer_updates_in_verification=0,source_sha=b['source_sha'],initial_fork_identical=True,precision_checks=precision,strict_forward_failures=sum(not q['strict_passed'] for q in precision))

def meta_stage(label):
 receipt=OUT/f'audit/{label}_verified.json'
 if receipt.exists():assert json.loads(receipt.read_text())['passed'];return
 reg=pd.read_csv(OUT/f'registry/{label}.csv')
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(meta_one,reg.to_dict('records')))
 precision=[dict(seed=r['seed'],scale=r['scale'],arm=r['arm'],**q) for r in rows for q in r.get('precision_checks',[])];failures=sum(r.get('strict_forward_failures',0) for r in rows)
 write_frame(OUT/f'audit/{label}_verified.csv',[{k:v for k,v in r.items() if k!='precision_checks'} for r in rows])
 if precision:write_frame(OUT/f'audit/{label}_precision_checks.csv',precision)
 dump(receipt,dict(passed=True,runs=len(rows),all_outer_steps_reconstructed=True,trace_query_gradients_analytic=True,meta_optimizer_updates=0,strict_forward_all_passed=failures==0,strict_forward_failures=failures,unchanged_predecessor_bound_passed=True,strict_thresholds_unchanged=True,precision_caveat='Fixed strict forward exceptions retained; separate pre-existing bound is only a consistency certificate'));print(label+' read-only gradient/Adam/bound verification PASSED; strict forward exceptions='+str(failures),flush=True)

def target_one(r):
 configure();d=OUT/r['directory'];assert sha(ROOT/r['path'])==r['sha256'];b=torch.load(ROOT/r['path'],weights_only=False);states=b['states'];assert len(states)==r['total']+1;src=torch.load(ROOT/b['source'],weights_only=False)['model'];assert all(torch.equal(v,states[0]['model'][k]) for k,v in src.items())
 for k,st in enumerate(states):
  assert st['update']==k and torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],b['C_initial'])
  assert not st['optimizer']['state'] if k==0 else len(st['optimizer']['state'])==8 and all(float(v['step'])==k for v in st['optimizer']['state'].values())
 e=pd.read_csv(d/'evaluations.csv',float_precision='round_trip');p=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip');validate_frozen_grid(e,p,r['total']);validate_training_records(pd.read_csv(d/'training.csv',float_precision='round_trip'),e);checks=pd.read_csv(d/'mathematical_checks.csv.gz');assert checks.passed.all() and len(checks)==7*len(states);return dict(seed=r['seed'],scale=r['scale'],phase=r['phase'],dose=r['dose'],states=len(states),all_four_probabilities=True,Adam_and_C_checked=True)

def main():
 meta_stage('reptile');meta_stage('fomaml')
