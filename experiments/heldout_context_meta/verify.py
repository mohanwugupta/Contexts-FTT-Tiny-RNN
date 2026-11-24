"""Read-only saved-gradient/Adam/state/position verification; no optimizer steps."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .protocol import queries
from .data import registry,directory
from experiments.context_experience_prior.verify import verify_position_rows,verify_component_rows
from experiments.meta_learning_scaling.target import validate_training_records

def adam_expected(before,gradient,norm,step):
 a=torch.as_tensor(before).clone();g=torch.as_tensor(gradient).clone();coef=torch.clamp(1./(torch.tensor(norm,dtype=g.dtype)+1e-6),max=1.);g.mul_(coef);m=a[1].lerp(g,.1);v=a[2].mul(.999).addcmul(g,g,value=.001);denom=(v.sqrt()/((1-.999**step)**.5)).add(1e-8);p=a[0].addcdiv(m,denom,value=-.001/(1-.9**step));return torch.stack([p,m,v]).numpy()
def check_adam(before,after,gradient,norm,step):np.testing.assert_allclose(after,adam_expected(before,gradient,norm,step),atol=2e-7,rtol=2e-6)
def flat(d):return np.concatenate([v.detach().reshape(-1).numpy() for v in d.values()])

def analytic_gradient(adapted,ep,prior,z,h):
 qs=queries(ep,prior);n=len(qs);z=z[:n].astype(float);h=h[:n].astype(float);wi,wh,bi,bh,wo,bo=[v.double().numpy() for v in adapted.values()];probs=np.exp(z-z.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True);y=np.array([ep['paths'][q['sequence']][1:] for q in qs]);dz=probs.copy()
 for i in range(n):dz[i,np.arange(7),y[i]]-=1
 dz/=n*7;gwo=np.einsum('ntc,nth->ch',dz,h);gbo=dz.sum((0,1));gi=np.zeros_like(wi);gh=np.zeros_like(wh);gb=np.zeros(8);future=np.zeros((n,8));contexts=np.array([CUES[q['context']] for q in qs]);tokens=np.array([ep['paths'][q['sequence']][:-1] for q in qs])
 for t in range(6,-1,-1):
  dh=dz[:,t]@wo+future@wh;xx=np.concatenate([np.eye(4)[tokens[:,t]],contexts],1);gi+=dh.T@xx;gh+=dh.T@(h[:,t-1] if t else np.zeros((n,8)));gb+=dh.sum(0);future=dh
 return np.concatenate([v.reshape(-1) for v in [gi,gh,gb,gb,gwo,gbo]])

def verify_meta_run(row):
 initialize_process();seed,prior,org,arm=[row[k] for k in ['seed','prior','organization','arm']];initial_state,_=initial(seed,prior,org);last=initial_state;checked=0;inner_count=0;maxerr=0
 for b in range(1,5):
  path=OUT/f'meta/s{seed:02d}_{prior}_{org}_{arm}_b{b}.pt';obj=torch.load(path,weights_only=False);rec=obj['records'];npz=np.load(path.with_suffix('.npz'));hist=npz['inner'];steps=npz['inner_steps'];outer=npz['outer'];osteps=npz['outer_steps'];grad=npz['query_gradient'];z=npz['query_logits'];h=npz['query_hidden'];ep=json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text())['episodes'];meta=pd.read_csv(path.with_name(path.stem+'_episodes.csv'));assert len(rec)==241 and len(meta)==240
  if last is not None:assert all(torch.equal(v,rec[0]['meta'][k]) for k,v in last.items())
  for i,r in enumerate(rec[1:]):
   if r['status']!='completed':assert r['meta'] is None;last=None;continue
   assert all(torch.equal(v,r['before'][k]) for k,v in last.items());np.testing.assert_array_equal(hist[i,0,0,:172],flat(r['before']));assert np.all(steps[i,0]==0) and np.all(hist[i,0,1:]==0);np.testing.assert_array_equal(steps[i],np.repeat(np.arange(21,dtype=np.float32)[:,None],8,axis=1));np.testing.assert_array_equal(hist[i,-1,0,:172],flat(r['adapted']));np.testing.assert_array_equal(hist[i,-1,0,172:],flat(r['adapted_policy']));np.testing.assert_array_equal(outer[i,0],flat(r['before']));np.testing.assert_array_equal(outer[i+1,0],flat(r['meta']));assert np.all(osteps[i+1]==r['episode']);check_adam(outer[i],outer[i+1],grad[i],float(meta.iloc[i].query_gradient_norm),r['episode']);expected=analytic_gradient(r['adapted'],ep[i],prior,z[i],h[i]);err=np.max(abs(expected-grad[i]));maxerr=max(maxerr,float(err));np.testing.assert_allclose(grad[i],expected,rtol=2e-4,atol=1e-4);assert bool(meta.iloc[i].query_unchanged) and bool(meta.iloc[i].policy_reset) and bool(meta.iloc[i].inner_optimizer_empty);last=r['meta'];checked+=1;inner_count+=20
 return dict(seed=seed,prior=prior,organization=org,arm=arm,outer_updates=checked,inner_updates=inner_count,analytic_gradient_max_error=maxerr)

def verify_target(row):
 initialize_process();p=ROOT/row['path'];assert sha(p)==row['sha256'];b=torch.load(p,weights_only=False);assert len(b['states'])==331;d=OUT/row['directory'];valid=0
 for k,st in enumerate(b['states']):
  assert st['update']==k
  if st['status']!='completed':assert st['model'] is None;continue
  valid+=1;assert torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],b['C_initial'])
  if k==0:
   init,_=initial(row['seed'],row['prior'],row['organization'],row['arm'],row['dose']);assert all(torch.equal(v,st['model'][n]) for n,v in init.items());assert not st['optimizer']['state'] and all(torch.count_nonzero(v)==0 for v in st['policy'].values())
  for state in st['optimizer']['state'].values():
   assert int(state['step'])==k
   if state['exp_avg'].shape==(8,7):assert torch.count_nonzero(state['exp_avg'][:,6])==torch.count_nonzero(state['exp_avg_sq'][:,6])==0
 pos=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip');good=pos[pos.status=='completed'];verify_position_rows(good);ev=pd.read_csv(d/'evaluations.csv',float_precision='round_trip');tr=pd.read_csv(d/'training.csv',float_precision='round_trip');validate_training_records(tr[tr.status=='completed'],ev);means=good.groupby(['update','cue']).agg(ce_pos=('ce','mean'),acc_pos=('correct','mean')).reset_index().merge(ev,on=['update','cue']);np.testing.assert_allclose(means.ce_pos,means.ce,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(means.acc_pos,means.accuracy,atol=1e-12,rtol=1e-12);dd=pd.read_csv(d/'diagnostics.csv');assert len(dd)==24 and dd.main_unchanged.all() and dd.main_start_hash.eq(dd.clone_start_hash).all();co=pd.read_csv(d/'mathematical_checks.csv.gz');assert len(co)==7*valid and co.passed.all();dc=pd.read_csv(d/'decomposition.csv.gz',float_precision='round_trip');verify_component_rows(dc);assert (dc.C_minus_avg_operator_error<=dc.C_minus_avg_allowance).all();return dict(run_id=row['run_id'],states=331,valid_states=valid,position_rows=len(pos),target_steps=int(tr.optimizer_step_executed.sum()),diagnostic_steps=int(dd.diagnostic_update_executed.sum()/2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--meta',action='store_true');a=p.parse_args();initialize_process();rows=[]
 if a.meta:
  reg=pd.read_csv(OUT/'registry/meta_runs.csv')
  with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(verify_meta_run,reg.to_dict('records')))
  write_frame(OUT/'audit/meta_verified.csv',rows);dump(OUT/'audit/meta_verified.json',dict(passed=True,runs=160,outer_updates=sum(x['outer_updates'] for x in rows),inner_updates=sum(x['inner_updates'] for x in rows),analytic_query_gradient=True,adam_from_query_gradient=True,inner_packed_histories=True,optimizer_updates=0))
 else:
  reg=pd.read_csv(OUT/'registry/new_targets.csv')
  with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(verify_target,reg.to_dict('records')))
  parent_verify();write_frame(OUT/'audit/targets_verified.csv',rows);dump(OUT/'audit/targets_verified.json',dict(passed=True,runs=320,states=sum(x['states'] for x in rows),valid_states=sum(x['valid_states'] for x in rows),position_rows=sum(x['position_rows'] for x in rows),target_updates=sum(x['target_steps'] for x in rows),diagnostic_updates=sum(x['diagnostic_steps'] for x in rows),predecessors_unchanged=True))
 print('Read-only verification passed',flush=True)
if __name__=='__main__':main()
