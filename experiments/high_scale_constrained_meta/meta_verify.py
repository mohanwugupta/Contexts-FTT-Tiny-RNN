"""Saved-data native interpolation/Adam, radial projection and analytic query audit."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .runtime import pack
from .train import epsilon
from experiments.meta_objective_balance.verify import unflatten
from experiments.heldout_context_meta.verify import analytic_gradient,check_adam
from experiments.heldout_context_meta.protocol import support,queries
from experiments.structured_overlap_meta.verify import check_forward

def project_array(values,template,floors,enabled):
 d=unflatten(values,template);arrays=[v.numpy() for v in d.values()];w,rr,b1,b2,wo,bo=arrays
 if enabled:
  for name,x in [('token',w[:,:4]),('context_A',w[:,4]),('context_B',w[:,5]),('context_C',w[:,6]),('output',wo)]:
   s=float(np.sqrt(np.mean(x.astype(float)**2)))
   if not np.isfinite(s) or s==0:raise ValueError('Invalid independent projection norm')
   x*=np.float32(max(1.,floors[name]/s))
 return np.concatenate([v.reshape(-1) for v in arrays])

def one(job):
 s,a,phase=job;initialize_process();stem=phase.lower();dest=OUT/f'meta/s{s:02d}_{a}';reg=json.loads((OUT/f'registry/{stem}_s{s:02d}.json').read_text())['episodes'];b=torch.load(dest/f'{stem}_e0.pt',weights_only=False);template=b['model'];opt=b['optimizer'];from .runtime import outer_optimizer
 m=source_model(template);o=None if opt is None else outer_optimizer(m)
 if o is not None:o.load_state_dict(opt)
 previous=pack(m,o);initial=torch.load(OUT/f'initial/s{s:02d}_High.pt',weights_only=False);floors=initial['floors'];tab=pd.read_csv(dest/f'{stem}_episodes.csv',float_precision='round_trip');pr=pd.read_csv(dest/f'{stem}_projections.csv',float_precision='round_trip');assert len(tab)==1920 and len(pr)==9600;count=0;max_grad=0.
 for path in sorted(dest.glob(stem+'_chunk*.npz')):
  data=np.load(path)
  for j in range(60):
   e=reg[count];k=count+1;np.testing.assert_array_equal(data['before'][j],previous);np.testing.assert_array_equal(data['inner_steps_array'][j],np.full(8,20));schedule=e['support_P2'] if phase=='Reptile' else support(e,'P2','Transfer');np.testing.assert_array_equal(data['support_context'][j],['ABC'.index(u['context']) for u in schedule]);np.testing.assert_array_equal(data['support_sequence'][j],[u['sequence'] for u in schedule]);assert np.isfinite(data['inner_total_loss'][j]).all();np.testing.assert_allclose(data['inner_total_loss'][j],data['inner_ce_before'][j]+.2*data['inner_execution_loss'][j],atol=2e-6,rtol=1e-6)
   if phase=='Reptile':
    expected=previous[0]+np.float32(epsilon(k))*(data['adapted'][j]-previous[0]);np.testing.assert_allclose(data['unprojected'][j,0],expected,atol=2e-7,rtol=2e-6);assert np.isnan(tab.iloc[count].query_ce) and np.isnan(data['gradient'][j]).all() and np.count_nonzero(data['unprojected'][j,1:])==0
   else:
    st=unflatten(data['adapted'][j],template);qs=queries(e,'P2');n=len(qs);z=data['query_logits'][j,:n];h=data['query_hidden'][j,:n];check_forward(st,[e['paths'][v['sequence']] for v in qs],[v['context'] for v in qs],z,h);g=analytic_gradient(st,e,'P2',z,h);saved=data['gradient'][j];max_grad=max(max_grad,float(abs(g-saved).max()));np.testing.assert_allclose(saved,g,rtol=2e-4,atol=1e-4);norm=float(tab.iloc[count].query_gradient_norm);check_adam(previous,data['unprojected'][j],saved,norm,k);ce=np.mean([core.score(z[ii],np.array(e['paths'][q['sequence']][1:]))['ce'].mean() for ii,q in enumerate(qs)]);np.testing.assert_allclose(ce,tab.iloc[count].query_ce,atol=2e-5,rtol=2e-5)
   expected=project_array(data['unprojected'][j,0],template,floors,a=='P');np.testing.assert_allclose(data['after'][j,0],expected,atol=2e-7,rtol=2e-6);np.testing.assert_array_equal(data['after'][j,1:],data['unprojected'][j,1:]);state=unflatten(data['after'][j,0],template);w,R,b1,b2,wo,bo=[v.double().numpy() for v in state.values()];S=np.eye(8);delta=[];center=[]
   for t in range(7):
    v=wo@S@(w[:,4]-w[:,5]);delta.append(np.linalg.norm(v));center.append(np.linalg.norm(v-v.mean()));S=np.eye(8)+R@S
   np.testing.assert_allclose(data['scale_after_effective_contrast'][j],delta,atol=1e-10,rtol=1e-10);np.testing.assert_allclose(data['scale_after_centered_contrast'][j],center,atol=1e-10,rtol=1e-10);x=pr[pr.episode.eq(k)];assert len(x)==5
   if a=='P':assert (x.after_rms>=x.floor-1e-7-2e-7*x.floor).all()
   else:np.testing.assert_array_equal(data['after'][j],data['unprojected'][j]);assert x.multiplier.eq(1).all()
   previous=data['after'][j];count+=1
  end=int(path.stem.split('chunk')[1]);cp=dest/f'{stem}_e{end}.pt';rr=json.loads((dest/f'{stem}_e{end}.json').read_text());assert sha(cp)==rr['checkpoint_sha'] and sha(path)==rr['chunk_sha'];bundle=torch.load(cp,weights_only=False);mm=source_model(bundle['model']);oo=None if phase=='Reptile' else outer_optimizer(mm)
  if oo is not None:oo.load_state_dict(bundle['optimizer']);assert all(float(v['step'])==end for v in oo.state.values())
  np.testing.assert_array_equal(pack(mm,oo),previous);assert bundle['next_episode']==end+1
 assert count==1920;return dict(seed=s,arm=a,phase=phase,episodes=count,inner_steps=count*20,outer_steps=count,max_analytic_gradient_error=max_grad,all_moments_preserved=True,all_effective_contrasts_verified=True,optimizer_updates_in_verification=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['Reptile','FOMAML'],required=True);phase=p.parse_args().phase;assert json.loads((OUT/f'audit/{phase.lower()}_complete.json').read_text())['passed'];jobs=[(s,a,phase) for s in SEEDS for a in ['U','P']]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,jobs))
 write_frame(OUT/f'audit/{phase.lower()}_verified.csv',rows);dump(OUT/f'audit/{phase.lower()}_verified.json',dict(passed=True,runs=12,episodes=23040,inner_updates=460800,outer_updates=23040,verification_optimizer_steps=0));append('## '+phase+' saved-data audit passed\nAll23040outer transitions checked independently, original interpolation or native query-gradient/Adam mathematics, post-outer radial projections/moments/steps, actual20-step support/objectives, all7effectivecontrast positions and atomic checkpoint hashes. No training replay in audit.');print(phase+' saved-data verification PASSED',flush=True)
if __name__=='__main__':main()
