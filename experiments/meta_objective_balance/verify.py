"""Saved-data analytic gradients/Adam and inherited target checks; no training."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .protocol import ARMS,critical_mask,old
from experiments.heldout_context_meta import verify as historical
from experiments.heldout_context_meta_scaling_10k.verify import pack_state,verify_positions
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def unflatten(v,template):
 d={};i=0
 for k,t in template.items():d[k]=torch.from_numpy(v[i:i+t.numel()].reshape(t.shape).copy());i+=t.numel()
 assert i==len(v);return d

def analytic(adapted,ep,z,h,critical=False):
 qs=old.queries(ep,'P2');n=len(qs);z=z.astype(float);h=h.astype(float);wi,wh,bi,bh,wo,bo=[v.double().numpy() for v in adapted.values()];pr=np.exp(z-z.max(-1,keepdims=True));pr/=pr.sum(-1,keepdims=True);y=np.array([ep['paths'][q['sequence']][1:] for q in qs]);dz=pr.copy()
 for i in range(n):dz[i,np.arange(7),y[i]]-=1
 mask=critical_mask(ep,qs).astype(float) if critical else np.ones((n,7));dz*=mask[:,:,None]/mask.sum();gwo=np.einsum('ntc,nth->ch',dz,h);gbo=dz.sum((0,1));gi=np.zeros_like(wi);gh=np.zeros_like(wh);gb=np.zeros(8);future=np.zeros((n,8));contexts=np.array([CUES[q['context']] for q in qs]);tokens=np.array([ep['paths'][q['sequence']][:-1] for q in qs])
 for t in range(6,-1,-1):
  dh=dz[:,t]@wo+future@wh;xx=np.concatenate([np.eye(4)[tokens[:,t]],contexts],1);gi+=dh.T@xx;gh+=dh.T@(h[:,t-1] if t else np.zeros((n,8)));gb+=dh.sum(0);future=dh
 return np.concatenate([v.reshape(-1) for v in [gi,gh,gb,gb,gwo,gbo]])

def meta_one(row):
 initialize_process();s=int(row['seed']);a=row['arm'];d=OUT/f'meta/s{s:02d}_{a}';template,opt,src=source(s);previous=pack_state(template,opt);pairs=json.loads((OUT/f'registry/pairs_seed_{s}.json').read_text())['pairs'];lam,beta=ARMS[a];count=0;maxerror=0.;table=pd.read_csv(d/'episodes.csv');assert len(table)==len(pairs)
 p0=torch.load(d/'p0.pt',weights_only=False);np.testing.assert_array_equal(pack_state(p0['model'],p0['optimizer']),previous)
 for chunk in sorted(d.glob('chunk_*.npz')):
  data=np.load(chunk)
  for j in range(len(data['combined_gradient'])):
   k=count+1;pair=pairs[count];assert pair['pair']==k;np.testing.assert_array_equal(data['outer_before'][j],previous)
   for kind in ['shared','conflict']:
    ep=pair[kind];ad=unflatten(data[kind+'_adapted'][j],template);z=data[kind+'_query_logits'][j];h=data[kind+'_query_hidden'][j]
    for crit in ([False,True] if kind=='conflict' else [False]):
     g=analytic(ad,ep,z,h,crit);saved=data[kind+('_gradient_critical' if crit else '_gradient_all')][j];maxerror=max(maxerror,float(abs(g-saved).max()));np.testing.assert_allclose(saved,g,rtol=2e-4,atol=1e-4)
    assert data[kind+'_inner_ce'][j].shape==(20,) and np.isfinite(data[kind+'_inner_ce'][j]).all() and np.isfinite(data[kind+'_inner_gradient_norm'][j]).all()
   gs=data['shared_gradient_all'][j];ga=data['conflict_gradient_all'][j];gc=data['conflict_gradient_critical'][j];g=(1-lam)*gs+lam*((1-beta)*ga+beta*gc);np.testing.assert_allclose(data['combined_gradient'][j],g,atol=2e-6,rtol=3e-6);norm=float(table.iloc[count].combined_norm);np.testing.assert_allclose(norm,np.linalg.norm(g),atol=2e-6,rtol=3e-6);historical.check_adam(previous,data['outer_after'][j],g,norm,960+k);previous=data['outer_after'][j];count+=1
  end=int(chunk.stem.split('_')[1]);cp=d/f'p{end}.pt';r=json.loads((d/f'p{end}.json').read_text());assert sha(cp)==r['sha256'] and sha(chunk)==r['chunk_sha'];b=torch.load(cp,weights_only=False);np.testing.assert_array_equal(pack_state(b['model'],b['optimizer']),previous);assert b['next_pair']==end+1 and all(float(st['step'])==960+end for st in b['optimizer']['state'].values())
 assert count==row['pairs'] and count*40==row['inner_updates'];return dict(seed=s,arm=a,pairs=count,query_gradients_checked=count*3,analytic_gradient_max_error=maxerror,optimizer_updates_in_audit=0,source_sha=sha(src))

def target_one(row):
 with bind(historical,OUT=OUT,initial=initial,verify_position_rows=verify_positions):return historical.verify_target(row)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);p.add_argument('--targets',action='store_true');a=p.parse_args();kind='targets' if a.targets else 'meta';reg=pd.read_csv(OUT/f'registry/{a.stage}_{"targets" if a.targets else "meta_runs"}.csv');fn=target_one if a.targets else meta_one
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(fn,reg.to_dict('records')))
 write_frame(OUT/f'audit/{a.stage}_{kind}_verified.csv',rows);dump(OUT/f'audit/{a.stage}_{kind}_verified.json',dict(passed=True,runs=len(rows),optimizer_updates=0,full_record_audit=True));print(a.stage,kind,'verification PASSED',flush=True)
if __name__=='__main__':main()
