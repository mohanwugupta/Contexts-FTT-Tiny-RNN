"""Read-only analytic BPTT/Adam validation of every saved episode and target."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from experiments.meta_objective_balance.verify import unflatten
from experiments.heldout_context_meta import verify as historical
from experiments.heldout_context_meta_scaling_10k.verify import pack_state,verify_positions
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def query_weights(e,kind):
 if kind=='early':return np.ones((2,7))/14
 if kind=='novel':
  w=np.array(e['novel_mask'],float);return w/w.sum()
 w=np.ones((4,7))/28;m=np.array(e['critical_mask'],float)
 return .75*w+.25*m/m.sum() if m.any() else w

def validate_cues(e,actual):
 np.testing.assert_array_equal(actual,['ABC'.index(e['roles'][r]) for r in ['X']*10+['Y']*10])

def check_forward(state,paths,contexts,z,h):
 wi,wh,bi,bh,wo,bo=[v.double().numpy() for v in state.values()];hidden=np.zeros((len(paths),8));cc=np.array([CUES[c] for c in contexts]);hh=[];zz=[]
 for t in range(7):
  x=np.concatenate([np.eye(4)[[p[t] for p in paths]],cc],1);hidden=x@wi.T+hidden@wh.T+bi+bh;hh.append(hidden.copy());zz.append(hidden@wo.T+bo)
 np.testing.assert_allclose(h,np.stack(hh,1),atol=2e-5,rtol=2e-5);np.testing.assert_allclose(z,np.stack(zz,1),atol=2e-5,rtol=2e-5)

def analytic(state,paths,contexts,z,h,w):
 z=z.astype(float);h=h.astype(float);wi,wh,bi,bh,wo,bo=[v.double().numpy() for v in state.values()];pr=np.exp(z-z.max(-1,keepdims=True));pr/=pr.sum(-1,keepdims=True);y=np.array([p[1:] for p in paths]);dz=pr.copy()
 for i in range(len(paths)):dz[i,np.arange(7),y[i]]-=1
 dz*=w[:,:,None];gwo=np.einsum('ntc,nth->ch',dz,h);gbo=dz.sum((0,1));gi=np.zeros_like(wi);gh=np.zeros_like(wh);gb=np.zeros(8);future=np.zeros((len(paths),8));tokens=np.array([p[:-1] for p in paths]);cc=np.array([CUES[c] for c in contexts])
 for t in range(6,-1,-1):
  dh=dz[:,t]@wo+future@wh;xx=np.concatenate([np.eye(4)[tokens[:,t]],cc],1);gi+=dh.T@xx;gh+=dh.T@(h[:,t-1] if t else np.zeros_like(future));gb+=dh.sum(0);future=dh
 ce=np.log(np.exp(z-z.max(-1,keepdims=True)).sum(-1))+z.max(-1)-np.take_along_axis(z,y[:,:,None],2)[:,:,0]
 return np.concatenate([v.reshape(-1) for v in [gi,gh,gb,gb,gwo,gbo]]),float((ce*w).sum())

def meta_one(row):
 initialize_process();s=int(row['seed']);a=row['arm'];d=OUT/f'meta/s{s:02d}_{a}';template,opt,src=source(s);previous=pack_state(template,opt);eps=json.loads((OUT/f'registry/episodes_seed_{s}_{a}.json').read_text());tab=pd.read_csv(d/'episodes.csv');count=0;maxerror=0.;p0=torch.load(d/'e0.pt',weights_only=False);np.testing.assert_array_equal(pack_state(p0['model'],p0['optimizer']),previous)
 for chunk in sorted(d.glob('chunk_*.npz')):
  data=np.load(chunk)
  for j in range(len(data['combined_gradient'])):
   e=eps[count];k=count+1;assert e['episode']==k;np.testing.assert_array_equal(data['outer_before'][j],previous);validate_cues(e,data['actual_support_cues'][j]);assert np.all(data['early_inner_steps'][j]==10) and np.all(data['final_inner_steps'][j]==20)
   for part in ['early','familiar','novel']:
    early=part=='early';state=unflatten(data['early_model' if early else 'final_model'][j],template);paths=e['paths']['X']+(e['paths']['Y'] if part=='familiar' else []);contexts=[e['roles']['X']]*2+[e['roles']['Y']]*2 if part=='familiar' else [e['roles']['X' if early else 'Q']]*2
    check_forward(state,paths,contexts,data[part+'_logits'][j],data[part+'_hidden'][j]);g,ce=analytic(state,paths,contexts,data[part+'_logits'][j],data[part+'_hidden'][j],query_weights(e,part));saved=data[part+'_gradient'][j];maxerror=max(maxerror,float(abs(g-saved).max()));np.testing.assert_allclose(saved,g,rtol=2e-4,atol=1e-4);np.testing.assert_allclose(ce,tab.iloc[count][part+'_ce'],rtol=2e-5,atol=2e-5)
   if not e['critical_cells']:assert np.isnan(tab.iloc[count].critical_ce) and abs(tab.iloc[count].familiar_ce-tab.iloc[count].all_ce)<1e-7
   else:np.testing.assert_allclose(tab.iloc[count].familiar_ce,.75*tab.iloc[count].all_ce+.25*tab.iloc[count].critical_ce,atol=2e-6,rtol=2e-6)
   g=.2*data['early_gradient'][j]+.5*data['familiar_gradient'][j]+.3*data['novel_gradient'][j];np.testing.assert_allclose(g,data['combined_gradient'][j],atol=2e-6,rtol=3e-6);norm=float(tab.iloc[count].combined_norm);np.testing.assert_allclose(norm,np.linalg.norm(g),atol=2e-6,rtol=3e-6);historical.check_adam(previous,data['outer_after'][j],g,norm,960+k);previous=data['outer_after'][j];count+=1
  end=int(chunk.stem.split('_')[1]);cp=d/f'e{end}.pt';receipt=json.loads((d/f'e{end}.json').read_text());assert sha(cp)==receipt['sha256'] and sha(chunk)==receipt['chunk_sha'];b=torch.load(cp,weights_only=False);np.testing.assert_array_equal(pack_state(b['model'],b['optimizer']),previous);assert b['next_episode']==end+1 and all(float(st['step'])==960+end for st in b['optimizer']['state'].values())
 assert count==row['episodes'] and count*20==row['inner_updates'];return dict(seed=s,arm=a,episodes=count,query_gradients_checked=count*3,analytic_gradient_max_error=maxerror,optimizer_updates_in_audit=0,source_sha=sha(src))

def target_one(row):
 with bind(historical,OUT=OUT,initial=initial,verify_position_rows=verify_positions):return historical.verify_target(row)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);p.add_argument('--targets',action='store_true');p.add_argument('--receipt-suffix',default='');a=p.parse_args();kind='targets' if a.targets else 'meta';reg=pd.read_csv(OUT/f'registry/{a.stage}_{"targets" if a.targets else "meta_runs"}.csv');fn=target_one if a.targets else meta_one
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(fn,reg.to_dict('records')))
 write_frame(OUT/f'audit/{a.stage}_{kind}_verified{a.receipt_suffix}.csv',rows);dump(OUT/f'audit/{a.stage}_{kind}_verified{a.receipt_suffix}.json',dict(passed=True,runs=len(rows),optimizer_updates=0,full_record_audit=True));print(a.stage,kind,'verification PASSED',flush=True)
if __name__=='__main__':main()
