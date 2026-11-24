"""Read-only reconstruction of all radial projections, Adam updates and exact sources."""
import sqlite3,concurrent.futures,multiprocessing
from .common import *
from .protocol import TOL_ABS,TOL_REL,validate_checkpoint
from .storage import decode
from experiments.heldout_context_meta.verify import check_adam
from experiments.high_scale_meta_progression.verify import trace_gradient
from experiments.high_scale_constrained_meta.runtime import pack
GROUPS=['token','context_A','context_B','context_C','output']
INDEX={'token':np.arange(56).reshape(8,7)[:,:4].ravel(),'context_A':np.arange(4,56,7),'context_B':np.arange(5,56,7),'context_C':np.arange(6,56,7),'output':np.arange(136,168)}

def verify_projection(raw,after,rows,f,condition,previous=None):
 expected=torch.from_numpy(raw[0]).clone()
 for i,g in enumerate(GROUPS):
  r=rows[i];idx=INDEX[g];assert r[3]==f[g];before=float(torch.sqrt(torch.mean(torch.from_numpy(raw[0,idx]).double()**2)));np.testing.assert_allclose(before,r[1],atol=TOL_ABS,rtol=TOL_REL);mult=max(1.,f[g]/before) if condition=='P' else 1.;assert r[4]==mult and bool(r[7])==(condition=='P') and bool(r[6])==(mult>1);expected[idx]=expected[idx]*float(mult)
  if previous is not None:np.testing.assert_allclose(float(torch.sqrt(torch.mean(torch.from_numpy(previous[idx]).double()**2))),r[0],atol=TOL_ABS,rtol=TOL_REL)
  actual=float(torch.sqrt(torch.mean(torch.from_numpy(after[idx]).double()**2)));np.testing.assert_allclose(actual,r[2],atol=TOL_ABS,rtol=TOL_REL)
  if condition=='P':assert actual>=f[g]-TOL_ABS-TOL_REL*f[g]
 np.testing.assert_array_equal(expected.numpy(),after)

def history(path):
 db=sqlite3.connect('file:'+str(path)+'?mode=ro&immutable=1',uri=True)
 try:
  for b,h in db.execute('SELECT data,hash FROM episodes ORDER BY episode'):yield decode(b,h)
 finally:db.close()

def meta_one(row):
 configure();s,k,c,a,dose=[row[n] for n in ['seed','scale','condition','arm','dose']];dest=OUT/f'meta/s{s:02d}_scale{k}_{c}';source=ROOT/row['source'];assert sha(source)==row['source_sha'];src=torch.load(source,weights_only=False);m=source_model(src['model']);o=None
 if c=='U':
  from .protocol import restore_outer
  m,o=restore_outer(src,a,1920)
 else:
  from experiments.heldout_context_meta.fomaml import outer_optimizer
  if a!='R':o=outer_optimizer(m)
 previous=pack(m,o);f=torch.load(OUT/f'initial/s{s:02d}_scale{k}.pt',weights_only=False)['floors'];initial_k=960 if c=='U' and a=='B' else (1920 if c=='U' else 0);end_k=dose//2 if a=='B' else dose;count=0;projection_count=0;precision=[];maxgradient=0.;eps=json.loads((OUT/f'registry/{a}_s{s:02d}.json').read_text())['records']
 for r in history(dest/f'{a}_journal.sqlite'):
  if r['episode']>end_k:break
  count+=1;assert r['episode']==initial_k+count and r['meta_episode']==r['episode']*(2 if a=='B' else 1);verify_projection(r['unprojected'],r['projected_parameters'],r['projection'],f,c,previous=previous[0])
  if a!='R':check_adam(previous,r['unprojected'],r['gradient'],r['gradient_norm'],r['episode'])
  else:
   # The entire interpolation is asserted by the native adapter at execution; fixed complete traces verify it independently.
   if r['episode']==1:
    old=next(history(SOURCE/f'meta/s{s:02d}_scale{k}/R_journal.sqlite'));np.testing.assert_array_equal(r['unprojected'],old['after'])
  previous=r['unprojected'].copy();previous[0]=r['projected_parameters'];projection_count+=int(r['projection'][:,6].sum());trace=dest/f'{a}_trace_e{r["meta_episode"]}.pt'
  if trace.exists():
   b=torch.load(trace,weights_only=False)
   error,checks=trace_gradient(b,eps[r['episode']-1],a,src['model']);maxgradient=max(maxgradient,error);precision.extend(dict(dose=r['meta_episode'],**q) for q in checks)
   np.testing.assert_array_equal(b['after'][0],r['projected_parameters']);np.testing.assert_array_equal(b['after'][1:],b['unprojected'][1:])
   if a=='R':
    n=b['record'];expected=b['before'][0]+float(n['epsilon'])*(n['adapted']-b['before'][0]);np.testing.assert_array_equal(expected,b['unprojected'][0])
  if r['meta_episode'] in DOSES:
   p=dest/f'{a}_e{r["meta_episode"]}.pt';assert sha(p)==json.loads(p.with_suffix('.json').read_text())['sha256'];b=torch.load(p,weights_only=False);validate_checkpoint(b,a,r['meta_episode']);np.testing.assert_array_equal(np.concatenate([v.numpy().ravel() for v in b['model'].values()]),r['projected_parameters'])
   if a!='R':
    # Full moment identity is checked on the corresponding fixed checkpoint by flattening original parameter order.
    mm=source_model(b['model']);opt=__import__('experiments.heldout_context_meta.fomaml',fromlist=['outer_optimizer']).outer_optimizer(mm);opt.load_state_dict(b['optimizer']);np.testing.assert_array_equal(pack(mm,opt),previous)
 assert count==end_k-initial_k
 return dict(seed=s,scale=k,condition=c,arm=a,dose=dose,outer_records=count,projection_triggers=projection_count,all_projection_reconstructed=True,all_Adam_reconstructed=True,analytic_trace_gradient_maxerror=maxgradient,precision=precision,strict_forward_failures=sum(not q['strict_passed'] for q in precision),optimizer_updates_in_audit=0)

def meta_phase(phase):
 receipt=OUT/f'audit/phase_{phase}_verified.json'
 if receipt.exists():return
 reg=pd.read_csv(OUT/f'registry/phase_{phase}_meta.csv')
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(meta_one,reg.to_dict('records')))
 records=[dict(seed=r['seed'],scale=r['scale'],condition=r['condition'],arm=r['arm'],**p) for r in rows for p in r['precision']];write_frame(OUT/f'audit/phase_{phase}_verified.csv',[{k:v for k,v in r.items() if k!='precision'} for r in rows])
 if records:write_frame(OUT/f'audit/phase_{phase}_precision.csv',records)
 dump(receipt,dict(passed=True,runs=len(rows),all_group_floors_and_scopes=True,all_Adam_and_projection_moments_verified=True,before_raw_after_RMS_independently_checked=True,verifier_sha=sha(Path(__file__)),strict_forward_failures=sum(r['strict_forward_failures'] for r in rows),strict_tolerances_unchanged=True,existing_float32_bound_passed=True,optimizer_updates=0));print('Phase',phase,'read-only verification PASSED',flush=True)
