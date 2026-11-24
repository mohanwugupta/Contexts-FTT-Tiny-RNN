import argparse,concurrent.futures,multiprocessing,traceback
from .common import *
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.heldout_context_meta import target as native

def configure():initialize_process();torch.set_num_threads(json.loads((OUT/'audit/benchmark.json').read_text())['threads'])
def one(job):
 seed,src,s=job;configure()
 if src=='Fresh' and s==2:
  from .reuse_audit import matches_older_control
  model,_=initial(seed,'P2','I',src,s)
  if matches_older_control(seed,model):raise RuntimeError('Exact legacy control match: register archived reference instead of running target')
 dest=OUT/f'target/s{seed:02d}_P2_I_{src}_d{s}';assert not dest.exists(),'Existing attempt; no automatic retry'
 try:
  with bind(native,OUT=OUT,initial=initial,codes=codes,initialize_process=configure):r=native.target_run((seed,'P2','I',src,s))
  return dict(r,source=src,scale=s,reused=False)
 except Exception as e:
  tb=e.__traceback__;frame=None
  while tb:
   if tb.tb_frame.f_code.co_name=='target_run':frame=tb.tb_frame.f_locals
   tb=tb.tb_next
  frame=frame or {};m=frame.get('m');o=frame.get('o');n=int(max([float(v['step']) for v in o.state.values()]+[0])) if o is not None else 0;dest.mkdir(exist_ok=True);atomic_save(dest/'failure_state.pt',dict(model=None if m is None else m.state_dict(),optimizer=None if o is None else o.state_dict(),states=frame.get('states'),training=frame.get('training'),evaluations=frame.get('evals'),positions=frame.get('positions')));r=dict(seed=seed,arm=src,dose=s,source=src,scale=s,status='blocked_numerical_or_native_integrity',directory=str(dest.relative_to(OUT)),target_updates=n,diagnostic_updates=int(frame.get('dn',0)),reused=False,error=repr(e),traceback=traceback.format_exc());dump(dest/'failure.json',r);return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','replicate'],required=True);stage=p.parse_args().stage;assert json.loads((OUT/'audit/source_audit.json').read_text())['passed']
 if stage=='screen':jobs=[(s,a,k) for s in range(6) for a in ['Fresh','U','P'] for k in SCALES if k!=2]
 else:
  sel=json.loads((OUT/'audit/selection.json').read_text());assert sha(OUT/'SCALE_SELECTION.md')==sel['decision_sha'];jobs=[(s,'Fresh',k) for s in range(6,20) for k in sorted(set([2]+sel['selected']))]
 dump(OUT/f'audit/{stage}_started.json',dict(jobs=jobs,code=codes(),time=time.time()));rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(one,jobs):rows.append(r);print(stage+' '+str(len(rows))+'/'+str(len(jobs))+' '+r['status'],flush=True)
 if stage=='screen':rows+=pd.read_csv(OUT/'registry/reused_scale2.csv').to_dict('records')
 write_frame(OUT/f'registry/{stage}_runs.csv',rows);dump(OUT/f'audit/{stage}_complete.json',dict(passed=True,cells=len(rows),new_runs=len(jobs),target_updates=sum(r['target_updates'] for r in rows if not r['reused']),clones=sum(r['diagnostic_updates'] for r in rows if not r['reused']),blocked=sum(r['status']!='completed' for r in rows),no_meta_updates=True));append('## '+stage+' execution complete\nAll registered cells retained, including failures; exactscale2 references are reused, not independent replication. No meta-training, projection during target, normalization or numerical rescue.')
if __name__=='__main__':main()
