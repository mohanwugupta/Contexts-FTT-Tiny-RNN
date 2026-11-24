"""Fixed1920-episode stages, cooperative failure stop, atomic60-episode chunks."""
import argparse,concurrent.futures,multiprocessing,traceback
from .common import *
from .runtime import episode,outer_optimizer,scale_metrics
from .validation import validate_record

def jobs_for(phase):
 if phase not in ['Reptile','FOMAML']:raise ValueError('Original two phases only')
 return [(s,a,phase) for s in SEEDS for a in ['U','P']]
def target_jobs():return [(s,'P2','I',a+'-'+p,d) for s in SEEDS for a in ['U','P'] for p in ['R','F'] for d in [240,960,1920]]
def epsilon(k):
 if k not in range(1,1921):raise ValueError('Fixed Reptile budget')
 return .1+(.01-.1)*(k-1)/239 if k<=240 else .01

def checkpoint(m,o,s,a,phase,k,regsha):return dict(model=copy.deepcopy(m.state_dict()),optimizer=None if o is None else copy.deepcopy(o.state_dict()),seed=s,arm=a,phase=phase,episode=k,next_episode=k+1,registry_sha=regsha,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),code=codes())

def run(job):
 s,a,phase=job;initialize_process();torch.set_num_threads(json.loads((OUT/'audit/meta_benchmark.json').read_text())['selected']['threads']);stem=phase.lower();dest=OUT/f'meta/s{s:02d}_{a}';dest.mkdir(exist_ok=True);started=dest/f'{stem}_started.json';assert not started.exists(),'Existing attempt cannot be replayed';stop=OUT/'audit/SCIENTIFIC_STOP.json'
 if stop.exists():return dict(seed=s,arm=a,phase=phase,status='not_started_program_stop',episodes=0,inner_updates=0,outer_updates=0)
 init=torch.load(OUT/f'initial/s{s:02d}_High.pt',weights_only=False);floors=init['floors'];path=OUT/f'initial/s{s:02d}_High.pt' if phase=='Reptile' else dest/'reptile_e1920.pt';b=torch.load(path,weights_only=False);m=source_model(b['model']);o=None if phase=='Reptile' else outer_optimizer(m);reg=OUT/f'registry/{stem}_s{s:02d}.json';regsha=sha(reg);eps=json.loads(reg.read_text())['episodes'];assert len(eps)==1920;dump(started,dict(source=str(path.relative_to(ROOT)),source_sha=sha(path),registry_sha=regsha,code=codes(),time=time.time()));atomic_save(dest/f'{stem}_e0.pt',checkpoint(m,o,s,a,phase,0,regsha));write_frame(dest/f'{stem}_initial_scale.csv',[dict(seed=s,arm=a,phase=phase,group=g,rms=v,floor=floors[g],recurrent_spectral_norm=scale_metrics(m)['recurrent_spectral_norm']) for g,v in zip(floors,scale_metrics(m)['rms'])]);records=[];scalars=[];projections=[];done=inner=outer=0;t=time.monotonic()
 try:
  for e in eps:
   if stop.exists():raise RuntimeError('Cooperative stop after another scientific failure')
   k=e['episode'];r=episode(m,o,e,phase,a,floors,epsilon(k) if phase=='Reptile' else None);validate_record(r,e);inner+=20;outer+=1;done=k;records.append(r)
   if o is not None:assert all(float(v['step'])==k for v in o.state.values())
   scalars.append(dict(episode=k,kind=e['kind'],query_ce=r['query_ce'],query_gradient_norm=r['query_gradient_norm'],query_count=r['query_count'],inner_steps=20,outer_executed=True,projected_groups=sum(v['projected'] for v in r['projection']),projection_change_norm=r['projection_change_norm'],outer_update_norm=r['outer_update_norm'],net_update_norm=r['net_update_norm'],recurrent_spectral_norm=r['scale_after']['recurrent_spectral_norm'],effective_contrast_mean=float(r['scale_after']['effective_contrast'].mean()),centered_contrast_mean=float(r['scale_after']['centered_contrast'].mean()),nonfinite=False,epsilon=r['epsilon']))
   projections.extend(dict(episode=k,query_ce=r['query_ce'],outer_update_norm=r['outer_update_norm'],recurrent_spectral_norm=r['scale_after']['recurrent_spectral_norm'],nonfinite=False,**v) for v in r['projection'])
   if k%60==0:
    fields=['before','adapted','adapted_policy','inner_steps_array','gradient','query_logits','query_hidden','unprojected','after'];arr={f:np.stack([x[f] for x in records]) for f in fields}
    for where in ['scale_before','scale_after']:
     for field in ['rms','effective_contrast','centered_contrast']:arr[where+'_'+field]=np.stack([x[where][field] for x in records])
    for field in ['ce_before','execution_loss','total_loss','gradient_norm','hidden_absmax']:arr['inner_'+field]=np.array([[u[field] for u in x['updates']] for x in records])
    arr['support_context']=np.array([['ABC'.index(u['context']) for u in x['updates']] for x in records],np.int8);arr['support_sequence']=np.array([[u['sequence'] for u in x['updates']] for x in records],np.int8);p=dest/f'{stem}_chunk{k:04d}.npz';pending=p.with_suffix('.pending');assert not p.exists()
    with pending.open('xb') as f:np.savez_compressed(f,**arr);f.flush();os.fsync(f.fileno())
    os.replace(pending,p);cp=dest/f'{stem}_e{k}.pt';h=atomic_save(cp,checkpoint(m,o,s,a,phase,k,regsha));dump(dest/f'{stem}_e{k}.json',dict(checkpoint_sha=h,chunk_sha=sha(p),episodes=k));write_frame(dest/f'{stem}_scalars{k:04d}.csv',scalars[-60:]);write_frame(dest/f'{stem}_projection{k:04d}.csv',projections[-300:]);records.clear()
    if k in [240,960,1920]:print(f'{phase} s{s:02d} {a} {k}/1920',flush=True)
  write_frame(dest/f'{stem}_episodes.csv',scalars);write_frame(dest/f'{stem}_projections.csv',projections);row=dict(seed=s,arm=a,phase=phase,status='completed',episodes=done,inner_updates=inner,outer_updates=outer,source_sha=sha(path),wall_seconds=time.monotonic()-t);dump(dest/f'{stem}_complete.json',row);return row
 except Exception as exc:
  failure=getattr(exc,'failure_payload',dict(error=repr(exc),inner_steps=0,outer_executed=False));atomic_save(dest/f'{stem}_failure.pt',dict(payload=failure,model=m.state_dict(),optimizer=None if o is None else o.state_dict(),unsaved_completed_records=records,scalars=scalars,projections=projections,rng=torch.get_rng_state()));row=dict(seed=s,arm=a,phase=phase,status='blocked_failure',episodes=done,inner_updates=inner+failure['inner_steps'],outer_updates=outer+int(failure['outer_executed']),error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False);dump(dest/f'{stem}_failure.json',row)
  try:dump(stop,row)
  except FileExistsError:pass
  return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['Reptile','FOMAML'],required=True);phase=p.parse_args().phase;assert json.loads((OUT/'audit/preflight_decision.json').read_text())['passed'];assert json.loads((OUT/'audit/meta_benchmark.json').read_text())['passed'];assert '1 passed' in (OUT/'audit/17_GREEN_native_meta_reference.log').read_text();seal=json.loads((OUT/'audit/meta_registry_seal.json').read_text());core.hashes(OUT,seal['files'])
 if phase=='FOMAML':assert json.loads((OUT/'audit/reptile_verified.json').read_text())['passed']
 dump(OUT/f'audit/{phase.lower()}_started.json',dict(code=codes(),jobs=jobs_for(phase),time=time.time(),decision_sha=sha(OUT/'audit/preflight_decision.json')));rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  for row in pool.map(run,jobs_for(phase)):rows.append(row);print(phase+' runs '+str(len(rows))+'/12 '+row['status'],flush=True)
 write_frame(OUT/f'registry/{phase.lower()}_runs.csv',rows);ok=all(r['status']=='completed' for r in rows);dump(OUT/f'audit/{phase.lower()}_complete.json',dict(passed=ok,runs=12,episodes=sum(r['episodes'] for r in rows),inner_updates=sum(r['inner_updates'] for r in rows),outer_updates=sum(r['outer_updates'] for r in rows)));append('## '+phase+' phase complete\nAll12run slots retained. '+('Fixed1920budget completed without extra normalization or tuning.' if ok else 'Scientific failure retained; cooperative stop/no next phase or rescue.'))
 if not ok:raise RuntimeError('Scientific failure: stop all further model training')
if __name__=='__main__':main()
