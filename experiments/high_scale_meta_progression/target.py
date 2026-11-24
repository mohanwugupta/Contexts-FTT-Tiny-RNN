"""Native target updates and frozen measures; exact prefix/Adam continuation."""
from .common import *
from .protocol import restore_target,validate_resume,target_cue
from .audit import archived_target
from experiments.context_experience_prior.measures import measure_model
from experiments.heldout_context_meta.mechanisms import extra_measures
from experiments.heldout_context_meta.target import validate_training_records
from experiments.heldout_context_meta_scaling_10k.verify import verify_positions
from experiments.high_scale_constrained_meta.audit import metrics

def validate_frozen_grid(e,p,last):
 if len(e)!=(last+1)*6 or e.duplicated(['update','cue']).any() or set(e.cue)!=set(CUES) or set(e['update'])!=set(range(last+1)):raise ValueError('Invalid frozen six-cue grid')
 if len(p)!=(last+1)*42 or p.duplicated(['update','cue','position']).any():raise ValueError('Invalid position grid')
 verify_positions(p);v=p.groupby(['update','cue']).agg(ce_pos=('ce','mean'),acc_pos=('correct','mean')).reset_index().merge(e,on=['update','cue'],validate='one_to_one');np.testing.assert_allclose(v.ce_pos,v.ce,atol=1e-12,rtol=1e-12);np.testing.assert_allclose(v.acc_pos,v.accuracy,atol=1e-12,rtol=1e-12)

def evaluate(m,k,records):
 obs={cue:core.trace(m,cue) for cue in CUES}
 for cue,x in obs.items():
  if not all(np.isfinite(x[v]).all() for v in ['logits','probabilities','hidden','ce','margin']):raise FloatingPointError('Nonfinite frozen evaluation')
  records['evaluations'].append(dict(update=k,cue=cue,status='completed',ce=float(x['ce'].mean()),accuracy=float(x['accuracy'].mean()),above_chance=bool(x['ce'].mean()>np.log(4))))
  for t in range(7):records['position_predictions'].append(dict(update=k,cue=cue,status='completed',position=t+1,token=core.TARGET_PATH[t],target=int(core.TARGET[t]),ce=x['ce'][t],correct=bool(x['accuracy'][t]),margin=x['margin'][t],**{f'logit_{j}':x['logits'][t,j] for j in range(4)},**{f'probability_{j}':x['probabilities'][t,j] for j in range(4)},**{f'hidden_{j}':x['hidden'][t,j] for j in range(8)}))
 g,cc,dc=measure_model(m,obs);eg,ed=extra_measures(m,obs,cc);g.update(eg);g.update(metrics(m.state_dict()))
 for a,b in zip(dc,ed):a.update(b)
 records['geometry'].append(dict(update=k,**g));records['mathematical_checks'].extend(dict(update=k,status='completed',**v) for v in cc);records['decomposition'].extend(dict(update=k,**v) for v in dc)

def snapshot(m,p,o,k):return dict(update=k,status='completed',model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict()))

def run_target(job):
 seed,scale,phase,dose,total=job;configure();dest=OUT/f'target_v2/s{seed:02d}_scale{scale}_{phase}_e{dose}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():
  r=json.loads((dest/'complete.json').read_text());assert sha(ROOT/r['path'])==r['sha256'];return r
 if (dest/'failure.json').exists():return json.loads((dest/'failure.json').read_text())
 d,src=initial(seed,'P2','I',f'{scale}-{phase}',dose);srcsha=sha(src);code=codes();start=time.monotonic();names=['training','evaluations','position_predictions','geometry','mathematical_checks','decomposition','diagnostics'];records={k:[] for k in names};states=[];reused=0;new=0;clones=0;prefix=None
 progress=sorted(dest.glob('progress_*.pt'))
 if progress:
  latest=progress[-1];cert=json.loads(latest.with_suffix('.json').read_text());assert sha(latest)==cert['sha256'];b=torch.load(latest,weights_only=False);assert b['source_sha']==srcsha and b['code']==code and b['total']==total;records=b['records'];states=b['states'];m,p,o=restore_target(states[-1]);reused=b['reused'];new=b['new'];clones=b['clones'];k=states[-1]['update'];torch.set_rng_state(b['rng'])
 elif phase=='Fresh' and scale in [4,6]:
  old=archived_target(seed,scale);b=torch.load(old['states'],weights_only=False);states=b['states'];m,p,o=restore_target(states[-1]);k=330;reused=330;prefix=str(old['states'].relative_to(ROOT))
  for name in names:
   q=old['directory']/(name+('.csv.gz' if name in ['position_predictions','mathematical_checks','decomposition'] else '.csv'));records[name]=pd.read_csv(q,float_precision='round_trip').to_dict('records')
 else:
  assert not (dest/'started.json').exists(),'Interrupted before a recovery checkpoint: retain attempt, do not replay';m=source_model(d);p=core.policy();o=core.optimizer(m,p);k=0;states=[snapshot(m,p,o,0)];evaluate(m,0,records)
 if not (dest/'started.json').exists():dump(dest/'started.json',dict(job=job,source=str(src.relative_to(ROOT)),source_sha=srcsha,code=code,prefix=prefix,reused_updates=reused,time=time.time()))
 c=m.recurrent.weight_ih_l0[:,6].detach().clone()
 try:
  for update in range(k+1,total+1):
   if (OUT/'audit/SCIENTIFIC_STOP.json').exists():raise RuntimeError('Cooperative scientific stop')
   q=core.step(m,p,o,target_cue(update));new+=int(q['optimizer_step_executed']);records['training'].append(dict(update=update,training_cue=target_cue(update),**q))
   if q['status']!='completed':raise FloatingPointError(q['status'])
   assert torch.equal(c,m.recurrent.weight_ih_l0[:,6]);states.append(snapshot(m,p,o,update));evaluate(m,update,records)
   if update in core.DIAG:
    rr=core.diagnostic(m,p,o);clones+=sum(v['diagnostic_update_executed'] for v in rr)//2;records['diagnostics'].extend(dict(update=update,**v) for v in rr)
   if update%330==0:
    path=dest/f'progress_{update:04d}.pt';h=atomic_save(path,dict(source_sha=srcsha,code=code,total=total,states=states,records=records,reused=reused,new=new,clones=clones,rng=torch.get_rng_state()));dump(path.with_suffix('.json'),dict(sha256=h,update=update))
  e=pd.DataFrame(records['evaluations']);pos=pd.DataFrame(records['position_predictions']);validate_frozen_grid(e,pos,total);validate_training_records(pd.DataFrame(records['training']),e)
  for st in states:
   assert st['update']==0 or all(float(v['step'])==st['update'] for v in st['optimizer']['state'].values())
   assert torch.equal(st['model']['recurrent.weight_ih_l0'][:,6],c)
  assert all(r['passed'] for r in records['mathematical_checks'])
  for name in names:write_frame(dest/(name+('.csv.gz' if name in ['position_predictions','mathematical_checks','decomposition'] else '.csv')),records[name])
  path=dest/'states.pt';h=atomic_save(path,dict(states=states,source=str(src.relative_to(ROOT)),source_sha=srcsha,C_initial=c,seed=seed,scale=scale,phase=phase,dose=dose));r=dict(seed=seed,scale=scale,phase=phase,dose=dose,total=total,status='completed',directory=str(dest.relative_to(OUT)),path=str(path.relative_to(ROOT)),sha256=h,source_sha=srcsha,target_updates=new,reused_updates=reused,diagnostic_updates=clones,states=total+1,wall_seconds=time.monotonic()-start);dump(dest/'complete.json',r);return r
 except Exception as exc:
  atomic_save(dest/'failure.pt',dict(model=m.state_dict(),policy=p.state_dict(),optimizer=o.state_dict(),states=states,records=records,new=new,clones=clones,reused=reused,rng=torch.get_rng_state()));r=dict(seed=seed,scale=scale,phase=phase,dose=dose,total=total,status='blocked_integrity',directory=str(dest.relative_to(OUT)),target_updates=new,reused_updates=reused,diagnostic_updates=clones,error=repr(exc));dump(dest/'failure.json',r)
  if not (OUT/'audit/SCIENTIFIC_STOP.json').exists():dump(OUT/'audit/SCIENTIFIC_STOP.json',r)
  raise
