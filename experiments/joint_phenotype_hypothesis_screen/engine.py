"""Bounded native runs, complete resumable checkpoints and compressed histories."""
from .common import *
from .storage import Exposure,validate_resume
from . import adapter as a,rules

class CleanStop(Exception):pass
def bank(seed):return json.loads((OUT/f'registry/tasks_s{seed:02}.json').read_text())
def sources(seed):return json.loads((OUT/'registry/control_sources.json').read_text())[str(seed)]
def support(e,candidate):return e['support_P2']*(3 if candidate=='B04' else 1)
def epsilons(candidate):
 n=80 if candidate=='B04' else 240
 return np.array([.1+(.01-.1)*j/(n-1) for j in range(n)])
def frozen_identity():
 identity=runtime_identity();put(OUT/'registry/runtime_identity.json',identity);return digest(identity)
def require_identity():
 saved=json.loads((OUT/'registry/runtime_identity.json').read_text());assert saved==clean(runtime_identity()),'Runtime changed after freeze';return digest(saved)
def due(deadline):return deadline is not None and time.monotonic()>=deadline
def charge_callback(ledger,kind,name):
 def fn(calls=1):ledger.charge(kind,name,ledger.run_count(name)+1,calls)
 return fn
def from_logits(logits,biases):
 z=np.asarray(logits);scores=[[core.score(row) for row in step] for step in z]
 return dict(logits=z,output_bias=np.asarray(biases),**{k:np.array([[v[k] for v in step] for step in scores]) for k in ['probabilities','ce','accuracy','margin']})
def rescale_raw(raw):
 z=raw['logits'];bias=raw['output_bias'][:,None,None,:];return from_logits(.25*(z-bias)+bias,raw['output_bias'])
def raw_observations(obs):
 return dict(**{field:np.array([[obs[n][c][field] for c in PROBES] for n in TIMES]) for field in ['logits','probabilities','ce','accuracy','margin']},output_bias=np.array([obs[n]['output_bias'] for n in TIMES]))
def metric_rows(raw,candidate,seed,role):
 rows=[]
 for i,k in enumerate(TIMES):
  familiar=float(raw['ce'][i,:2].mean())
  for j,c in enumerate(PROBES):
   ce=float(raw['ce'][i,j].mean());correct=int(raw['accuracy'][i,j].sum());gap=ce-familiar
   rows.append(dict(candidate=candidate,seed=int(seed),role=role,update=k,cue=c,ce=ce,correct_count=correct,accuracy=correct/7,margin=float(raw['margin'][i,j].mean()),minimum_margin=float(raw['margin'][i,j].min()),familiar_ce=familiar,gap=gap,gain_from_zero=float(raw['ce'][0,j].mean()-ce),useful=rules.useful(ce,correct),strict=rules.strict(ce,correct),historical_strict_gap_rule=bool(ce<=.1 and correct==7 and gap<=.1),floor=ce<=.05,above_chance=ce>float(np.log(4)),status='completed'))
 return rows
def save_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.pending')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def save_results(dest,candidate,seed,role,raw,training,snapshots,checks,metadata):
 save_npz(dest/'observations.npz',**raw);csv(dest/'metrics.csv',metric_rows(raw,candidate,seed,role));csv(dest/'practice.csv',training);atomic(dest/'checkpoints.pt',snapshots);put(dest/'mathematics.json',checks)
 rows=training if isinstance(training,list) else training.to_dict('records')
 if len(rows):
  f=pd.DataFrame(rows);save_npz(dest/'practice.npz',**{c:f[c].to_numpy() for c in f if f[c].dtype!=object},training_cue=f.training_cue.to_numpy(dtype=str))
 result=dict(passed=True,candidate=candidate,seed=seed,role=role,status='completed',**metadata,files={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name not in ['complete.json','progress.pt']});put(dest/'complete.json',result);return result
def accepted(dest):
 p=dest/'complete.json'
 if not p.exists():return None
 r=json.loads(p.read_text())
 for n,h in r['files'].items():assert sha(dest/n)==h,'Changed cache '+str(dest/n)
 return r
def materialize_control(seed,label):
 dest=OUT/f'runs/{label}_s{seed}';done=accepted(dest)
 if done:return done
 dest.mkdir(parents=True,exist_ok=True);s=sources(seed)[label];b=torch.load(ROOT/s['states'],weights_only=False);states={x['update']:x for x in b['states'] if x['update'] in TIMES};assert set(states)==set(TIMES);assert digest(states[0]['model'])==digest(torch.load(ROOT/s['initializer'],weights_only=False)['model'])
 p=pd.read_csv(ROOT/s['positions'],float_precision='round_trip');p=p[p['update'].isin(TIMES)&p.cue.isin(PROBES)];zs=[]
 for k in TIMES:
  zs.append([p[p['update'].eq(k)&p.cue.eq(c)].sort_values('position')[[f'logit_{j}' for j in range(4)]].to_numpy() for c in PROBES])
 raw=from_logits(np.array(zs),[states[k]['model']['sequence_head.bias'].numpy() for k in TIMES]);tr=pd.read_csv(ROOT/s['training'],float_precision='round_trip');tr=tr[tr['update'].le(330)];assert len(tr)==330 and [cue(k) for k in range(1,331)]==tr.training_cue.tolist()
 out=save_results(dest,label,seed,'pilot' if seed<2 else ('reference' if label=='F00' else 'confirmation_control'),raw,tr,states,[dict(passed=True,status='source_archived_math_checks',source=s['states'])],dict(reused=True,reused_meta_inner=s['meta_inner_steps'],reused_target=330,new_meta_inner=0,new_target=0,source=s,wall_seconds=0.))
 if label=='A00':
  sd=OUT/f'runs/A06_static_s{seed}';sd.mkdir(parents=True,exist_ok=True)
  if not accepted(sd):save_results(sd,'A06_static',seed,'pilot' if seed<2 else 'confirmation_control',rescale_raw(raw),[],{},[dict(passed=True,status='posthoc_bias_aware_logit_formula')],dict(reused=True,new_meta_inner=0,new_target=0,wall_seconds=0.,control='derived frozen logits; no new learned trajectory',source=s))
 return out
def initial(seed,candidate):
 s=sources(seed);name='A00' if candidate.startswith('A') else 'F00';base=torch.load(ROOT/s[name]['initializer'],weights_only=False)['model']
 if candidate=='B05':
  m,record=a.wide(base,seed,bank(seed)['width']);put(OUT/f'registry/width_calibration_s{seed}.json',record);return m
 m=a.make(base,candidate)
 if candidate=='A06':m=a.output_scale(m)
 return m
def target(seed,candidate,role,deadline=None):
 identity=require_identity();dest=OUT/f'runs/{candidate}_s{seed}';dest.mkdir(parents=True,exist_ok=True);done=accepted(dest)
 if done:return done
 if (dest/'failure.json').exists():return json.loads((dest/'failure.json').read_text())|dict(candidate=candidate,seed=seed)
 ledger=Exposure();name=f'target/{candidate}_s{seed}';charge=charge_callback(ledger,'target',name);progress=dest/'progress.pt';start=time.monotonic()
 if progress.exists():
  cp=torch.load(progress,weights_only=False);validate_resume(cp,identity,ledger.run_count(name));m,p,o=a.restore(cp['state']);training=cp['training'];obs=cp['observations'];snapshots=cp['snapshots'];checks=cp['checks'];k=cp['cursor'];elapsed=cp['elapsed']
 else:
  assert ledger.run_count(name)==0,'Uncertain target operations; no automatic replay';m=initial(seed,candidate) if candidate.startswith('A') else a.make(torch.load(OUT/f'runs/{candidate}_s{seed}/meta_complete.pt',weights_only=False)['model'],candidate);p=core.policy();eta=json.loads((OUT/'registry/sgd_calibration.json').read_text())['eta_final'] if candidate=='A04' else None;o=a.optimizer(m,p,candidate,eta);assert not o.state;training=[];obs={};snapshots={};checks=[];k=0;elapsed=0.
  def evaluate(n):
   xx=a.probes(m,p,o);obs[n]=xx|{'output_bias':m.sequence_head.bias.detach().numpy().copy()};snapshots[n]=a.snapshot(m,p,o,n,candidate);checks.append(dict(update=n,**a.mathematics(m,xx,candidate)))
  evaluate(0)
 def save(k):atomic(progress,dict(identity=identity,cursor=k,state=a.snapshot(m,p,o,k,candidate),training=training,observations=obs,snapshots=snapshots,checks=checks,elapsed=elapsed+time.monotonic()-start))
 for k in range(k+1,331):
  if due(deadline):save(k-1);raise CleanStop('Saved complete target state before invocation time limit')
  r=a.update(m,p,o,cue(k),candidate,k,charge);training.append(dict(update=k,training_cue=cue(k),**r))
  if r['status']!='completed':
   save(k);put(dest/'failure.json',dict(status='numerical_failure',update=k,record=r));return dict(candidate=candidate,seed=seed,status='numerical_failure')
  if k in TIMES:
   xx=a.probes(m,p,o);obs[k]=xx|{'output_bias':m.sequence_head.bias.detach().numpy().copy()};snapshots[k]=a.snapshot(m,p,o,k,candidate);checks.append(dict(update=k,**a.mathematics(m,xx,candidate)))
  if k%10==0:save(k)
 result=save_results(dest,candidate,seed,role,raw_observations(obs),training,snapshots,checks,dict(reused=False,new_target=330,new_meta_inner=0,wall_seconds=elapsed+time.monotonic()-start,identity=identity))
 print(f'Target {candidate} seed{seed}:330/330, {result["wall_seconds"]:.2f}s; equivalent remaining target cost {(3960 if candidate.startswith("A") else 3960)*result["wall_seconds"]/330:.1f}s before conditional confirmation',flush=True);return result

def augmentation(m,b,candidate,eidx,step,validation=False):
 m.cue_basis=None;m.mask_cue=False
 if candidate=='B02':m.cue_basis=torch.tensor(b['background_Q' if validation else 'meta_Q'][eidx],dtype=next(m.parameters()).dtype)
 if candidate=='B03':
  if validation:
   e=b['background'][eidx]
   # Validation pattern fixed before outcomes: reuse first three sealed shared masks.
   shared_indices=[j for j,e in enumerate(b['meta']) if e['kind']=='shared'];which=sum(e['kind']=='shared' for e in b['background'][:eidx]);mask=b['missing_masks'][shared_indices[which]] if e['kind']=='shared' else [False]*20
  else:mask=b['missing_masks'][eidx]
  m.mask_cue=mask[step%20]

def pretrain(seed,candidate,deadline=None):
 identity=require_identity();dest=OUT/f'runs/{candidate}_s{seed}';dest.mkdir(parents=True,exist_ok=True);complete=dest/'meta_complete.pt'
 if complete.exists():
  cert=json.loads((dest/'meta_complete.json').read_text());assert sha(complete)==cert['sha256'];return cert
 assert candidate in [f'B{i:02}' for i in range(1,7)];b=bank(seed);eps=epsilons(candidate);episodes=b['meta'][:len(eps)];ledger=Exposure();name=f'meta/{candidate}_s{seed}';charge=charge_callback(ledger,'meta_inner',name);path=dest/'meta_progress.pt';start=time.monotonic()
 if (dest/'meta_failure.json').exists():return json.loads((dest/'meta_failure.json').read_text())
 if path.exists():
  cp=torch.load(path,weights_only=False);validate_resume(cp,identity,ledger.run_count(name));meta=a.make(cp['meta'],candidate);epidx=cp['episode'];inner=cp['inner'];records=cp['records'];outer_records=cp['outer_records'];elapsed=cp['elapsed'];task,p,o=a.restore(cp['task']) if cp['task'] is not None else (None,None,None);set_rng(cp['rng'])
 else:
  assert ledger.run_count(name)==0,'Uncertain meta operations; no automatic replay';meta=initial(seed,candidate);epidx=0;inner=0;records=[];outer_records=[];task=p=o=None;elapsed=0.
 def save(epidx,inner,task):
  atomic(path,dict(identity=identity,cursor=ledger.run_count(name),meta=copy.deepcopy(meta.state_dict()),episode=epidx,inner=inner,task=None if task is None else a.snapshot(task,p,o,ledger.run_count(name),candidate),records=records,outer_records=outer_records,elapsed=elapsed+time.monotonic()-start,rng=rng()))
 for ei in range(epidx,len(episodes)):
  e=episodes[ei];schedule=support(e,candidate)
  if task is None:task=copy.deepcopy(meta);p=core.policy();o=core.optimizer(task,p);inner=0;assert not o.state and all(torch.count_nonzero(v)==0 for v in p.state_dict().values())
  for j in range(inner,len(schedule)):
   if due(deadline):save(ei,j,task);raise CleanStop('Saved complete meta/inner state before invocation time limit')
   augmentation(task,b,candidate,ei,j);r=schedule[j];q=a.update(task,p,o,r['context'],candidate,j+1,charge,path=e['paths'][r['sequence']],meta=True);records.append(dict(episode=ei+1,inner_update=j+1,context=r['context'],sequence=r['sequence'],kind=e['kind'],**q))
   if q['status']!='completed':save(ei,j+1,task);put(dest/'meta_failure.json',dict(status='numerical_failure',episode=ei+1,inner=j+1,record=q));return dict(status='numerical_failure')
  before=digest(meta.state_dict());core.reptile(meta,task,float(eps[ei]));outer_records.append(dict(episode=ei+1,epsilon=float(eps[ei]),inner_updates=len(schedule),kind=e['kind'],pair=e['pair'],before_hash=before,adapted_hash=digest(task.state_dict()),after_hash=digest(meta.state_dict())));task=None;inner=0;save(ei+1,0,None)
  if (ei+1)%60==0 or ei+1==len(episodes):print(f'Meta {candidate} seed{seed}: {ei+1}/{len(episodes)}, {ledger.run_count(name)} inner updates',flush=True)
 atomic(complete,dict(model=copy.deepcopy(meta.state_dict()),variant=candidate,seed=seed,identity=identity,rng=rng(),episode=len(episodes),cursor=4800));csv(dest/'meta_episodes.csv',outer_records);csv(dest/'meta_updates.csv.gz',records);put(dest/'meta_complete.json',dict(passed=True,status='completed',seed=seed,candidate=candidate,episodes=len(episodes),inner_updates=4800,interpolations=len(episodes),task_bank_sha=sha(OUT/f'registry/tasks_s{seed:02}.json'),sha256=sha(complete),wall_seconds=elapsed+time.monotonic()-start))
 result=json.loads((dest/'meta_complete.json').read_text());print(f'Completed {candidate} seed{seed} ordinary pretraining {result["wall_seconds"]:.2f}s, projected remaining12 comparable pilots {12*result["wall_seconds"]:.1f}s; width/gate timing may differ',flush=True);return result

def own_probes(m,e):
 rows=[];m.mask_cue=False
 for c in e['pair']:
  seq=next(r['sequence'] for r in e['support_P2'] if r['context']==c);x=a.probes(m,path=e['paths'][seq],role_probes=[c])[c];rows.append(dict(cue=c,ce=float(x['ce'].mean()),correct_count=int(x['accuracy'].sum()),accuracy=float(x['accuracy'].mean()),margin=float(x['margin'].mean())))
 return rows
def background(seed,candidate,fresh=False,deadline=None):
 identity=require_identity();key=f'{candidate}_{"Fresh" if fresh else "meta"}_s{seed}';dest=OUT/f'background/{key}';receipt=dest/'complete.json'
 if receipt.exists():
  r=json.loads(receipt.read_text());assert sha(dest/'results.pt')==r['sha256'];return r
 dest.mkdir(parents=True,exist_ok=True);b=bank(seed)
 if fresh:assert candidate=='B00';model=torch.load(ROOT/sources(seed)['F00']['initializer'],weights_only=False)['model']
 elif candidate=='B00':model=torch.load(ROOT/sources(seed)['B00']['initializer'],weights_only=False)['model']
 else:model=torch.load(OUT/f'runs/{candidate}_s{seed}/meta_complete.pt',weights_only=False)['model']
 ledger=Exposure();name='background/'+key;charge=charge_callback(ledger,'auxiliary',name);rows=[];raw=[]
 for ei,e in enumerate(b['background']):
  cache=dest/f'task_{ei}.pt'
  if cache.exists():
   saved=torch.load(cache,weights_only=False);assert saved['identity']==identity;rows.extend(saved['rows']);raw.append(saved);continue
  if due(deadline):raise CleanStop('Background validation stopped between exact cached tasks')
  if ledger.run_count(name)!=sum(len(x['records']) for x in raw):raise ValueError('Uncertain background task; no duplicate work')
  m=a.make(model,candidate);p=core.policy();o=core.optimizer(m,p);augmentation(m,b,candidate,ei,0,True);pre=own_probes(m,e);records=[]
  for j,r in enumerate(e['support_P2']):
   augmentation(m,b,candidate,ei,j,True);q=a.update(m,p,o,r['context'],candidate,j+1,charge,path=e['paths'][r['sequence']],meta=True);records.append(q)
   if q['status']!='completed':break
  post=own_probes(m,e) if q['status']=='completed' else [dict(cue=c,ce=np.nan,correct_count=0,accuracy=np.nan,margin=np.nan) for c in e['pair']]
  rr=[dict(candidate=candidate,seed=seed,fresh=fresh,task=ei,kind=e['kind'],cue=x['cue'],before_ce=x['ce'],after_ce=y['ce'],gain=x['ce']-y['ce'],after_correct=y['correct_count'],useful=rules.useful(y['ce'],y['correct_count']),finite=bool(q['status']=='completed' and np.isfinite(y['ce']))) for x,y in zip(pre,post)]
  saved=dict(identity=identity,rows=rr,task=e,pre=pre,post=post,records=records,state=a.snapshot(m,p,o,20,candidate));atomic(cache,saved);rows.extend(rr);raw.append(saved)
 f=pd.DataFrame(rows);task=f.groupby('task').agg(gain=('gain','mean'),finite=('finite','all'),useful=('useful','all'),kind=('kind','first'));viable=bool(task.finite.all() and (task.gain>1e-4).sum()>=4 and task.groupby('kind').gain.mean().gt(0).all());competent=bool(task.useful.sum()>=4 and all(task[task.kind.eq(k)].useful.sum()>=2 for k in ['shared','conflict']));atomic(dest/'results.pt',raw);csv(dest/'metrics.csv',rows);r=dict(passed=True,candidate=candidate,seed=seed,fresh=fresh,adaptation_viable=viable,background_competent=competent,tasks_useful=int(task.useful.sum()),mean_gain=float(task.gain.mean()),mean_shared_gain=float(task[task.kind.eq('shared')].gain.mean()),mean_conflict_gain=float(task[task.kind.eq('conflict')].gain.mean()),sha256=sha(dest/'results.pt'));put(receipt,r);return r
