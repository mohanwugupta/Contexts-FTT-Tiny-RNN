"""Exactly four native target histories; frozen probes reuse accepted arrays."""
from .protocol import *

def authorized_update(seed,condition,k,c):
 if seed not in [2,3,4,5] or condition!='A06' or k not in range(1,331) or c!=cue(k):raise ValueError('Only registered A06/A-B target practice authorized')
 return True
def validate_cursor(saved,charged,old_identity,new_identity):
 if saved!=charged or old_identity!=new_identity:raise ValueError('Uncertain operations or changed protocol; do not replay')
 return True
def frozen(m,p,o,vector,cursor):
 before=digest((m.state_dict(),p.state_dict(),o.state_dict(),rng(),cursor,{n:v.training for n,v in m.named_modules()},[v.grad for v in list(m.parameters())+list(p.parameters())]));x,y=core.tensors()
 with torch.no_grad():z,h,_=m(x,torch.tensor(vector,dtype=next(m.parameters()).dtype))
 after=digest((m.state_dict(),p.state_dict(),o.state_dict(),rng(),cursor,{n:v.training for n,v in m.named_modules()},[v.grad for v in list(m.parameters())+list(p.parameters())]));assert before==after
 logits=z[0].detach().numpy().astype(float);finite=bool(np.isfinite(logits).all() and torch.isfinite(h).all());d=core.score(logits) if finite else dict(logits=logits,probabilities=np.full((7,4),np.nan),ce=np.full(7,np.nan),accuracy=np.full(7,False),margin=np.full(7,np.nan));d['status']='completed' if finite else 'nonfinite_frozen';return d
def observe(m,p,o,k):return {c:frozen(m,p,o,core.CUES[c],dict(update=k,task='target',next_cue=cue(k+1) if k<330 else None)) for c in PROBES}
def save_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.pending')
 with temp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def observations_array(observations):
 times=sorted(observations);return dict(updates=np.array(times),probes=np.array(PROBES),**{field:np.array([[observations[k][c][field] for c in PROBES] for k in times]) for field in ['logits','probabilities','ce','accuracy','margin','status']},output_bias=np.array([observations[k]['bias'] for k in times]))
def require_identity():
 value=identity();put(OUT/'registry/runtime_identity.json',value);return digest(value)
def train(seed,deadline=None):
 configure();runid=require_identity();dest=OUT/f'runs/A06_s{seed}';dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json';failure=dest/'failure.json';ledger=Ledger()
 if done.exists():
  r=json.loads(done.read_text());assert sha(dest/'states.pt')==r['states_sha'] and sha(dest/'original.npz')==r['observations_sha'];return r
 if failure.exists():return json.loads(failure.read_text())
 initial=torch.load(ROOT/load_sources()['controls'][str(seed)]['initializer'],weights_only=False)['model'];path=dest/'progress.pt';start=time.monotonic()
 if path.exists():
  cp=torch.load(path,weights_only=False);validate_cursor(cp['update'],ledger.seed_count(seed),cp['identity'],runid);m,p,o=adapter.restore(cp['states'][cp['update']]);states=cp['states'];observations=cp['observations'];records=cp['records'];begin=cp['update'];elapsed=cp['elapsed']
 else:
  assert ledger.seed_count(seed)==0,'Uncertain target operations without checkpoint';m=source_model(a06_state(initial));p=core.policy();o=core.optimizer(m,p);assert not o.state and all(torch.count_nonzero(v)==0 for v in p.state_dict().values());assert digest(m.state_dict())==digest(a06_state(initial));states={0:adapter.snapshot(m,p,o,0,'A06')};observations={0:observe(m,p,o,0)|{'bias':m.sequence_head.bias.detach().numpy().copy()}};records=[];begin=0;elapsed=0.
 def save(k):atomic(path,dict(identity=runid,update=k,states=states,observations=observations,records=records,elapsed=elapsed+time.monotonic()-start))
 for k in range(begin+1,331):
  if deadline is not None and time.monotonic()>=deadline:save(k-1);return dict(seed=seed,status='clean_time_limit',completed=k-1)
  c=cue(k);authorized_update(seed,'A06',k,c);ledger.charge(seed,k)
  try:r=core.step(m,p,o,c)
  except Exception as ex:
   save(k-1);atomic(dest/'failure_state.pt',dict(model=m.state_dict(),policy=p.state_dict(),optimizer=o.state_dict(),rng=rng()));put(failure,dict(seed=seed,status='technical_failure_uncertain_step',update=k,error=repr(ex),no_automatic_retry=True));break
  records.append(dict(update=k,training_cue=c,**r));states[k]=adapter.snapshot(m,p,o,k,'A06');observations[k]=observe(m,p,o,k)|{'bias':m.sequence_head.bias.detach().numpy().copy()};assert torch.equal(states[k]['model']['recurrent.weight_ih_l0'][:,6],states[0]['model']['recurrent.weight_ih_l0'][:,6])
  if k%10==0:save(k)
  if r['status']!='completed':save(k);put(failure,dict(seed=seed,status='numerical_failure',update=k,record=r,no_automatic_retry=True));break
 last=max(states);atomic(dest/'states.pt',dict(states=states,source=load_sources()['controls'][str(seed)]['initializer'],condition='A06',seed=seed,scale=5,identity=runid));save_npz(dest/'original.npz',**observations_array(observations));csv(dest/'practice.csv',records)
 result=dict(passed=last==330 and not failure.exists(),seed=seed,condition='A06',status='completed' if not failure.exists() else json.loads(failure.read_text())['status'],states=len(states),optimizer_attempts=ledger.seed_count(seed),physical_optimizer_calls=sum(bool(r['optimizer_step_executed']) for r in records),states_sha=sha(dest/'states.pt'),observations_sha=sha(dest/'original.npz'),wall_seconds=elapsed+time.monotonic()-start);put(done,result);print(json.dumps(result),flush=True);return result

def original_cache(seed,condition):
 dest=OUT/f'original_cache/{condition}_s{seed}.npz'
 if dest.exists():return dict(np.load(dest))
 s=load_sources()['controls'][str(seed)];states=load_states(seed,condition)
 if condition=='A00':
  frame=pd.read_csv(ROOT/s['positions'],float_precision='round_trip');frame=frame[frame['update'].le(330)&frame.cue.isin(PROBES)];times=sorted(states);logits=np.array([[frame[frame['update'].eq(k)&frame.cue.eq(c)].sort_values('position')[[f'logit_{j}' for j in range(4)]].to_numpy() for c in PROBES] for k in times]);bias=np.array([states[k]['model']['sequence_head.bias'].numpy() for k in times]);obs={k:{c:core.score(logits[i,j])|dict(status='completed') for j,c in enumerate(PROBES)}|dict(bias=bias[i]) for i,k in enumerate(times)};arrays=observations_array(obs)
 elif seed<2:
  bundle=np.load(SCREEN/f'runs/A06_s{seed}/observations.npz');metric=pd.read_csv(SCREEN/f'runs/A06_s{seed}/metrics.csv');times=metric['update'].drop_duplicates().to_numpy();arrays=dict(updates=times,probes=np.array(PROBES),**{k:bundle[k] for k in ['logits','probabilities','ce','accuracy','margin','output_bias']},status=np.full((len(times),5),'completed'))
 else:arrays=dict(np.load(OUT/f'runs/A06_s{seed}/original.npz'))
 save_npz(dest,**arrays);return arrays

def novel(seed,condition):
 configure();runid=require_identity();dest=OUT/f'novel_cache/{condition}_s{seed}';dest.mkdir(parents=True,exist_ok=True);bank=json.loads((OUT/'registry/cue_vectors.json').read_text());states=load_states(seed,condition);completed=0;forwards=0
 for n in NOVEL_TIMES:
  path=dest/f'update{n:03}.npz';receipt=path.with_suffix('.json')
  if receipt.exists():assert json.loads(receipt.read_text())['identity']==runid and sha(path)==json.loads(receipt.read_text())['sha256'];completed+=1;continue
  if n not in states:put(dest/f'missing{n:03}.json',dict(seed=seed,condition=condition,update=n,status='missing_state',original_path=load_sources()['controls'][str(seed)]['states'] if condition=='A00' else f'A06_s{seed}',no_reconstruction=True));continue
  st=states[n];m=source_model(st['model']);p=core.policy();p.load_state_dict(st['policy']);o=core.optimizer(m,p);o.load_state_dict(copy.deepcopy(st['optimizer']));before=digest((m.state_dict(),p.state_dict(),o.state_dict(),rng()));results=[frozen(m,p,o,row['native_vector'],dict(update=n,condition=condition,seed=seed,task='frozen_only')) for row in bank];assert before==digest((m.state_dict(),p.state_dict(),o.state_dict(),rng()));forwards+=len(bank)
  save_npz(path,cues=np.array([r['cue'] for r in bank]),update=np.array(n),**{key:np.array([r[key] for r in results]) for key in ['logits','probabilities','ce','accuracy','margin','status']},output_bias=m.sequence_head.bias.detach().numpy());put(receipt,dict(passed=True,identity=runid,seed=seed,condition=condition,update=n,sequence_forward_evaluations=12,state_hash=digest(st),sha256=sha(path),all_model_policy_Adam_RNG_states_unchanged=True));completed+=1
 print('Frozen novel',seed,condition,'checkpoints',completed,'new sequence forwards',forwards,flush=True)

def run(deadline=None):
 inspect();require_identity()
 for seed in [2,3,4,5]:
  r=train(seed,deadline)
  if r['status']=='clean_time_limit':return r
 rows=[]
 for seed in SEEDS:
  for c in ['A00','A06']:
   original_cache(seed,c);states=load_states(seed,c);rows.extend(coverage(seed,c,states,load_sources()['controls'][str(seed)]['states'] if c=='A00' else (f'results/joint_phenotype_hypothesis_screen/v1/runs/A06_s{seed}/checkpoints.pt' if seed<2 else f'results/a06_novel_cue_followup/v1/runs/A06_s{seed}/states.pt')));novel(seed,c)
 csv(OUT/'registry/state_coverage.csv',rows);l=Ledger();novel_count=sum(json.loads(p.read_text())['sequence_forward_evaluations'] for p in (OUT/'novel_cache').glob('*/update*.json'));new_original=sum(json.loads((OUT/f'runs/A06_s{s}/complete.json').read_text())['states']*5 for s in [2,3,4,5]);original_b=sum(r['update'] in NOVEL_TIMES and r['status']=='available' for r in rows)*5
 put(OUT/'audit/execution_complete.json',dict(passed=True,target_attempts=l.count,target_cap=1320,new_meta_updates=0,optimizer_fixtures=0,missing_states=sum(r['status']!='available' for r in rows),available_states=sum(r['status']=='available' for r in rows),experimentB_unique_cells=novel_count+original_b,experimentB_novel_forward_sequences=novel_count,experimentB_original_reused_cells=original_b,dense_new_original_forward_sequences=new_original,total_new_forward_sequences=new_original+novel_count,static_controls_use_only_arrays=True));append('## Fixed execution complete\nExactly registered four A06 histories attempted, no new meta/training controls/replays. Full new states and actual pre-update losses saved, C columns fixed at every step. All12 unknown cues evaluated frozen at all available registered states/model, per-step probes preserve complete model/policy/Adam/RNG/grad/mode/cursor. Old states missing, no replacement. Read numerical outcomes only after this fixed execution grid.')
