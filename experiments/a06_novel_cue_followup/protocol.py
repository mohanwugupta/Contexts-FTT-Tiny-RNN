"""Fixed follow-up design, native source contracts and durable finite budget."""
import copy,json,os,time,random,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from experiments.context_experience_prior import core
from experiments.context_experience_prior.registry import allowed
from experiments.meta_learning_scaling.common import source_model
from experiments.joint_phenotype_hypothesis_screen import adapter
from experiments.joint_phenotype_hypothesis_screen.common import put,csv,sha,digest,rng,set_rng,atomic,configure,cue
from experiments.meta_objective_balance.statistics import interval
ROOT=core.ROOT;SCREEN=ROOT/'results/joint_phenotype_hypothesis_screen/v1';CACHED=ROOT/'results/a06_cached_readout_comparison/v1';OUT=ROOT/'results/a06_novel_cue_followup/v1';NOVEL_TIMES=[0,10,20,40,60,100,170,330];PROBES=['A','B','C','Mixed-average','NoContext'];SEEDS=list(range(6));CUE_SEED=2026112306
def append(s):
 with (OUT/'SCRATCHPAD.md').open('a') as f:f.write('\n'+s+'\n')
def a06_state(d):
 changed=copy.deepcopy(d);changed['sequence_head.weight'].mul_(.25);return changed
def coverage(seed,condition,available,path):return [dict(seed=seed,condition=condition,update=k,status='available' if k in available else 'missing_archived_state',path=path,reason='' if k in available else 'Original A06 only saved named states; no reconstruction/new optimization authorized') for k in range(331)]
def cue_bank():
 rng=np.random.default_rng(CUE_SEED);vectors=[]
 for _ in range(8):
  while True:
   v=rng.normal(size=3);v=v/np.linalg.norm(v)
   if min(np.linalg.norm(v-c) for c in list(np.eye(3))+vectors)>1e-10:break
  vectors.append(v)
 phi=float(rng.uniform(0,2*np.pi));perp=np.array([np.cos(phi),np.sin(phi),0.]);C=np.array([0.,0.,1.])
 for degrees in [15,30,45,60]:v=C+np.tan(np.deg2rad(degrees))*perp;vectors.append(v/np.linalg.norm(v))
 rows=[]
 for i,v in enumerate(vectors):
  assert abs(np.linalg.norm(v)-1)<=1e-12 and min(np.linalg.norm(v-c) for c in np.eye(3))>1e-10
  native=v.astype(np.float32);assert abs(np.linalg.norm(native)-1)<=1e-6
  rows.append(dict(cue=f'Random{i+1:02}' if i<8 else f'C{[15,30,45,60][i-8]}',kind='random_sphere' if i<8 else 'C_arc',vector=v.tolist(),native_vector=native.tolist(),norm=float(np.linalg.norm(v)),native_norm=float(np.linalg.norm(native)),similarity_A=float(v[0]),similarity_B=float(v[1]),similarity_C=float(v[2]),distance_A=float(np.linalg.norm(v-np.eye(3)[0])),distance_B=float(np.linalg.norm(v-np.eye(3)[1])),distance_C=float(np.linalg.norm(v-C)),angle_C_degrees=float(np.rad2deg(np.arccos(np.clip(v[2],-1,1)))),perpendicular_direction=None if i<8 else perp.tolist(),generation_seed=CUE_SEED,perpendicular_azimuth=phi if i>=8 else None))
 return rows
class Ledger:
 def __init__(self,folder=OUT/'registry',max_steps=1320):
  self.path=Path(folder)/'optimizer_events.jsonl';self.path.parent.mkdir(parents=True,exist_ok=True);self.events=[json.loads(v) for v in self.path.read_text().splitlines()] if self.path.exists() else [];self.max=max_steps;self.keys={(r['seed'],r['update']) for r in self.events}
 @property
 def count(self):return len(self.events)
 def seed_count(self,seed):return sum(r['seed']==seed for r in self.events)
 def charge(self,seed,update):
  if seed not in [2,3,4,5] or not 1<=update<=330 or (seed,update) in self.keys or self.count>=self.max:raise ValueError('Unregistered, duplicate or over-budget optimizer operation')
  row=dict(seed=seed,condition='A06',update=update,physical_optimizer_calls=1)
  with self.path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
  self.events.append(row);self.keys.add((seed,update))
def paired_interval(values):
 x=np.asarray(values,float)
 if x.shape!=(6,):raise ValueError('Exactly6 paired trained seeds, not cues or positions')
 return dict(interval(x,SEEDS),n_unchanged=int((x==0).sum()))
def codes():return {str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/a06_novel_cue_followup/execution.py']}
def identity():
 return dict(code=codes(),config_sha=sha(ROOT/'configs/a06_novel_cue_followup.yaml'),source_registry_sha=sha(OUT/'registry/sources.json'),cue_registry_sha=sha(OUT/'cue_registry.csv'),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),dtype='float32',device='CPU')
def load_sources():return json.loads((OUT/'registry/sources.json').read_text())
def load_states(seed,condition):
 s=load_sources()['controls'][str(seed)]
 if condition=='A00':b=torch.load(ROOT/s['states'],weights_only=False);return {r['update']:r for r in b['states'] if r['update']<=330}
 if seed<2:return torch.load(SCREEN/f'runs/A06_s{seed}/checkpoints.pt',weights_only=False)
 p=OUT/f'runs/A06_s{seed}/states.pt'
 if p.exists():return torch.load(p,weights_only=False)['states']
 return {}

def inspect():
 configure();done=OUT/'audit/inspect.json'
 if done.exists():
  s=load_sources()
  for p,h in s['inputs'].items():assert sha(ROOT/p)==h,p
  return
 bank=cue_bank();csv(OUT/'cue_registry.csv',[dict(**{k:v for k,v in r.items() if k not in ['vector','native_vector','perpendicular_direction']},c1=r['vector'][0],c2=r['vector'][1],c3=r['vector'][2],native_c1=r['native_vector'][0],native_c2=r['native_vector'][1],native_c3=r['native_vector'][2]) for r in bank]);put(OUT/'registry/cue_vectors.json',bank);put(OUT/'registry/cue_seal.json',dict(passed=True,sha256=sha(OUT/'cue_registry.csv'),vectors_sha=sha(OUT/'registry/cue_vectors.json'),generation_seed=CUE_SEED,outcomes_inspected=False,random_distribution='Gaussian direction uniform sphere',C_arc='one independent private perpendicular direction, four fixed angles',unit_double_atol=1e-12,native_float32_atol=1e-6,identical_exclusion_atol=1e-10))
 screen=json.loads((SCREEN/'manifest.json').read_text());cache=json.loads((CACHED/'manifest.json').read_text());expected=screen['inputs']|screen['code']|{str((SCREEN/n).relative_to(ROOT)):h for n,h in screen['outputs'].items()}|cache['inputs']|cache['code']|{str((CACHED/n).relative_to(ROOT)):h for n,h in cache['outputs'].items()};inputs={str((SCREEN/'manifest.json').relative_to(ROOT)):sha(SCREEN/'manifest.json'),str((CACHED/'manifest.json').relative_to(ROOT)):sha(CACHED/'manifest.json')}
 def pin(p):
  p=Path(p);rel=str(p.relative_to(ROOT));assert p.exists(),'Missing source; no A00 retraining authorized: '+rel
  if rel not in expected:
   for n in list(screen['inputs']):
    if n.endswith('manifest.json'):
     parent=json.loads((ROOT/n).read_text());expected.update(parent.get('code',{}));expected.update(parent.get('inputs',{}))
  h=sha(p);assert expected.get(rel)==h,'Changed source '+rel;inputs[rel]=h;return h
 control_file=SCREEN/'registry/control_sources.json';pin(control_file);controls=json.loads(control_file.read_text());sources={};rows=[];missing=[];pilot_checks=[]
 for n in ['reference_thresholds.json','summary.csv','metrics.csv','practice.csv.gz','registry/runtime_identity.json','registry/derived_controls.csv','registry/inherited_precision_exceptions.csv','registry/optimizer_events.jsonl']:pin(SCREEN/n)
 for n in ['comparison.csv','positions.csv','source_audit.json']:pin(CACHED/n)
 native=['experiments/context_experience_prior/core.py','experiments/context_experience_prior/registry.py','experiments/high_scale_initialization/core.py','experiments/meta_learning_scaling/common.py','experiments/high_scale_meta_progression/protocol.py','experiments/joint_phenotype_hypothesis_screen/adapter.py','experiments/joint_phenotype_hypothesis_screen/rules.py','experiments/joint_phenotype_hypothesis_screen/analysis.py','experiments/meta_objective_balance/statistics.py','experiments/linear_scale_sweep/analysis.py','confirmatory_srn/model.py','confirmatory_srn/config.py','execution_objective_search/objectives.py']
 for n in native:pin(ROOT/n)
 for seed in SEEDS:
  s=controls[str(seed)]['A00'];sources[str(seed)]=s
  for key in ['initializer','states','evaluations','training','positions']:pin(ROOT/s[key])
  initial=torch.load(ROOT/s['initializer'],weights_only=False);assert initial['arm']=='R' and initial['episode']==1920 and initial['scale']==5 and initial['optimizer'] is None
  taskpath=ROOT/f'results/high_scale_meta_progression/v1/registry/R_s{seed:02}.json';pin(taskpath);assert initial['registry_sha']==sha(taskpath);tasks=json.loads(taskpath.read_text())['records'];assert len(tasks)==1920 and all(allowed(p) for e in tasks for p in e['paths'])
  b=torch.load(ROOT/s['states'],weights_only=False);st={r['update']:r for r in b['states'] if r['update']<=330};assert set(st)==set(range(331)) and digest(st[0]['model'])==digest(initial['model']);assert not st[0]['optimizer']['state'] and all(torch.count_nonzero(v)==0 for v in st[0]['policy'].values())
  rows.extend(coverage(seed,'A00',st,s['states']))
  if seed<2:
   for n in ['checkpoints.pt','progress.pt','observations.npz','metrics.csv','practice.csv','complete.json']:pin(SCREEN/f'runs/A06_s{seed}/{n}')
   old=torch.load(SCREEN/f'runs/A06_s{seed}/checkpoints.pt',weights_only=False);assert digest(old[0]['model'])==digest(a06_state(initial['model']));assert digest(old[0]['policy'])==digest(st[0]['policy']) and digest(old[0]['optimizer'])==digest(st[0]['optimizer']);rows.extend(coverage(seed,'A06',old,str((SCREEN/f'runs/A06_s{seed}/checkpoints.pt').relative_to(ROOT))))
   frame=pd.read_csv(SCREEN/f'runs/A06_s{seed}/metrics.csv',float_precision='round_trip');npz=np.load(SCREEN/f'runs/A06_s{seed}/observations.npz');labels=frame['update'].drop_duplicates().tolist();assert labels==list(old)
   for i,k in enumerate(labels):
    for j,c in enumerate(PROBES):
     score=core.score(npz['logits'][i,j]);rr=frame[frame['update'].eq(k)&frame.cue.eq(c)].iloc[0];np.testing.assert_allclose(score['ce'].mean(),rr.ce,atol=1e-12,rtol=1e-12);assert score['accuracy'].sum()==rr.correct_count
   pilot_checks.append(dict(seed=seed,only_output_weight_changes=True,bias_unchanged=True,all_other_parameters_identical=True,policy_and_empty_Adam_native=True,old_states=len(old),all8_probe_states_available=all(n in old for n in NOVEL_TIMES)))
   for r in rows[-331:]:
    if r['status']!='available':missing.append(r)
  else:rows.extend([dict(seed=seed,condition='A06',update=k,status='registered_new_run',path=f'results/a06_novel_cue_followup/v1/runs/A06_s{seed}/states.pt',reason='Exactly330 native A/B updates authorized') for k in range(331)])
 put(OUT/'registry/sources.json',dict(inputs=inputs,controls=sources,native_functions=native,source_manifest=sha(SCREEN/'manifest.json'),cached_manifest=sha(CACHED/'manifest.json')));csv(OUT/'registry/state_coverage_initial.csv',rows);csv(OUT/'registry/missing_archived_states.csv',missing);put(OUT/'audit/inspect.json',dict(passed=True,A00_full_histories=6,A06_old_pilots=pilot_checks,missing_pilot_states=len(missing),new_optimizer_budget=1320,novel_bank_sealed=True,no_source_changed=True,background_tasks_checked=11520,optimizer_steps=0));append('## Source/bank audit GREEN\nAll6 A00 full histories/parents and source/task hashes resolved. Ordinary Reptile1920 only, all11520 background tasks satisfy native target exclusions. Old A06 initial tensors/policy/Adam and cached outputs verified; only readoutW quartered, bias unchanged. Cue bank sealed before outcomes. Exactly642 unavailable old-pilot states retained; all8 frozen-checkpoint states available. No training/optimizer fixture executed.');print('Source/bank verified; 642 old-pilot intermediate states remain unavailable; no optimization yet',flush=True)
