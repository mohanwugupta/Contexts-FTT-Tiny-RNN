"""Fixed, evaluation-only protocol and append-only stage provenance."""
from pathlib import Path
import argparse, hashlib, importlib, json, sys, traceback
import numpy as np
import pandas as pd
from context_sequence.provenance.hashing import sha256, object_hash
from experiments.output_compatible_trajectories.common import clean, table, environment
from experiments.output_compatible_trajectories.continuation import SEQUENCE, frozen_weight_guard, future_summary
from experiments.context_mixing_sequence_continuity.source_adapter import records, cues, load_original
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/context_failure_selective_repair/v1'
P4=ROOT/'results/output_compatible_trajectories/v1_20261111'
D=ROOT/'results/context_mixing_sequence_continuity/v1'
NAMES=('A','B','Mixed'); K=(3,5)
H=[f'hidden_{j}' for j in range(8)]; Z=[f'logit_{j}' for j in range(4)]; P=[f'probability_{j}' for j in range(4)]
CONTRACT=dict(version='v1',seeds=list(range(20)),k=list(K),sequence=list(SEQUENCE),cue={'A':[1.,0.,0.],'B':[0.,1.,0.],'Mixed':[.5,.5,0.]},
 rank=2,absolute_rank_tolerance=1e-8,relative_rank_tolerance=1e-4,margin=1e-6,box_tolerance=1e-10,identity_tolerance=1e-10,zero_tolerance=1e-12,
 fd_scales=[1e-3,1e-4],safety=.95,maximum_amplitude=1.,random_seed=20261113,bootstrap_seed=20260925,bootstrap_draws=10000,
 model_updates=0,optimization_steps=0,primary='Mixed k3',references=['A_history_Mixed_update','B_history_Mixed_update'],claim_requires_all20=True)
CONTRACT['sha256']=object_hash(CONTRACT)


def dump(path,x):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(clean(x),f,indent=2,allow_nan=False);f.write('\n')

def save(out,name,df):
 p=Path(out)/'tables'/f'{name}.csv'
 with p.open('x') as f:df.to_csv(f,index=False)

def report(out,name,text):
 with (Path(out)/name).open('x') as f:f.write(text.rstrip()+'\n')

def append(out,text):
 with (Path(out)/'scratchpad.md').open('a') as f:f.write('\n'+text.rstrip()+'\n')

def codes():return {str(p.relative_to(ROOT)):sha256(p) for base in ['experiments','tests'] for p in sorted((ROOT/base/'context_failure_selective_repair').glob('*.py'))}

def check_files(base,files):
 for name,want in files.items():
  p=Path(base)/name
  if not p.is_file() or sha256(p)!=want:raise ValueError('Immutable bytes missing/changed: '+str(p))
 return True

def read(out,name):return pd.read_csv(Path(out)/'tables'/f'{name}.csv',float_precision='round_trip')

def select(df,seed,k,source,branch):return df[(df.seed==seed)&(df.k==k)&(df.source_context==source)&(df.branch==branch)].sort_values('transition').reset_index(drop=True)

def native(df,seed,name):return df[(df.seed==seed)&(df.cue==name)].sort_values('transition').reset_index(drop=True)

def refs(df,seed):return tuple({'raw_ce':native(df,seed,g).raw_ce.to_numpy()} for g in ['A','B'])

def bootstrap(df,value='effect'):
 if df.seed.duplicated().any():raise ValueError('Bootstrap unit must be trained seed, not transition or reference')
 if sorted(df.seed.tolist())!=list(range(20)):raise ValueError('Explicit full20 seed rows required including missing')
 x=df[value].to_numpy(float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,ci_low=np.nan,ci_high=np.nan,n_valid=0,n_seeds=20,resampling_unit='seed',status='not_estimable')
 draws=np.random.default_rng(CONTRACT['bootstrap_seed']).choice(x,(CONTRACT['bootstrap_draws'],n),replace=True).mean(1)
 lo,hi=np.quantile(draws,[.025,.975]);return dict(mean=x.mean(),ci_low=lo,ci_high=hi,n_valid=n,n_seeds=20,resampling_unit='seed',status='complete_seed_support' if n==20 else 'conditional_estimable_seeds')

def verify_stage(out,stage):
 m=json.loads((Path(out)/'manifests'/f'{stage}.json').read_text())
 if not m['passed'] or m['contract_sha']!=CONTRACT['sha256']:raise ValueError('Prerequisite blocked: '+stage)
 check_files(ROOT,m['inputs']);check_files(ROOT,m['code']);check_files(out,m['outputs']);return m

def run(stage,out=OUT):
 out=Path(out).resolve()
 if out==P4 or out==D or 'context_failure_selective_repair' not in str(out):raise ValueError('Cannot overwrite sealed predecessor')
 if not (out/'scratchpad.md').is_file():raise FileNotFoundError('Initialize scratchpad before implementation')
 stages=['r0','r1','r2','r3','r4','r5'];i=stages.index(stage)
 for prior in stages[:i]:verify_stage(out,prior)
 if stage=='r0':
  from .audit import inputs
  ins=inputs();dump(out/'audit/frozen_inputs.json',ins);dump(out/'manifests/config.json',CONTRACT)
 else:ins=json.loads((out/'audit/frozen_inputs.json').read_text())
 code=codes();check_files(ROOT,ins);dump(out/'manifests'/f'{stage}_started.json',dict(stage=stage,code=code,contract=CONTRACT,**environment()))
 existing={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()};passed=False
 try:
  with frozen_weight_guard():status=importlib.import_module('experiments.context_failure_selective_repair.'+{'r0':'audit','r1':'factorial','r2':'geometry','r3':'repair','r4':'decision','r5':'reporting'}[stage]).run(out)
  check_files(ROOT,ins)
  if code!=codes():raise ValueError('Code changed during stage')
  passed=True
 except Exception as e:
  status={'error':str(e)};dump(out/'audit'/f'{stage}_error.json',dict(error=str(e),traceback=traceback.format_exc()));raise
 finally:
  outputs={str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and str(p.relative_to(out)) not in existing and p.name not in ['scratchpad.md','.DS_Store']}
  dump(out/'manifests'/f'{stage}.json',dict(stage=stage,passed=passed,contract_sha=CONTRACT['sha256'],code=code,inputs=ins,outputs=outputs,status=status,**environment()))
  append(out,f'## {stage} '+('completed' if passed else 'BLOCKED')+'\n'+json.dumps(clean(status))+'\nManifest verifies original source/checkpoint bytes; no training or optimization.')
 print(json.dumps(clean(status)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--output',default=str(OUT));a=p.parse_args();run(a.stage,a.output)
