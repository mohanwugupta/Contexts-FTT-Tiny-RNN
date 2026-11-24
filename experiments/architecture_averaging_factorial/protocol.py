"""Fixed factorial protocol and source-independent causal-window feasibility."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/architecture_averaging_factorial/v1'
SEQ=np.array([0,1,2,0,3,1,2,3]);TARGET=SEQ[1:]
ARCHS=['FF-Linear','FF-Tanh','RNN-Linear','RNN-Tanh'];STEPS=[170,330,660,1000]
CUES={name:entry['vector'] for name,entry in json.loads((ROOT/'configs/novelty_registry_v1.json').read_text())['cues'].items()}
CONFIG=dict(seeds=list(range(20)),architectures=ARCHS,checkpoints=STEPS,primary_endpoint=1000,updates_per_model=1000,planned_updates=80000,input_slots=3,slot_order='oldest_to_current',padding=4,token_features=15,context_features=3,hidden_dim=8,source_weight_std=2.,recurrent_initial_spectral_norm=.9,lr=.005,gradient_clip=1.,lambda_execution=.2,execution_c_time=.5,execution_c_risk=1.,execution_gamma=2.,schedule='A,B,A,B,...',float32_roundoff_multiplier=64,double_identity_atol=1e-10,double_identity_rtol=1e-12,onset_tolerance=1e-8,saturation_abs=.95,bootstrap_draws=10000,bootstrap_seed=20260925,reliability_min_strong=16,positive_gap=1e-6,material_gap=.1,success_ce=.5,success_accuracy=6/7,success_max_gap=.1,context_logit_distance=1.,context_probability_tv=.1,context_hidden_distance=1.,historical_models_not_pooled=True)


def windows():
 return np.eye(5,dtype=np.float32)[np.array([[4 if j<0 else SEQ[j] for j in range(t-2,t+1)] for t in range(7)])].reshape(7,15)

def validate_windows(x):
 if np.asarray(x).shape!=(7,15) or not np.array_equal(x,windows()):raise ValueError('Causal history input fixture incorrect')
 return True

def linear_feasibility(x):
 x=np.asarray(x,float);x=np.c_[x,np.ones(7)];d=x.shape[1];constraints=[]
 for i,y in enumerate(TARGET):
  for c in range(4):
   if c==y:continue
   r=np.zeros((4,d));r[c]=x[i];r[y]=-x[i];constraints.append(r.ravel())
 result=linprog(np.zeros(4*d),A_ub=np.array(constraints),b_ub=-np.ones(21),bounds=[(None,None)]*(4*d),method='highs')
 if not result.success:raise ValueError('Targets not linearly separable from fixture')
 w=result.x.reshape(4,d);z=x@w.T;correct=bool(np.array_equal(z.argmax(1),TARGET));margins=[]
 for i,y in enumerate(TARGET):margins.extend(z[i,y]-np.delete(z[i],y))
 if not correct or min(margins)<1-1e-7:raise ValueError('Feasibility witness did not classify targets')
 return dict(passed=True,solver='scipy.optimize.linprog highs',minimum_margin=float(min(margins)),weights=w[:,:-1].tolist(),bias=w[:,-1].tolist())

def training_cue(update):
 if update not in range(1,1001):raise ValueError('Outside fixed training budget')
 return 'A' if update%2 else 'B'

def hash_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()

def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def append(text):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+text.rstrip()+'\n')

def codes():return {str(p.relative_to(ROOT)):hash_file(p) for base in ['experiments','tests'] for p in (ROOT/base/'architecture_averaging_factorial').glob('*.py')}
