"""Fixed current-token/blocked-schedule experiment, without positional inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/recurrent_learning_transfer/v1'
SEQ=np.array([0,1,2,0,3,1,2,3]);TARGET=SEQ[1:];ARCHS=['RNN-Linear','RNN-Tanh'];DIAG=[0,10,20,40,80,120,170,330];ENDPOINTS=[170,330]
CUES={k:v['vector'] for k,v in json.loads((ROOT/'configs/novelty_registry_v1.json').read_text())['cues'].items()}
SCHEDULE=make_training_schedule(33,10).context.tolist()
CONFIG=dict(seeds=list(range(20)),architectures=ARCHS,current_token_features=4,context_features=3,hidden=8,source_std=2.,initial_recurrent_spectral_norm=.9,post_update_recurrent_projection=False,scientific_updates_per_model=330,total_scientific_updates=13200,diagnostic_checkpoints=DIAG,total_diagnostic_updates=640,primary=170,secondary=330,schedule='A10/B10,33 alternating blocks,startsA',learning_rate=.005,joint_clip=1.,lambda_execution=.2,within_gain_min=1e-4,direct_specificity_min=.001,direction_excess_min=0.,own_negative_tolerance=1e-6,cycle_cost_min=.05,cycle_fraction_min=.5,reacquisition_gain=.05,strong_familiar_ce=.5,strong_familiar_accuracy=6/7,reliability_strong_min=16,mixed_success_ce=.5,mixed_success_accuracy=6/7,mixed_success_gap=.1,positive_gap=1e-6,material_gap=.1,double_atol=1e-10,double_rtol=1e-12,float32_bound_factor=64,bootstrap_draws=10000,bootstrap_seed=20260925,D1_adapter_hidden_atol=2e-4,D1_adapter_logit_atol=.002)

def tokens():return torch.eye(4)[torch.tensor(SEQ[:-1])]

def training_cue(update):
 if update not in range(1,331):raise ValueError('Outside registered330 updates')
 return SCHEDULE[update-1]

def hash_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def append(text):
 with (OUT/'scratchpad.md').open('a') as f:f.write('\n'+text.rstrip()+'\n')

def codes():return {str(p.relative_to(ROOT)):hash_file(p) for base in ['experiments','tests'] for p in (ROOT/base/'recurrent_learning_transfer').glob('*.py')}
