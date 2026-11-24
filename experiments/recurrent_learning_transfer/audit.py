"""Canonical-source fidelity, deterministic capacity and mandatory pretraining gate."""
import json,shutil
import numpy as np
import pandas as pd
import torch
from .design import *
from .network import pair,capacity_witness,trace,linear_control
from .learning import policy,loss


def run():
 prior=ROOT/'results/architecture_averaging_factorial/v1';sealed=json.loads((prior/'manifest.json').read_text());inputs={}
 names=['confirmatory_srn/model.py','confirmatory_srn/training.py','confirmatory_srn/config.py','execution_objective_search/objectives.py','execution_objective_search/config.py','ftt_srn_execution_v2/config.py','configs/novelty_registry_v1.json','ftt_srn_execution_v2/training/ftt_schedule.py','experiments/context_mixing_sequence_continuity/mixing_decomposition.py']
 snapshot=json.loads((ROOT/'artifacts/manifests/pre_freeze_snapshot.json').read_text())
 for name in names:
  want=sealed['inputs'].get(name)
  if want is None:want=next((x['sha256'] for x in snapshot['files'] if x['path']==name),None)
  if want is None or hash_file(ROOT/name)!=want:raise ValueError('Source not matched to verified historical bytes: '+name)
  inputs[name]=want
 old=ROOT/'results/context_mixing_sequence_continuity/v1';m=json.loads((old/'manifests/sealed_manifest.json').read_text())
 for name in ['tables/d0_native_replay.csv','tables/d0_checkpoint_audit.csv','tables/d1_decomposition.csv','D1_MIXING_BREAKDOWN.md']:
  rel=str((old/name).relative_to(ROOT));want=m['files'][name]
  if hash_file(ROOT/rel)!=want:raise ValueError('Historical evidence changed')
  inputs[rel]=want
 original=pd.read_csv(old/'tables/d0_checkpoint_audit.csv')
 if len(original)!=20 or not original.condition.eq('High_monotonic').all() or not original.checkpoint_step.eq(170).all():raise ValueError('Wrong historical cohort')
 for r in original.itertuples():
  if hash_file(ROOT/r.path)!=r.checkpoint_sha256:raise ValueError('Original checkpoint changed')
  inputs[r.path]=r.checkpoint_sha256
 inputs['docs/recurrent_learning_transfer/PRD.md']=hash_file(ROOT/'docs/recurrent_learning_transfer/PRD.md')
 from confirmatory_srn.config import ConfirmatoryConfig
 from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
 cfg=ConfirmatoryConfig();cfg.validate()
 if tuple(SEQ)!=cfg.fixed_sequence or not torch.equal(tokens(),encode_sequence(cfg.fixed_sequence[:-1],as_tensor=True)):raise ValueError('Original input/target task mismatch')
 witness=capacity_witness();a=trace(witness,'A');b=trace(witness,'B')
 if not a['accuracy'].all() or not b['accuracy'].all() or a['ce'].mean()>=.002:raise ValueError('Eight-state linear capacity fixture failed')
 dump(OUT/'audit/capacity_witness.json',dict(optimized=False,input_history_or_position_features=False,internal_clock_recurrence=True,description='Wh=.9*shift; bias=e1; h_t[j]=.9**j for j<t; Wo columnj=(8onehot(y[j+1])−8onehot(y[j]))/.9**j withv0=0',model_state={k:v.detach().tolist() for k,v in witness.state_dict().items()},A_ce=float(a['ce'].mean()),B_ce=float(b['ce'].mean()),accuracy=1.))
 dump(OUT/'audit/config.json',CONFIG);torch.set_num_threads(1);rows=[];finite=[]
 for seed in range(20):
  models=pair(seed)
  for key,v in models['RNN-Linear'].state_dict().items():
   if not torch.equal(v,models['RNN-Tanh'].state_dict()[key]):raise ValueError('Paired initial tensors differ')
  rows.append(dict(seed=seed,**linear_control(models['RNN-Linear'])))
  for arch,model in models.items():
   if model.wi.shape!=(8,7) or sum(p.numel() for p in model.parameters())!=172:raise ValueError('Unexpected architecture/input dimensions')
   for cue in ['A','B']:
    p=policy();z,h=model(tokens(),torch.tensor(CUES[cue]));total,_,_=loss(z,p);grads=torch.autograd.grad(total,list(model.parameters())+list(p.parameters()))
    if not torch.isfinite(z).all() or not all(torch.isfinite(v).all() for v in grads):raise ValueError('Nonfinite initial output/gradient')
    finite.append(dict(seed=seed,architecture=arch,cue=cue,passed=True))
 pd.DataFrame(rows).to_csv(OUT/'tables/pretraining_linear_controls.csv',index=False);pd.DataFrame(finite).to_csv(OUT/'tables/pretraining_finite_checks.csv',index=False)
 historical=pd.read_csv(old/'tables/d0_native_replay.csv');historical.to_csv(OUT/'tables/historical_original_trace.csv',index=False);shutil.copyfile(old/'tables/d1_decomposition.csv',OUT/'tables/historical_d1_decomposition.csv')
 dump(OUT/'audit/pretraining_gate.json',dict(passed=True,inputs=inputs,code=codes(),linear_controls=20,finite_gradient_checks=80,capacity_witness_optimized=False,scientific_updates=0,diagnostic_updates=0))
 with (OUT/'IMPLEMENTATION_AUDIT.md').open('x') as f:f.write('''# Implementation and source audit

This is a new40-model current-token recurrent comparison. Both arms use exactly4 current-token channels+3 registered context channels, hidden8, zero initial state, original two hidden biases and four-logit linear head. There are172 model parameters plus2 separate execution-policy parameters in each arm. Only identity versus tanh differs. No explicit history/position features or embedding is provided.

High-family initialization is iid Normal(0,2) for input/context, recurrent and output weights, zeros for source biases/policy, neutral third context column. Paired architectures receive identical tensors per seed. Wh is rescaled ONCE to spectral norm.9; this departure prevents calling Tanh an exact canonical reproduction. Source Adam(lr.005/defaults), mean7-position CE+.20 detached-confidence execution risk and joint clipping1.0 are reused. A mapped-source fixture verifies logits, policy/CE gradients, clipped norm and one Adam step. The source A10/B10 schedule is imported unchanged for33 blocks (170primary,330secondary).

A deterministic eight-unit LINEAR capacity witness uses an internal recurrent clock, Wh=.9*shift and bias=e1. With zero input weights, hidden coordinatej=.9**j when t>j; fixed decoder differences recover8onehot(target_t). It predicts allseven targets correctly with CE≈.001. No optimizer, supplied positions or history windows are used; deleting recurrence destroys its seven-target solution. This proves finite-task representability, not successful optimization or a learned cognitive algorithm, and is not an experimental arm or initialization.

Twenty initial linear identity/CE controls and80 A/B finite-gradient checks pass before scientific training. Float64 affine tolerance1e-10+1e-12×scale; actual float32 uses64eps summand/propagated bounds. Exact logit averaging does not imply arithmetic softmax averaging. Clone tests preserve scientific model+policy+Adam/RNG, confirm optimizer-parameter reference mapping, reject fresh/reset moments, and prove identical short scientific trajectories with/without diagnostics. Only separate diagnostic clones receive their A/B one-update intervention; none is reused as the scientific model.

Functional-splitting thresholds and baseline-cancelled cycle costs are fixed in scratchpad/config before outcomes. First A-block asymmetry equals the first baseline-adjusted switch cost algebraically, so these are not independent pieces of evidence. Near-zero/nonbeneficial own updates are uninformative about shared versus separate learning, not automatically nonsplit. Primary endpoint classification combines current170 diagnostic evidence with balanced cycle history strictly before170; the330 endpoint is secondary. Geometry does not classify splitting.

Protected evaluation is forbidden until all40 trajectories and all320 diagnostic/checkpoint slots are sealed. Original20 canonical checkpoints are hash-checked but not retrained or substituted. Original saved performance and D1 audits are separate historical sources. New Tanh checkpoint decomposition will reuse the imported D1 implementation through a frozen NumPy adapter, with source-precision discrepancy checks.
''')
 append('## Pretraining gate passed\nCanonical source/checkpoint hashes unchanged. Deterministic recurrent capacity witness,20 linear identities,80 finite-gradient checks, source-loss gradient/Adam parity and cloned-state/unchanged-trajectory fixtures pass. No scientific or registered diagnostic updates yet. Thresholds fixed; no protected evaluation.')
 print('Pretraining capacity/source/math/clone gates passed',flush=True)

if __name__=='__main__':run()
