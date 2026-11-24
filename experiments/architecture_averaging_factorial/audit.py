"""Mandatory pretraining input/source/math/finite-gradient gate."""
import json,shutil,inspect
import numpy as np
import pandas as pd
import torch
from .protocol import *
from .model import make_models,midpoint_control
from .training import make_policy,full_loss


def validate_cue_registry(cues):
 expected={k:v['vector'] for k,v in json.loads((ROOT/'configs/novelty_registry_v1.json').read_text())['cues'].items()}
 if set(cues)!=set(expected) or any(not np.array_equal(cues[k],expected[k]) for k in expected):raise ValueError('Actual canonical cue registry mismatch')
 if cues['A']!=[1,0,0] or cues['B']!=[0,1,0] or cues['Mixed']!=[.5,.5,0] or any(v[2]!=0 for v in cues.values()):raise ValueError('Wrong physical/third-channel definition')
 return True

def validate_legacy(d):
 if (d.hidden_identity_residual>1e-11).any() or (d.affine_residual>1e-11).any() or (d.observed_float32_residual>d.roundoff_bound).any():raise ValueError('Inherited D1 algebra/roundoff audit failed')
 return True

def run():
 from confirmatory_srn.config import ConfirmatoryConfig
 from experiments.context_mixing_sequence_continuity.mixing_decomposition import exact_components
 prior=ROOT/'results/context_mixing_sequence_continuity/v1';old=json.loads((prior/'manifests/sealed_manifest.json').read_text());inputs={}
 names=['confirmatory_srn/model.py','confirmatory_srn/training.py','confirmatory_srn/config.py','execution_objective_search/config.py','execution_objective_search/objectives.py','ftt_srn_execution_v2/config.py','configs/novelty_registry_v1.json','experiments/context_mixing_sequence_continuity/mixing_decomposition.py']
 for name in names:
  want=old['inputs'].get(name,old['code_hashes'].get(name))
  if want is None:
   snapshot=json.loads((ROOT/'artifacts/manifests/pre_freeze_snapshot.json').read_text());want=next((r['sha256'] for r in snapshot['files'] if r['path']==name),None)
  if want is None:raise ValueError('Source not covered by prior seal: '+name)
  if hash_file(ROOT/name)!=want:raise ValueError('Source changed: '+name)
  inputs[name]=want
 for name in ['tables/d1_decomposition.csv','tables/d0_native_replay.csv','tables/d0_checkpoint_audit.csv','D1_MIXING_BREAKDOWN.md']:
  rel=str((prior/name).relative_to(ROOT));inputs[rel]=old['files'][name]
  if hash_file(ROOT/rel)!=inputs[rel]:raise ValueError('Prior evidence hash mismatch')
 checkpoints=pd.read_csv(prior/'tables/d0_checkpoint_audit.csv')
 if len(checkpoints)!=20 or not checkpoints.condition.eq('High_monotonic').all() or not checkpoints.checkpoint_step.eq(170).all():raise ValueError('Historical reference cohort mismatch')
 for r in checkpoints.itertuples():
  if hash_file(ROOT/r.path)!=r.checkpoint_sha256:raise ValueError('Original checkpoint changed')
  inputs[r.path]=r.checkpoint_sha256
 inputs['docs/architecture_averaging_factorial/PRD.md']=hash_file(ROOT/'docs/architecture_averaging_factorial/PRD.md')
 cfg=ConfirmatoryConfig();cfg.validate()
 if tuple(cfg.fixed_sequence)!=tuple(SEQ):raise ValueError('Source sequence mismatch')
 if cfg.high_scale!=2. or cfg.lambda_execution!=.2 or cfg.normative_c_time!=.5 or cfg.normative_c_risk!=1. or cfg.normative_gamma!=2. or cfg.learning_rate!=.005 or cfg.gradient_clip!=1.:raise ValueError('Unexpected original source contract')
 validate_cue_registry(CUES);validate_windows(windows());dump(OUT/'audit/input_feasibility.json',linear_feasibility(windows()));dump(OUT/'audit/config.json',CONFIG)
 oldd=pd.read_csv(prior/'tables/d1_decomposition.csv');validate_legacy(oldd);shutil.copyfile(prior/'tables/d1_decomposition.csv',OUT/'tables/historical_d1_decomposition.csv')
 # Exercise imported exact D1 equations, not a new decomposition or original rollout.
 d1=exact_components(np.array([1.,-1.]),np.array([0.,.5]),np.array([.2,-.1]))
 if np.max(abs(d1['e']-d1['j']-d1['r']))>1e-11:raise ValueError('Imported D1 equations changed')
 historical=pd.read_csv(prior/'tables/d0_native_replay.csv');hist=[]
 for seed,g in historical.groupby('seed'):
  ce=g.groupby('cue').raw_ce.mean();hist.append(dict(cohort='historical_canonical_High_not_factorial',seed=seed,original_update=170,A_ce=ce.A,B_ce=ce.B,Mixed_ce=ce.Mixed,familiar_ce=(ce.A+ce.B)/2,gap=ce.Mixed-(ce.A+ce.B)/2,input='current-token onehot4, not newwindow15',schedule='historical A10/B10, not newABAB'))
 pd.DataFrame(hist).to_csv(OUT/'tables/historical_reference.csv',index=False)
 math=[];finite=[]
 torch.set_num_threads(1)
 for seed in range(20):
  for name,m in make_models(seed).items():
   if not m.nonlinear:math.append(dict(seed=seed,architecture=name,**midpoint_control(m)))
   for cue in ['A','B']:
    policy=make_policy();z,h=m(torch.tensor(windows()),torch.tensor(CUES[cue]));total,ce,ex=full_loss(z,policy);grads=torch.autograd.grad(total,list(m.parameters())+list(policy.parameters()))
    if not torch.isfinite(z).all() or not torch.isfinite(h).all() or not all(torch.isfinite(v).all() for v in grads):raise ValueError('Nonfinite pretraining output/gradient')
    finite.append(dict(seed=seed,architecture=name,cue=cue,finite=True))
 pd.DataFrame(math).to_csv(OUT/'tables/pretraining_linear_controls.csv',index=False);pd.DataFrame(finite).to_csv(OUT/'tables/pretraining_finite_gradients.csv',index=False)
 code=codes();gate=dict(passed=True,inputs=inputs,code=code,linear_controls=40,finite_gradient_checks=160,original_models_unchanged=20,optimizer_fixture_only_not_scientific=True,source_objective_fixture_log='audit/training_green.log',tests='audit/pretraining_refactor.log')
 dump(OUT/'audit/pretraining_gate.json',gate)
 audit='''# PRD implementation audit

Source: confirmatory_srn/model.py:_initialize uses iid Normal(0,init_scale=2) for both original recurrent matrices and the sequence head; zeros both hidden biases/output bias and neutral third cue column. Original execution policy has two zero parameters. confirmatory_srn/training.py applies mean7-token CE+.20*mean execution cost, detached confidence/uncertainty, Adam(lr.005, betas.9/.999,eps1e-8,weight_decay0) and global L2 clipping1.0 over model plus policy, then zeroes third-cue weights. These source modules and objective functions are hash-verified against the prior seal. Objective/policy/model gradients, clipping norm and one Adam update pass an independent mapped-source fixture in double arithmetic. Actual production uses float32.

The new15-dimensional causal token window is ordered oldest→current with5-way BOS slots. The LP witness classifies allseven targets with margin≥1; current-token-only negative fixture is infeasible. Cues are loaded EXACTLY from configs/novelty_registry_v1.json, including allfour individual Novel vectors, not manually redefined by desired predictions. Initially a mathematical equivalent cue expression was replaced with direct registry loading before training, eliminating bit-level trigonometric differences; no protected outcomes had been evaluated.

The four architectures share identical initial input/context/output tensors and zero biases per seed; recurrent arms share a matrix normalized once to spectral norm.9. FF has no recurrent weight or hidden feedback. Both source hidden biases are retained in allarms (effective b_h=b_ih+b_hh), so recurrence adds exactly64 scalars. FF model196 parameters/RNN260; independent policy2 each. No extra layers/normalization/embeddings or nonlinear output. Each scientific update resets hidden state and uses A/B whole sequences in exact ABAB order, not the historical block schedule.

Pretraining gates:40 original-seed linear midpoint controls pass double hidden/logit equality and CE convexity, with independent float32 propagated summand bounds;160 initial A/B output/gradient checks are finite. The linear identity is algebraic, not a statistical null, and softmax probabilities need not average. Linear target feasibility, neutral channel, reset, schedule and protected-cue exclusion have behavioral negative fixtures. No pilot fits, protected searches or scientific extra updates.

The new1000-update study is expressly user-authorized despite the repository's default no-training rule. It creates80 distinct new networks and saves170/330/660/1000 endpoints, never substitutes canonical High models. Both the window input and recurrent initialization differ from historical SRNs. Historical20 update170 checkpoints are only hash-checked; D1 decomposition/audit is reused byte-for-byte. Its imported exact e=j+r formulas and observed-roundoff bound pass; no residual norm is a fraction of CE caused.

All80 models and320 planned checkpoint slots will be registered and sealed before any Mixed/NoContext/Novel evaluation. Numerical failures retain planned missing endpoints, not substituted successful models. No later tuning, restarts or checkpoint selection. Scratchpad/config fix thresholds, onset, saturation, seed inference and qualified coverage before training.
'''
 with (OUT/'PRD_IMPLEMENTATION_AUDIT.md').open('x') as f:f.write(audit)
 append('## Pretraining gate passed\n40 linear algebra controls,160 finite-gradient checks, all20 historical checkpoint hashes and reused D1 audit passed. Source objective gradient/clipping/Adam fixture and behavioral RED→GREEN logs retained. Scientific updates so far0; no protected outcomes. No parameter changes from protected performance.')
 print(json.dumps(dict(passed=True,linear_controls=40,finite_gradient_checks=160,scientific_updates=0)),flush=True)

if __name__=='__main__':run()
