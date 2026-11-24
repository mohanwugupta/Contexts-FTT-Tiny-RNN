"""Thin objective adapters; no new model, loss, optimizer or task adaptation."""
from .common import *
from .protocol import epsilon
from experiments.high_scale_constrained_meta.runtime import episode as original_episode
from experiments.meta_objective_balance import runtime as balanced
from experiments.structured_overlap_meta import runtime as overlap
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.heldout_context_meta_scaling_10k.verify import pack_state
from experiments.meta_objective_balance.runtime import paired_update
from experiments.structured_overlap_meta.runtime import meta_episode

def fresh_pack(model,opt):
 if opt['state']:return pack_state(model,opt)
 v=np.concatenate([x.detach().numpy().ravel() for x in model.values()]);return np.stack([v,np.zeros_like(v),np.zeros_like(v)])

def meta_step(m,o,e,arm,k):
 if arm in ['R','T']:
  r=original_episode(m,o,e,'Reptile' if arm=='R' else 'FOMAML','U',dict.fromkeys(['token','context_A','context_B','context_C','output'],2.),epsilon(k) if arm=='R' else None)
  assert all(not x['enabled'] and not x['projected'] for x in r['projection']);r.update(inner_updates=20,episodes=1,outer_updates=int(arm=='T'),interpolations=int(arm=='R'),sequence_presentations=20,query_examples=0 if arm=='R' else r['query_count'],query_gradient_evaluations=int(arm=='T'))
 elif arm=='B':
  with bind(balanced,pack_state=fresh_pack):r=paired_update(m,o,e,.5,0.)
  r.update(gradient=r['combined_gradient'],episodes=2,interpolations=0,sequence_presentations=40,query_examples=5,query_gradient_evaluations=3)
  # Native beta=0 still computes the critical-only gradient for its unchanged audit.
 elif arm in ['O1','O2']:
  with bind(overlap,pack_state=fresh_pack):r=meta_episode(m,o,e)
  r.update(gradient=r['combined_gradient'],episodes=1,interpolations=0,sequence_presentations=40,query_gradient_evaluations=3)
 else:raise ValueError('Unregistered objective')
 return r
