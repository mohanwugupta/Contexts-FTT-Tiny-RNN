"""Reuse the exact predecessor update; omit only unaudited packed snapshots."""
from .common import *
from experiments.heldout_context_meta import fomaml as source

def restore(state,optimizer_state):
 model=source_model(state);outer=source.outer_optimizer(model);outer.load_state_dict(copy.deepcopy(optimizer_state))
 assert core.hash_state(model,o=outer)==core.hash_state(source_model(state),o=outer)
 return model,outer

def compact_episode(model,outer,ep,prior,arm,full=False):
 if full:return source.episode(model,outer,ep,prior,arm)
 original=source.packed
 try:
  source.packed=lambda *a,**k:(np.empty((0,),np.float32),np.empty((0,),np.float32))
  return source.episode(model,outer,ep,prior,arm)
 finally:source.packed=original
