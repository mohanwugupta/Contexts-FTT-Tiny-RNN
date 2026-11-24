"""Process-local output/load adapters around unchanged predecessor operations."""
from contextlib import contextmanager
from .common import *
from experiments.heldout_context_meta import target as original_target
from experiments.heldout_context_meta import heldout as original_heldout
@contextmanager
def bind(module,**values):
 previous={k:getattr(module,k) for k in values}
 try:
  for k,v in values.items():setattr(module,k,v)
  yield
 finally:
  for k,v in previous.items():setattr(module,k,v)

def target_job(job):
 (OUT/'target').mkdir(parents=True,exist_ok=True)
 with bind(original_target,OUT=OUT,initial=initial,codes=codes):return original_target.target_run(job)

def heldout_job(job):
 with bind(original_heldout,OUT=OUT,initial=initial):return original_heldout.heldout_run((*job,'v2'))
