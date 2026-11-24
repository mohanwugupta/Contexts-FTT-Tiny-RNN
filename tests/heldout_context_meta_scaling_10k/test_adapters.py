from experiments.heldout_context_meta_scaling_10k.adapters import bind, target_job, heldout_job
from experiments.heldout_context_meta_scaling_10k.protocol import NEW_DOSES
from experiments.heldout_context_meta import target as old

def test_namespace_adapter_restores_source_globals_and_fixed_doses():
 original=old.OUT;value=object()
 with bind(old,OUT=value):assert old.OUT is value
 assert old.OUT==original
 try:
  with bind(old,OUT=value):raise ValueError('fixture')
 except ValueError:pass
 assert old.OUT==original
 assert len([(s,p,o,a,d) for d in NEW_DOSES for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']])==800
