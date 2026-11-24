from pathlib import Path
import torch
from experiments.high_scale_meta_progression.audit import archived_target
from experiments.high_scale_meta_progression.protocol import scaled_initial,restore_target

def test_exact_scale4_6_archived_history_and_full_adam_continuation():
 for seed in range(6):
  for scale in [4,6]:
   r=archived_target(seed,scale);b=torch.load(r['states'],weights_only=False);assert len(b['states'])==331
   assert all(torch.equal(v,b['states'][0]['model'][k]) for k,v in scaled_initial(seed,scale).items())
   m,p,o=restore_target(b['states'][330]);assert all(float(v['step'])==330 for v in o.state.values())
