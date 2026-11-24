"""Continue the original keyed generator, reserving corrected diagnostic paths."""
from contextlib import contextmanager
import numpy as np
from experiments.heldout_context_meta import protocol as source
DOSES=[0,240,960,1920,3840,5760,7680,10000]
NEW_DOSES=[1920,3840,5760,7680,10000]
AUDIT_EPISODES=sorted({961,962,10000}|{b*240+1 for b in range(4,42)}|set(NEW_DOSES))

def execution_prefix(block):return [ep for ep in block['episodes'] if 960<ep['episode']<=10000]

def reserve_block(seed,block,seen,reserved):
 reserved=set(reserved);original=source.draw;rejected=[];rng_end={}
 def reserved_draw(rng,prefix=None,branch_not=None):
  while True:
   path=original(rng,prefix,branch_not);rng_end.update(rng.bit_generator.state)
   if tuple(path) not in reserved:return path
   rejected.append(dict(path=path,reason='reserved_corrected_v2_diagnostic_sequence',prefix=prefix,branch_not=branch_not))
 try:
  source.draw=reserved_draw
  result=source.make_block(seed,block,seen)
 finally:source.draw=original
 result['reserved_rejections']=rejected
 result['generation_rng_end']=rng_end
 result['execution_count']=len(execution_prefix(result))
 result['unused_after_10000']=240-result['execution_count']
 assert not reserved.intersection(tuple(p) for e in result['episodes'] for p in e['paths'])
 return result
