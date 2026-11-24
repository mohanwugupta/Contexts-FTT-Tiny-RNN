from experiments.heldout_context_meta.heldout_correction import fresh_collection
from experiments.heldout_context_meta.protocol import make_block

def test_every_individual_sequence_unseen_not_just_task_pair():
 training=make_block(0,1,set());seen={tuple(p) for e in training['episodes'] for p in e['paths']};r=fresh_collection(0,seen);v=[tuple(p) for e in r['episodes'] for p in e['paths']]
 assert len(v)==24 and len(set(v))==24 and not seen.intersection(v)
