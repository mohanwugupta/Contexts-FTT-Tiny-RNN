from experiments.heldout_context_meta_scaling_10k.rng_audit import trace_block
from experiments.heldout_context_meta_scaling_10k.protocol import reserve_block

def test_midblock_state_recovery_preserves_exact_library():
 expected=reserve_block(3,42,set(),set());reg,states=trace_block(3,42,set(),set())
 assert reg==expected and len(states)==240
 assert states[-1]==reg['generation_rng_end']
 assert states[159]!=states[-1]
 assert reg['episodes'][159]['episode']==10000
