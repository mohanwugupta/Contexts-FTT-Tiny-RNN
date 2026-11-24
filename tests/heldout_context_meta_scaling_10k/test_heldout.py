import pandas as pd
from experiments.heldout_context_meta_scaling_10k.heldout import recovered_correct_before,DIAGNOSTIC_DOSES

def test_recovery_uses_both_own_and_cross_with_correct_context():
 b=pd.DataFrame([dict(practiced='A',other='B',own_gain=.2,cross_gain=-.1),dict(practiced='B',other='A',own_gain=.3,cross_gain=.05)])
 assert abs(recovered_correct_before(.4,b,'A')-.65)<1e-12
 assert abs(recovered_correct_before(.4,b,'B')-.6)<1e-12
 assert DIAGNOSTIC_DOSES==[960,3840,7680,10000]
