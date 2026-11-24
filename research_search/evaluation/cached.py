"""Tier0 expression control: cached arrays only, not a separately trained model."""
import numpy as np
from experiments.a06_cached_readout_comparison.analysis import static_logits
from .scoring import observations

def quarter_readout(logits,bias,updates,path,seed,candidate):
    z=static_logits(np.asarray(logits),np.asarray(bias));f,p=observations(z,updates,path,seed,candidate+'-static-quarter-view')
    f['has_practice_history']=False;p['has_practice_history']=False
    return f,p
