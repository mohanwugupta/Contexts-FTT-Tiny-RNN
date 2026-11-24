import pandas as pd,numpy as np,pytest
from experiments.high_scale_constrained_meta.final_analysis import paired

def fixture():return pd.DataFrame([dict(seed=s,arm=c+'-'+phase,dose=240,x=s+(2 if c=='P' else 0)*(1 if phase=='R' else 3)) for s in range(6) for phase in ['R','F'] for c in ['U','P']])
def test_reference_seed_unit_and_phases_not_pooled():
 raw,ci=paired(fixture(),['phase','dose'],['x']);assert set(raw.contrast)=={'P-U'} and len(raw)==12 and ci.n.eq(6).all();assert raw[raw.phase.eq('R')].effect.eq(2).all() and raw[raw.phase.eq('F')].effect.eq(6).all()
 bad=fixture();bad.loc[bad.seed.eq(5),'seed']=6
 with pytest.raises(ValueError):paired(bad,['phase','dose'],['x'])
