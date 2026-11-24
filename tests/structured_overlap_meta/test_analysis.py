import pandas as pd,numpy as np,pytest
from experiments.structured_overlap_meta.analysis import paired_table

def test_pair_reference_is_a_not_c_and_seed_unit():
 f=pd.DataFrame([dict(seed=s,arm=a,dose=480,x=s+v) for s in range(6) for a,v in [('A',3),('B',2),('C',8)]])
 raw,ci=paired_table(f,['dose'],['x'],list(range(6)),['B','C'])
 assert raw[raw.candidate=='B'].effect.eq(-1).all()
 assert raw[raw.candidate=='C'].effect.eq(5).all()
 assert ci.n.eq(6).all() and set(ci.contrast)=={'B-A','C-A'}
 with pytest.raises(ValueError):paired_table(f,['dose'],['x'],list(range(1,6)),['B'])
