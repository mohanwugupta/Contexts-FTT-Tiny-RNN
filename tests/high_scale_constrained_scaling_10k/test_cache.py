import pandas as pd,pytest
from experiments.high_scale_constrained_scaling_10k import cache

def test_prefill_never_filters_outcomes_or_duplicates_physical_cases(tmp_path):
 reg=pd.DataFrame([dict(seed=s,scale=4,condition=c,phase='T',dose=10000) for s in range(6) for c in ['U','P']])
 assert len(cache.jobs(reg,tmp_path))==12
 (tmp_path/'s00_scale4_U_T_e10000.json').write_text('{}')
 assert len(cache.jobs(reg,tmp_path))==11
 with pytest.raises(ValueError):cache.jobs(pd.concat([reg,reg.iloc[:1]]),tmp_path)
