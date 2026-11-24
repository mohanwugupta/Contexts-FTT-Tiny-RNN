import copy
import numpy as np
import pandas as pd
import pytest
from experiments.research_search_integration_v1_1.cached import block_effects, validate_grid


def grid():
    return pd.DataFrame([dict(update=u,cue=c,ce=2-u*.001+(c=='B')*.1,correct_count=3,margin=-1.) for u in range(331) for c in ['A','B','C','Mixed-average','NoContext']])


def test_signed_gains_identical_boundaries_and_floor():
    f=grid();f.loc[(f['update']==10)&(f.cue=='C'),'ce']=3
    g=block_effects(f)
    assert len(g)==33*5
    a=g[(g.block==1)&(g.cue=='C')].iloc[0]
    assert a.start==0 and a.end==10 and a.practiced=='A'
    assert a.gain<0 and a.start_ce==2
    assert g.loc[g.block==2,'practiced'].eq('B').all()
    f.loc[(f['update']==0)&(f.cue=='A'),'ce']=.01
    assert block_effects(f).query("block==1 and cue=='A'").iloc[0].start_floor


def test_exact_grid_duplicate_missing_and_wrong_cue():
    f=grid();validate_grid(f)
    for bad in [f.iloc[1:],pd.concat([f,f.iloc[[0]]]),f.replace({'NoContext':'Novel'})]:
        with pytest.raises(ValueError):validate_grid(bad)
