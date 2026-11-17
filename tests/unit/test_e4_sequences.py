import pytest
from context_sequence.task.replication import sequence_registry, validate_sequences, equivalence_class
from context_sequence.analysis.aggregation import seed_summary, paired_contrasts
import pandas as pd


def test_E4_T01_T02_T03_registry():
    r=sequence_registry();f=validate_sequences(r)
    assert f.valid.all() and len(f)==4
    seq=[tuple(x['tokens']) for x in r['sequences']]
    assert len(set(equivalence_class(s) for s in seq))==4
    assert seq[0]==(0,1,2,0,3,1,2,3)
    for s in seq:
        assert len(s)==8 and all(s.count(i)==2 for i in range(4))
        assert all(a!=b for a,b in zip(s,s[1:]))


def test_E4_T06_T07_clustering_and_missing_pairs():
    f=pd.DataFrame([dict(sequence_id=q,cohort='new',condition=c,seed=s,probe='A',transition=t,raw_ce=s+t+(c=='F01')) for q in ['S1','S2'] for c in ['F00','F01'] for s in range(3) for t in range(1,8)])
    out=seed_summary(f);assert len(out)==12
    assert not out.duplicated(['sequence_id','condition','seed','probe']).any()
    c=paired_contrasts(out,contrasts={'closure_weak':{'F01':1,'F00':-1}})
    assert len(c)==6 and c.effect.eq(1).all()
    with pytest.raises(ValueError):paired_contrasts(out.iloc[1:],contrasts={'closure_weak':{'F01':1,'F00':-1}})
