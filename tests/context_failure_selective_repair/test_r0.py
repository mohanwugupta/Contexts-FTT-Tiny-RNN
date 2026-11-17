import copy
import numpy as np
import pandas as pd
import pytest
from experiments.context_failure_selective_repair.audit import validate_rows, validate_cues, fidelity
from experiments.context_mixing_sequence_continuity.source_adapter import validate_record, load_original, records
from experiments.context_mixing_sequence_continuity.config import ROOT


def fixture():
    return pd.DataFrame([dict(seed=0,k=3,source_context='A',branch='B',schedule='pulse',cue='Mixed',transition=t,lag=t-3,token=[0,1,2,0,3,1,2,3][t-1],target=[0,1,2,0,3,1,2,3][t],cue_0=(0 if t==3 else .5),cue_1=(1 if t==3 else .5),cue_2=0.,cohort='canonical',condition='High_monotonic',checkpoint_sha256=records()[0]['checkpoint_sha256'],intervention_type='cue_only',model_optimizer_updates=0) for t in range(3,8)])

@pytest.mark.parametrize('field,value',[('schedule','persistent'),('token',3),('target',0),('cue_0',.7),('cue_2',1),('condition','Low_monotonic'),('checkpoint_sha256','bad')])
def test_reject_bad_reuse(field,value):
    q=fixture();q.loc[q.index[-1],field]=value
    with pytest.raises(ValueError):validate_rows(q,records())

def test_good_schedule_and_bad_cues():
    assert validate_rows(fixture(),records())
    assert validate_cues({'A':[1,0,0],'B':[0,1,0],'Mixed':[.5,.5,0]})
    for bad in [[1,1,0],[.5,.5,1]]:
        with pytest.raises(ValueError):validate_cues({'A':[1,0,0],'B':[0,1,0],'Mixed':bad})

@pytest.mark.parametrize('field,value',[('condition','High_sequence'),('seed',20),('checkpoint_step',169),('cohort','C0')])
def test_checkpoint_identity(field,value):
    r=copy.deepcopy(records()[0]);r[field]=value
    with pytest.raises(ValueError):validate_record(r)

def test_checkpoint_checksum():
    r=copy.deepcopy(records()[0]);r['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError):load_original(ROOT,r)

def test_replay_tolerance_not_zero_fill():
    a=pd.DataFrame({'hidden_0':[.3],'logit_0':[1.],'probability_0':[.25]})
    assert fidelity(a,a)['passed']
    b=a.copy();b.loc[0,'probability_0']+=1e-5
    with pytest.raises(ValueError):fidelity(a,b)
    with pytest.raises(ValueError):fidelity(a,b.iloc[:0])
