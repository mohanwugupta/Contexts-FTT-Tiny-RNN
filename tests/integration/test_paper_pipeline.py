from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from context_sequence.experiments.engine import evaluate_canonical, evaluate_constructive, seed_summary, lag_summary
from context_sequence.experiments.matching import match_acquisition
from context_sequence.paper.pipeline import optimizer_firewall
from context_sequence.evaluation.rollout import Model
def toy():
    rng=np.random.default_rng(4)
    return Model(rng.normal(size=(8,7)).astype("float32")*.2,rng.normal(size=(8,8)).astype("float32")*.2,np.zeros(8,"float32"),rng.normal(size=(4,8)).astype("float32"),np.zeros(4,"float32"))
CUES={'A':np.array([1,0,0]),'B':np.array([0,1,0]),'Mixed':np.array([.5,.5,0]),'NoContext':np.zeros(3)}
RECORD={'cohort':'canonical','condition':'High_monotonic','family':'High_monotonic','seed':0,'checkpoint_step':170,'checkpoint_sha256':'toy','training_objective':'test','model_family':'canonical_srn'}


def test_I7_L6_F2_end_to_end_no_optimizer():
    import torch
    with optimizer_firewall():
        with pytest.raises(RuntimeError):torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        tables=evaluate_canonical(toy(),CUES,RECORD)
        constructive=evaluate_constructive(toy(),CUES,RECORD)
    local=tables['a5_cue_generation_transition'];local=local[local.analysis_kind.eq('local')]
    assert local.groupby(['probe','transition']).prestate_hash.nunique().eq(1).all()
    assert constructive['intervention'].eq('natural').all()
    maintenance=tables['a4_maintenance_transition']
    keys=['probe','position','transition']
    def compare(probe,a,b):
        f=maintenance[maintenance.probe.eq(probe)]
        x=f[f.continuation.eq(a)].set_index(keys).raw_ce;y=f[f.continuation.eq(b)].set_index(keys).raw_ce
        assert np.array_equal(x,y)
    compare('Mixed','D0','D2');compare('NoContext','D0','D1')


def test_S4_positions_aggregate_within_seed():
    tables=evaluate_canonical(toy(),CUES,RECORD)
    s=seed_summary(tables['a4_maintenance_transition'])
    assert not s.duplicated(['cohort','condition','seed','probe','intervention','continuation','analysis_kind']).any()
    lag=lag_summary(tables['a4_maintenance_transition'])
    assert lag.groupby('lag').n_positions.first().to_dict()=={0:7,1:6,2:5,3:4,4:3,5:2,6:1}


def test_A2_matching_never_uses_probe_outcomes():
    rows=[]
    for family in ['High_monotonic','Low_monotonic']:
        for step in [10,170]:
            for probe in ['A','B','Mixed']:
                rows.append({'family':family,'seed':0,'training_sequence_index':step,'probe':probe,'intervention':'no_patch','raw_sequence_loss':1.})
    df=pd.DataFrame(rows);a=match_acquisition(df)[1]
    df.loc[df.probe.eq('Mixed'),'raw_sequence_loss']=999
    b=match_acquisition(df)[1];pd.testing.assert_frame_equal(a,b)


def test_R2_unique_keys():
    from context_sequence.paper.pipeline import validate_keys
    f=evaluate_constructive(toy(),CUES,RECORD);validate_keys(f)
    with pytest.raises(ValueError):validate_keys(pd.concat([f,f.iloc[:1]]))


def test_clean_import_has_no_historical_package_dependency():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    forbidden={'confirmatory_srn','phase1i_context_composition','phase1j_context_stability','phase1k_cue_generation','phase1l_contrast_closure','phase1f_shared_backbone','phase1h_plasticity_audit','human_configured_state'}
    for p in (root/'src/context_sequence').rglob('*.py'):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.ImportFrom) and n.module:assert n.module.split('.')[0] not in forbidden
