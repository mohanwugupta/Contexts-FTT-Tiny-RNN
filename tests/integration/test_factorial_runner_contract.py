"""Exercise the otherwise gated runner using untrained synthetic fixtures only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from context_sequence.training import factorial
from context_sequence.dynamics.pipeline import _factorial_evaluate
from context_sequence.task.novelty import registry_payload


def test_E3_E4_fixed_run_matrix_sequence_and_evaluator_without_training(tmp_path,monkeypatch):
    calls=[];sequence=(0,1,0,1,2,3,2,3)
    def forbidden(*a,**kw):raise AssertionError('No optimizer may run in this synthetic fixture test')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    def fixture(seed,condition,settings,tokens):
        calls.append((seed,condition,tuple(tokens)))
        model=factorial.initialize(seed)
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        # A simulated history is a fixture, not a claim that these weights were trained.
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        states=[state for _ in range(171)]
        log=pd.DataFrame({'initial_familiar_mean_ce':[np.log(4)]*170,'training_sequence_index':range(1,171),'training_context':factorial.training_schedule(),'seed':seed,'condition':condition})
        return model,log,states
    monkeypatch.setattr(factorial,'train_run',fixture)
    novel=registry_payload();(tmp_path/'novel.json').write_text(json.dumps(novel))
    out=tmp_path/'output'
    for d in ['checkpoints','tables','audit']:(out/d).mkdir(parents=True)
    config={'novelty_registry':'novel.json','novelty_registry_sha256':novel['registry_sha256'],'config_sha256':'synthetic-fixture-only'}
    fam,feas,summary,boot,records=_factorial_evaluate(tmp_path,out,config,'e4',[{'sequence_id':'S1','tokens':list(sequence)}],{'lambda_sep':.1,'lambda_close':.3,'margin':2.5})
    assert len(calls)==80 and len(set(calls))==80
    assert {s for s,_,_ in calls}==set(range(20))
    assert {c for _,c,_ in calls}=={'F00','F01','F10','F11'}
    assert {q for _,_,q in calls}=={sequence}
    assert fam.sequence_id.eq('S1').all() and len(fam)==80
    assert summary.sequence_id.eq('S1').all() and len(summary)==880
    assert boot['mean'].eq(0).all() and boot.n_seeds.eq(20).all()
    transitions=pd.read_csv(out/'tables/e4_probe_transition.csv')
    assert transitions.groupby('transition').target.first().tolist()==list(sequence[1:])
    registry=pd.DataFrame(records)
    assert registry.groupby('seed').initial_state_sha256.nunique().eq(1).all()
    gate=json.loads((out/'audit/e4_familiar_gate_frozen.json').read_text())
    assert gate['before_protected_probe_evaluation'] and gate['no_seed_exclusions']
