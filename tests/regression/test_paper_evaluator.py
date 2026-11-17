from pathlib import Path
import numpy as np
import pytest
from context_sequence.provenance.registry import checkpoint_records, load_model, audit_inputs
from context_sequence.evaluation.rollout import rollout
from context_sequence.provenance.regression import historical_regression
ROOT=Path(__file__).resolve().parents[2]


def test_E4_E5_L7_historical_regression(tmp_path):
    records=checkpoint_records(ROOT)
    assert len(records)==120
    result=historical_regression(ROOT,records)
    assert result['passed'].all()
    assert result.query('check=="C0_identity"').shape[0]==20


def test_F3_R1_missing_immutable(tmp_path):
    record=checkpoint_records(ROOT)[0];model=load_model(ROOT,record);before=model.digest()
    rollout(model,[.5,.5,0]);assert model.digest()==before
    bad=dict(record,path='missing.pt')
    with pytest.raises(FileNotFoundError):load_model(ROOT,bad)


def test_F2_no_optimization(monkeypatch):
    import torch
    def forbidden(*a,**kw):raise AssertionError('optimizer forbidden')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    record=checkpoint_records(ROOT)[0];model=load_model(ROOT,record)
    assert len(rollout(model,[1,0,0])['raw_ce'])==7
