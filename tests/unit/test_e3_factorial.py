import numpy as np
import pytest
import torch
from context_sequence.training.separation_regularizers import weak_loss, strong_loss, closure_loss
from context_sequence.training.factorial import CanonicalSRN, initialize, train_run, familiar_metrics, pilot_choice, objective, training_schedule


def test_E3_T01_T02_T03_T04_regularizer_gradients_and_stream():
    a=torch.tensor([[.1,0.]],requires_grad=True);b=torch.zeros_like(a)
    w=weak_loss(a,b);w.backward();assert a.grad[0,0]>0
    a.grad.zero_();s=strong_loss(a,b,1.5);s.backward();assert a.grad[0,0]<0
    assert strong_loss(a*100,b,1.5).item()==0
    z=torch.zeros((7,8),requires_grad=True);strong_loss(z,torch.zeros_like(z),2.5).backward();assert torch.isfinite(z.grad).all()
    models=[initialize(3) for _ in range(4)]
    for m in models[1:]:
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in models[0].state_dict().items())
    schedule=training_schedule();assert len(schedule)==170 and schedule[:10]==['A']*10 and schedule[10:20]==['B']*10


def test_E3_T05_T06_T07_stopgradient_recursive_neutral():
    m=initialize(1);seq=(0,1,2,0,3,1,2,3)
    a=m.trajectory(seq,[1,0,0])[1];b=m.trajectory(seq,[0,1,0])[1]
    a.retain_grad();b.retain_grad();loss,target,q=closure_loss(m,seq,a,b)
    assert not target.requires_grad
    loss.backward();assert a.grad is None and b.grad is None
    expected=m.trajectory(seq,[.5,.5,0])[1]
    torch.testing.assert_close(q,expected)
    assert torch.isfinite(m.recurrent.weight_hh_l0.grad).all()


def test_E3_T08_T09_T10_T12_training_and_evaluator():
    settings={'lambda_sep':.1,'lambda_close':.3,'margin':2.5}
    m=initialize(2);parts=objective(m,(0,1,2,0,3,1,2,3),'A','F11',settings)
    torch.testing.assert_close(parts['total'],parts['ce']+.1*parts['separation']+.3*parts['closure'])
    m,log,states=train_run(2,'F01',settings)
    assert len(log)==170 and len(states)==171
    assert log.parameter_hash.nunique()==170 and log.parameters_changed.all()
    assert set(log.training_context)=={'A','B'}
    assert (m.recurrent.weight_ih_l0[:,6]==0).all()
    assert familiar_metrics(m)['nonconstant_prediction']
    from context_sequence.training.factorial import numpy_model
    from context_sequence.dynamics.closure import natural
    for cue in ([1.,0.,0.],[0.,1.,0.]):
        logits,hidden=m.trajectory((0,1,2,0,3,1,2,3),cue)
        frozen=natural(numpy_model(m),cue)
        np.testing.assert_allclose(frozen['hidden'],hidden.detach().numpy(),atol=2e-4,rtol=0)
        np.testing.assert_allclose(frozen['logits'],logits.detach().numpy(),atol=2e-3,rtol=0)
    with pytest.raises(ValueError):objective(m,(0,1,2,0,3,1,2,3),'Novel30_A','F01',settings)


def test_E3_T11_T13_pilot_rejects_missing_arms_and_probes():
    import pandas as pd
    with pytest.raises(ValueError):pilot_choice(pd.DataFrame([{'probe':'Mixed'}]),{})
