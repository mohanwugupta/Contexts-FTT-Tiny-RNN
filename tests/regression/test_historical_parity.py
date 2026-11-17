"""Real source regressions skip explicitly when optional backing artifacts are absent."""
from pathlib import Path
import numpy as np
import pytest
import torch
from context_sequence.training.factorial import initialize, objective, training_schedule
from context_sequence.provenance.registry import checkpoint_records, load_model
from context_sequence.dynamics.closure import natural, references, single_patch_replay
from context_sequence.evaluation.rollout import rollout,midpoint
ROOT=Path(__file__).resolve().parents[2]


def test_E1_T08_E3_model_gradient_schedule_recreation():
    try:
        from confirmatory_srn.model import ConfirmatoryElmanSRN
        from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
    except ImportError as e:pytest.skip('blocked_missing_source: '+str(e))
    torch.manual_seed(17);old=ConfirmatoryElmanSRN(hidden_dim=8,init_scale=2.,context_alpha=1.)
    new=initialize(17)
    assert set(old.state_dict())==set(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    seq=(0,1,2,0,3,1,2,3);tokens=torch.nn.functional.one_hot(torch.tensor(seq[:-1]),4).float()
    z,h,_=old(tokens,torch.tensor([1.,0.,0.]));zn,hn=new.trajectory(seq,[1.,0.,0.])
    torch.testing.assert_close(z[0],zn,atol=0,rtol=0);torch.testing.assert_close(h[0],hn,atol=0,rtol=0)
    loss=torch.nn.functional.cross_entropy(z[0],torch.tensor(seq[1:]));loss.backward()
    part=objective(new,seq,'A','F00',{'lambda_sep':0.,'lambda_close':0.,'margin':2.5});part['total'].backward()
    for (n,p),(nn,pp) in zip(old.named_parameters(),new.named_parameters()):
        assert n==nn;torch.testing.assert_close(p.grad,pp.grad,atol=2e-6,rtol=2e-6)
    assert training_schedule()==make_training_schedule().context.tolist()


def test_E1_T08_T09_real_states_patches_hashes():
    provenance=ROOT/'analysis/phase1i_context_composition/phase1i_checkpoint_provenance.csv'
    if not provenance.exists():pytest.skip('blocked_missing_source: '+str(provenance))
    records=checkpoint_records(ROOT)
    for condition in ['High_monotonic','Low_monotonic','C0_canonical','C1_contrast_only','C2_closure_only','C3_full']:
        record=next(r for r in records if r['condition']==condition and r['seed']==0)
        if not (ROOT/record['path']).is_file():pytest.skip('blocked_missing_source: '+record['path'])
        m=load_model(ROOT,record);before=m.digest();a,b,target=references(m)
        oa=rollout(m,[1,0,0]);ob=rollout(m,[0,1,0])
        np.testing.assert_array_equal(a['hidden'],oa['hidden']);np.testing.assert_array_equal(b['logits'],ob['logits'])
        np.testing.assert_allclose(target,midpoint(oa['hidden'],ob['hidden']),atol=1e-7,rtol=0)
        native=natural(m,[.5,.5,0]);np.testing.assert_array_equal(native['logits'],rollout(m,[.5,.5,0])['logits'])
        p=single_patch_replay(m,[.5,.5,0])
        for position in range(7):
            old=rollout(m,[.5,.5,0],target=midpoint(oa['hidden'],ob['hidden']),patch='correct',position=position)
            new=p[p.position.eq(position+1)]
            np.testing.assert_allclose(new.raw_ce,old['raw_ce'][position:],atol=.002,rtol=0)
        assert m.digest()==before
