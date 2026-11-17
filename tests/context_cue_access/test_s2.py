import numpy as np
import torch
from experiments.context_cue_access.history import bindings_for,candidate_values,oracle_starts,parameter_count
from experiments.context_cue_access.simulation import schedules,evaluate
from experiments.context_cue_access.search import CueObjective,optimize
from context_sequence.dynamics.closure import references
from experiments.context_cue_access.audit import input_registry,sources


def fixture():
    from context_sequence.evaluation.rollout import Model
    r=np.random.default_rng(7);wx=r.normal(0,.1,(8,7)).astype('float32');wx[:,6]=0
    return Model(wx,r.normal(0,.2,(8,8)).astype('float32'),np.zeros(8,'float32'),r.normal(0,.3,(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_T18_T19_T20_T21_T22_T25_schedule_and_native_controls():
    m=fixture();a,b,_=references(m);target=np.array([.4,.7,0.]);template=np.tile(target,(7,1));before=m.digest()
    for k in [3,5]:
        initial=[]
        for arm in ['N0','E','L','R','I']:
            d=parameter_count(arm);assert d==({'N0':0,'E':2,'L':2,'R':2,'I':4}[arm])
            values=np.array([[.2,.8,.9,-.3]]) if arm=='I' else np.array([[.2,.8]])
            c=schedules(template,bindings_for(arm,k),values)
            np.testing.assert_array_equal(c[0,k-1:],template[k-1:])
            np.testing.assert_array_equal(c[0,:k-3],template[:k-3])
            if arm=='R':np.testing.assert_array_equal(c[0,k-3],c[0,k-2])
            if arm=='I':assert not np.array_equal(c[0,k-3],c[0,k-2])
            summary,t=evaluate(m,template,bindings_for(arm,k),values,k,(a,b));initial.append(summary.prep_initial_hash.iloc[0])
            assert summary.n_future.iloc[0]==7-k
            assert t.token.tolist()==[0,1,2,0,3,1,2] and t.target.tolist()==[1,2,0,3,1,2,3]
            if arm=='N0':
                from experiments.context_cue_access.core import source_trace
                np.testing.assert_array_equal(t[[f'hidden_{i}' for i in range(8)]],source_trace(m,template)[[f'hidden_{i}' for i in range(8)]])
        assert len(set(initial))==1
    assert before==m.digest()


def test_T23_independent_candidates_include_all_nested_menus():
    reg=input_registry(sources()['cues']);target=np.array([.2,.3,0.]);base=reg[['u','v']].to_numpy()
    vals,ids=candidate_values('I',reg,target)
    assert len(vals)==512+3*177 and len(ids)==len(vals)
    for u in base:
        for pair in [np.r_[target[:2],u],np.r_[u,u],np.r_[u,target[:2]]]:assert np.any(np.all(vals==pair,axis=1))
    second,_=candidate_values('I',reg,target);np.testing.assert_array_equal(vals,second)


def test_T24_T26_equal_budgets_tied_gradient_and_frozen_weights():
    m=fixture();template=np.tile([.5,.5,0.],(7,1));reg=input_registry(sources()['cues']);before=m.digest();counts=[]
    for arm in ['E','L','R','I']:
        candidates,_=candidate_values(arm,reg,template[0]);starts=oracle_starts(arm,sources()['cues'],template[0],candidates[0])
        hist,attempts=optimize(m,template,bindings_for(arm,3),starts,3)
        counts.append((len(hist),attempts.updates_executed.sum(),attempts.objective_evaluations.sum()))
        assert attempts.cue_parameters.eq(parameter_count(arm)).all()
        assert np.abs(attempts[[f'selected_{i}' for i in range(parameter_count(arm))]].to_numpy()).max()<=1.5
    assert len(set(counts))==1 and counts[0]==(486,480,486) and before==m.digest()
    tied=CueObjective(m,template,bindings_for('R',3),3);ind=CueObjective(m,template,bindings_for('I',3),3)
    u=torch.tensor([[.2,.8]],dtype=torch.float64,requires_grad=True);v=torch.cat([u.detach(),u.detach()],1).requires_grad_()
    gt=torch.autograd.grad(tied.loss_cues(u).sum(),u)[0];gi=torch.autograd.grad(ind.loss_cues(v).sum(),v)[0]
    np.testing.assert_allclose(gt.detach(),(gi[:,:2]+gi[:,2:]).detach(),atol=1e-12)
