import numpy as np
import torch
from context_sequence.evaluation.rollout import Model
from experiments.context_cue_access.core import source_trace
from experiments.context_cue_access.control import local_maps,control_matrices,finite_difference,rank_alignment,future_gradient


def fixture():
    r=np.random.default_rng(31);wx=r.normal(0,.08,(8,7)).astype('float32');wx[:,6]=0
    return Model(wx,r.normal(0,.08,(8,8)).astype('float32'),np.zeros(8,'float32'),r.normal(0,.2,(4,8)).astype('float32'),np.zeros(4,'float32'))


def test_T27_T28_local_derivatives_autodiff_and_source_fd():
    m=fixture();pre=np.arange(8,dtype='float32')/50;cue=np.array([.3,.4,0.]);h,_=m.step(2,cue,pre);B,J=local_maps(m,h)
    wh=torch.tensor(m.whh.astype(float));wc=torch.tensor(m.wih[:,4:6].astype(float));wx=torch.tensor(m.wih[:,:4].astype(float));bias=torch.tensor(m.bias.astype(float))
    p=torch.tensor(pre.astype(float),requires_grad=True);c=torch.tensor(cue[:2],dtype=torch.float64,requires_grad=True)
    fn=lambda pp,cc:torch.tanh(wh@pp+wc@cc+wx[:,2]+bias)
    bt,jt=torch.autograd.functional.jacobian(lambda cc:fn(p,cc),c),torch.autograd.functional.jacobian(lambda pp:fn(pp,c),p)
    np.testing.assert_allclose(B,bt,atol=1e-7);np.testing.assert_allclose(J,jt,atol=1e-7)
    for eps in [1e-3,1e-4]:
        for i in range(2):
            plus=cue.copy();minus=cue.copy();plus[i]+=eps;minus[i]-=eps
            fd=(m.step(2,plus,pre)[0].astype(float)-m.step(2,minus,pre)[0].astype(float))/(2*eps)
            np.testing.assert_allclose(B[:,i],fd,atol=5e-4,rtol=0)
        for i in range(8):
            plus=pre.astype(float);minus=pre.astype(float);plus[i]+=eps;minus[i]-=eps
            fd=(m.step(2,cue,plus)[0].astype(float)-m.step(2,cue,minus)[0].astype(float))/(2*eps)
            np.testing.assert_allclose(J[:,i],fd,atol=5e-4,rtol=0)


def test_T29_T30_chain_products_and_tied_sum():
    m=fixture();schedule=np.tile([.3,.4,0.],(7,1));t=source_trace(m,schedule)
    for k in [3,5]:
        R=control_matrices(m,t,k)
        assert R['late'].shape==(8,2) and R['two'].shape==(8,4) and R['tied'].shape==(8,2)
        np.testing.assert_array_equal(R['tied'],R['early']+R['late'])
        for eps in [1e-3,1e-4]:
            for name in ['early','late','tied']:
                for i in range(2):
                    fd,method=finite_difference(m,schedule,k,name,i,eps)
                    assert method=='central';np.testing.assert_allclose(fd,R[name][:,i],atol=1e-3,rtol=0)
    boundary=schedule.copy();boundary[:2,0]=1.5
    fd,method=finite_difference(m,boundary,3,'tied',0,1e-3)
    assert method=='backward_inside_box' and np.isfinite(fd).all()


def test_T31_T32_rank_deficiency_zero_gradient_and_alignment():
    v=np.arange(1,9,dtype=float);A=np.tile(v[:,None],(1,4));g=v.copy()
    q=rank_alignment(A,g);assert q['rank_absolute']==1 and q['rank_relative']==1 and q['rank_combined']==1
    assert abs(q['gradient_alignment']-1)<1e-12
    q=rank_alignment(A,np.zeros(8));assert np.isnan(q['gradient_alignment']) and q['alignment_status']=='undefined_zero_gradient'
    q=rank_alignment(np.zeros((8,2)),g);assert q['rank_combined']==0 and q['gradient_alignment']==0
    m=fixture();m.output[:]=0
    grad=future_gradient(m,np.zeros(8),[.3,.4,0.],3)
    np.testing.assert_array_equal(grad,0)
