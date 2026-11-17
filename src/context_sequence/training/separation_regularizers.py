"""Familiar-only fixed-hidden-size regularization for the new factorial design."""
import torch


def weak_loss(a,b):
    return (a-b).square().mean()


def strong_loss(a,b,margin):
    distance=torch.linalg.vector_norm(a-b,dim=-1)
    return torch.relu(margin-distance).square().mean()/a.shape[-1]


def closure_loss(model,sequence,a,b):
    target=((a+b)/2).detach()
    # A/B midpoint cue; numerically equal to physical Mixed. No output labels scored here.
    cue=(torch.tensor([1.,0.,0.])+torch.tensor([0.,1.,0.]))/2
    q=model.trajectory(sequence,cue)[1]
    return (q-target).square().mean(),target,q
