"""Full-right-SVD output kernel, admissible rays, and distribution invariance."""
import numpy as np
from .config import CONTRACT


def output_values(model,state,target=None):
    h=np.asarray(state,dtype=float);z=model.output.astype(float)@h+model.output_bias.astype(float)
    centered=z-z.mean();p=np.exp(z-z.max());p/=p.sum()
    ce=None if target is None else float(np.log(np.exp(z-z.max()).sum())+z.max()-z[target])
    return {'logits':z,'centered_logits':centered,'probabilities':p,'ce':ce,'argmax':int(p.argmax())}


def output_spaces(model):
    w=model.output.astype(float);m=w-w.mean(axis=0,keepdims=True)
    _,s,vt=np.linalg.svd(m,full_matrices=True)
    tol=max(m.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    rank=int(np.sum(s>tol));n=vt[rank:].T;r=vt[:rank].T
    if rank>min(w.shape[0]-1,w.shape[1]):raise ValueError('Centered output rank exceeds K−1')
    return {'M':m,'N':n,'R':r,'rank':rank,'nullity':n.shape[1],'singular_values':s,'rank_tolerance':tol,
        'null_residual':float(np.max(np.abs(m@n))) if n.size else 0.,
        'orthogonality_residual':float(np.max(np.abs(n.T@n-np.eye(n.shape[1])))) if n.size else 0.}


def valid_state(state,margin=1e-6):
    h=np.asarray(state,float)
    return bool(np.isfinite(h).all() and np.max(np.abs(h))<=1-margin+CONTRACT['box_tolerance'])


def ray_interval(anchor,direction,margin=1e-6):
    h=np.asarray(anchor,float);d=np.asarray(direction,float);lower=0.;upper=np.inf;bound=1-margin
    for x,v in zip(h,d):
        if abs(v)<1e-15:
            if abs(x)>bound+1e-12:return np.inf,-np.inf
        else:
            a=(-bound-x)/v;b=(bound-x)/v
            lower=max(lower,min(a,b));upper=min(upper,max(a,b))
    return lower,upper


def bounded_step(anchor,direction,target_radius,margin=1e-6):
    h=np.asarray(anchor,float);d=np.asarray(direction,float);norm=np.linalg.norm(d)
    if norm==0:
        return {'state':h.copy(),'realized_radius':0.,'feasible_max_radius':0.,'minimum_radius':0.,'feasible':valid_state(h,margin),'zero_radius':True}
    d=d/norm;lo,hi=ray_interval(h,d,margin)
    radius=min(float(target_radius),.9*max(hi,0)) if np.isfinite(hi) else float(target_radius)
    feasible=bool(hi>=lo and radius>=lo-1e-12 and radius>=0 and valid_state(h+radius*d,margin))
    return {'state':h+radius*d if feasible else h.copy(),'realized_radius':radius if feasible else 0.,'feasible_max_radius':float(max(hi,0)) if np.isfinite(hi) else 0.,'minimum_radius':float(lo) if np.isfinite(lo) else np.nan,'feasible':feasible,'zero_radius':not feasible or radius<=1e-12}


def fixed_radius_step(anchor,direction,radius):
    d=np.asarray(direction,float);norm=np.linalg.norm(d)
    if not norm:return {'state':np.asarray(anchor,float).copy(),'feasible':False}
    h=np.asarray(anchor,float)+d/norm*radius
    return {'state':h,'feasible':valid_state(h)}


def assert_equivalent(model,anchor,state,target=0,check_box=True):
    if check_box and not valid_state(state,margin=0):raise ValueError('Invalid hidden-state box; no future score permitted')
    a=output_values(model,anchor,target);b=output_values(model,state,target)
    residual=float(np.max(np.abs(a['centered_logits']-b['centered_logits'])))
    pdiff=float(np.max(np.abs(a['probabilities']-b['probabilities'])));cdiff=abs(a['ce']-b['ce'])
    argmax_equal=a['argmax']==b['argmax'] or (b['probabilities'][a['argmax']]>=b['probabilities'].max()-1e-10 and a['probabilities'][b['argmax']]>=a['probabilities'].max()-1e-10)
    if residual>CONTRACT['null_tolerance'] or pdiff>CONTRACT['softmax_tolerance'] or cdiff>CONTRACT['ce_equivalence_tolerance'] or not argmax_equal:raise ValueError(f'Current output equivalence failed: centered={residual}, softmax={pdiff}, CE={cdiff}')
    return {'passed':True,'centered_logit_residual':residual,'softmax_max_difference':pdiff,'immediate_ce_difference':cdiff,'argmax_equivalent':bool(argmax_equal)}
