"""Native64eps propagation contract, generalized only to registered task tokens."""
import numpy as np
from experiments.context_experience_prior import core
from experiments.joint_phenotype_hypothesis_screen.banks import deliver
from .validity import midpoint

def controls(state,native_z,native_h,double,task,onset=False):
    labels=['A','B','C','Mixed-average','NoContext'];tol=1e-10+1e-12*max(abs(double[c]['logits']).max() for c in labels);a,b,mix=[double[c] for c in ['A','B','Mixed-average']];r=midpoint(a['logits'],b['logits'],mix['logits'],tol);assert np.all(mix['ce']-(a['ce']+b['ce'])/2<=2*tol)
    d={k:v.detach().numpy().astype(float) for k,v in state.items()};wi=d['recurrent.weight_ih_l0'];R=d['recurrent.weight_hh_l0'];wo=d['sequence_head.weight'];bi=d['recurrent.bias_ih_l0'];bh=d['recurrent.bias_hh_l0'];bo=d['sequence_head.bias'];eps=64*np.finfo(np.float32).eps;bounds=[];assignment=task.get('context_assignment',['A','B'])
    for j,c in enumerate(labels):
        physical=assignment[0] if c=='A' else assignment[1] if c=='B' else c;cues=deliver(core.CUES[physical],7,onset);v=np.zeros(len(R));rows=[]
        for t,token in enumerate(task['path'][:-1]):
            previous=np.zeros(len(R)) if t==0 else native_h[j,t-1];v=abs(R)@v+eps*(1+abs(wi[:,token])+abs(wi[:,4:])@abs(cues[t])+abs(bi)+abs(bh)+abs(R)@abs(previous));zbound=abs(wo)@v+eps*(1+abs(wo)@abs(native_h[j,t])+abs(bo));assert np.all(abs(native_z[j,t]-double[c]['logits'][t])<=zbound+tol);rows.append(zbound)
        bounds.append(np.asarray(rows))
    bound=bounds[3]+(bounds[0]+bounds[1])/2+tol;res=native_z[3]-(native_z[0]+native_z[1])/2;assert np.all(abs(res)<=bound)
    targets=np.array(task['path'][1:]);scores=[core.score(z,targets)['ce'] for z in native_z];assert np.all(scores[3]-(scores[0]+scores[1])/2<=2*bound.max(1))
    return dict(**r,native_max_error=float(abs(res).max()),native_max_bound=float(bound.max()),float32_bound_factor=64,all4_logits_checked=True)
