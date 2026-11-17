import numpy as np
from context_sequence.evaluation.rollout import centered,midpoint


def geometry(hidden,logits,a,b,k):
    h=np.asarray(hidden,float);z=centered(logits);ha=a['hidden'][k];hb=b['hidden'][k];za=a['centered_logits'][k];zb=b['centered_logits'][k]
    hs=float(np.linalg.norm(ha-hb));zs=float(np.linalg.norm(za-zb));hd=float(np.linalg.norm(h-midpoint(ha,hb)));zd=float(np.linalg.norm(z-midpoint(za,zb)))
    axis=zb-za;coordinate=float((z-za)@axis/(axis@axis+1e-12));off=float(np.linalg.norm(z-(za+coordinate*axis)))
    return {'task_distance':zd,'task_separation':zs,'task_residual':zd/(zs+1e-12),'hidden_distance':hd,'hidden_separation':hs,'hidden_residual':hd/(hs+1e-12),'axis_coordinate':coordinate,'off_axis_distance':off}
