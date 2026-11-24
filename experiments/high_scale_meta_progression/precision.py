"""Separate pre-existing float32 bound; never relabel the strict replay outcome."""
from .common import *
from experiments.structured_overlap_meta.verify import check_forward

def bounded_forward(state,paths,contexts,z,h):
 strict=True;message=''
 try:check_forward(state,paths,contexts,z,h)
 except AssertionError as exc:strict=False;message=str(exc)
 wi,wh,bi,bh,wo,bo=[v.double().numpy() for v in state.values()];hidden=np.zeros((len(paths),8));bound=np.zeros_like(hidden);hh=[];zz=[];hb=[];zb=[];cc=np.array([CUES[c] for c in contexts]);eps=64*np.finfo(np.float32).eps
 for t in range(7):
  x=np.concatenate([np.eye(4)[[p[t] for p in paths]],cc],1);previous=np.zeros_like(hidden) if t==0 else h[:,t-1];bound=bound@abs(wh).T+eps*(1+abs(x)@abs(wi).T+abs(bi)+abs(bh)+abs(previous)@abs(wh).T);hidden=x@wi.T+hidden@wh.T+bi+bh;pred=hidden@wo.T+bo;out_bound=bound@abs(wo).T+eps*(1+abs(h[:,t])@abs(wo).T+abs(bo));hh.append(hidden.copy());zz.append(pred);hb.append(bound.copy());zb.append(out_bound)
 hh=np.stack(hh,1);zz=np.stack(zz,1);hb=np.stack(hb,1);zb=np.stack(zb,1);tol=1e-10+1e-12*max(abs(hh).max(),abs(zz).max());eh=abs(h-hh);ez=abs(z-zz);assert np.all(eh<=hb+tol) and np.all(ez<=zb+tol),'Outside unchanged predecessor float32 propagation bound'
 return dict(strict_passed=strict,strict_atol=2e-5,strict_rtol=2e-5,strict_failure=message,existing_bound_passed=True,predecessor_bound='experiments/context_experience_prior/measures.py:measure_model',epsilon=float(eps),double_tolerance=float(tol),max_float32_logit_error=float(ez.max()),max_float32_hidden_error=float(eh.max()),max_logit_bound_ratio=float((ez/(zb+tol)).max()),max_hidden_bound_ratio=float((eh/(hb+tol)).max()))
