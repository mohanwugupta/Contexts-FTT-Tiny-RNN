import numpy as np

def validate_exposure(rows,path,assignment=('A','B'),expected=330):
    if len(rows)!=expected:raise ValueError('Missing/extra target practice')
    for k,r in enumerate(rows,1):
        wanted='A' if (k-1)//10%2==0 else 'B'
        if r['update']!=k or r['context']!=wanted or r['path']!=list(path):raise ValueError('Protected target, sequence or schedule mismatch')
        if 'physical_context' in r and r['physical_context']!=assignment[0 if wanted=='A' else 1]:raise ValueError('Incorrect context assignment')
    return True

def midpoint(a,b,m,tolerance=1e-10):
    residual=np.asarray(m)-(np.asarray(a)+np.asarray(b))/2
    assert np.isfinite(residual).all() and np.max(np.abs(residual))<=tolerance
    return dict(passed=True,max_error=float(np.abs(residual).max()),tolerance=float(tolerance))

def background_allowed(path,forbidden):
    from experiments.context_experience_prior.registry import allowed,orbit
    return allowed(path) and not any(equivalent(path,p) for p in forbidden)

def equivalent(a,b):
    from experiments.context_experience_prior.registry import orbit
    return bool(orbit(a)&orbit(b) or (orbit(a[:-1])|orbit(a[1:]))&(orbit(b[:-1])|orbit(b[1:])))
