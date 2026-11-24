"""Pure saved-record integrity checks, no model operations."""
import numpy as np
from .protocol import old

def validate_pair_record(r,p,lam,beta):
 try:
  assert r['pair']==p['pair'] and r['inner_updates']==40 and r['outer_updates']==1 and r['query_forwards']==2 and r['same_meta_initialization']
  for k in ['shared','conflict']:
   u=r[k]['updates'];s=old.support(p[k],'P2','Transfer');assert len(u)==20 and [{kk:x[kk] for kk in ['inner_update','block','context','sequence']} for x in u]==s
   assert all(x['status']=='completed' and x['optimizer_step_executed'] for x in u)
   np.testing.assert_allclose([x['total_loss'] for x in u],[x['ce_before']+.2*x['execution_loss'] for x in u],rtol=1e-6,atol=2e-6)
  gs=r['shared']['gradient_all'];ga=r['conflict']['gradient_all'];gc=r['conflict']['gradient_critical'];g=(1-lam)*gs+lam*((1-beta)*ga+beta*gc)
  np.testing.assert_allclose(r['combined_gradient'],g,rtol=3e-6,atol=2e-6);np.testing.assert_allclose(r['combined_norm'],np.linalg.norm(g),rtol=3e-6,atol=2e-6)
  assert r['clipped']==bool(r['combined_norm']>1) and np.isfinite(g).all()
 except (AssertionError,KeyError,TypeError) as e:raise ValueError('Invalid paired-gradient/budget/schedule record') from e
 return True
