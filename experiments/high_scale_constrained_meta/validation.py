"""Pure source schedule/objective/step/projection/moment invariants."""
import numpy as np
from experiments.heldout_context_meta.protocol import support
from .initialization import TOL_ABS,TOL_REL

def validate_record(r,e):
 try:
  assert r['inner_steps']==20 and r['outer_executed'] and r['inner_reset'] and r['policy_reset'] and r['query_unchanged'] and r['moments_preserved'];np.testing.assert_array_equal(r['inner_steps_array'],np.full(8,20));schedule=e['support_P2'] if r['phase']=='Reptile' else support(e,'P2','Transfer');assert len(r['updates'])==20
  for u,s in zip(r['updates'],schedule):
   assert all(u[k]==s[k] for k in ['inner_update','block','context','sequence']);assert u['optimizer_step_executed'] and u['status']=='completed';np.testing.assert_allclose(u['total_loss'],u['ce_before']+.2*u['execution_loss'],rtol=1e-6,atol=2e-6)
  assert np.isfinite(r['after']).all() and np.isfinite(r['unprojected']).all();np.testing.assert_array_equal(r['after'][1:],r['unprojected'][1:]);assert len(r['projection'])==5
  if r['arm']=='U':np.testing.assert_array_equal(r['after'],r['unprojected']);assert all(not p['enabled'] for p in r['projection'])
  else:assert all(p['enabled'] and p['after_rms']>=p['floor']-TOL_ABS-TOL_REL*p['floor'] for p in r['projection'])
  if r['phase']=='FOMAML':assert r['query_count']==(1 if e['kind']=='shared' else 4) and np.isfinite(r['gradient']).all() and np.isfinite(r['query_ce'])
 except (AssertionError,KeyError,TypeError) as exc:raise ValueError('Invalid native episode/schedule/projection/Adam record') from exc
 return True
