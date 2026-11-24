"""Pure episode-record, schedule, objective and temporal-gradient checks."""
import numpy as np

def validate_record(r,e):
 try:
  assert r['inner_updates']==20 and r['outer_updates']==1 and r['early_at_update']==10 and r['final_at_update']==20 and r['meta_unchanged_until_outer'] and r['query_preserved_inner'];assert r['query_examples']==8 and r['query_raw_prediction_cells']==56;np.testing.assert_array_equal(r['early_inner_steps'],np.full(8,10));np.testing.assert_array_equal(r['final_inner_steps'],np.full(8,20));assert len(r['updates'])==20
  for k,u in enumerate(r['updates'],1):
   role='X' if k<=10 else 'Y';assert u['inner_update']==k and u['role']==role and u['cue']==e['roles'][role] and u['status']=='completed' and u['optimizer_step_executed'] and u['examples']==2 and u['positions']==14;np.testing.assert_allclose(u['total_loss'],u['ce_before']+.2*u['execution_loss'],rtol=1e-6,atol=2e-6)
  g=.2*r['early_gradient']+.5*r['familiar_gradient']+.3*r['novel_gradient'];np.testing.assert_allclose(r['combined_gradient'],g,rtol=3e-6,atol=2e-6);np.testing.assert_allclose(r['combined_norm'],np.linalg.norm(g),rtol=3e-6,atol=2e-6);assert np.isfinite(g).all();assert r['eligible_cells']==e['eligible_count'] and r['critical_cells']==e['critical_cells']
  if e['n_exceptions']==0:assert np.isnan(r['critical_ce']) and r['familiar_ce']==r['all_ce']
  else:np.testing.assert_allclose(r['familiar_ce'],.75*r['all_ce']+.25*r['critical_ce'],atol=2e-6,rtol=1e-6)
 except (AssertionError,KeyError,TypeError) as exc:raise ValueError('Invalid structured episode record/gradient/checkpoint') from exc
 return True
