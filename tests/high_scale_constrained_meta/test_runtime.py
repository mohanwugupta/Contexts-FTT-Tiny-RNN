import copy,json,numpy as np,torch
from experiments.high_scale_constrained_meta.common import OUT,source_model,core
from experiments.high_scale_constrained_meta.initialization import initialize,project
from experiments.high_scale_constrained_meta.runtime import episode
from experiments.context_experience_prior.meta import run_episode
from experiments.heldout_context_meta import fomaml

def same(a,b):assert all(torch.equal(v,b[k]) for k,v in a.items())
def test_optimization_native_reptile_and_two_step_fomaml_reference():
 eps=json.loads((OUT/'registry/numerical_fixture_episodes.json').read_text());base,_,floors,_=initialize(0);m=copy.deepcopy(base);ref=copy.deepcopy(base);r=run_episode(ref,eps[0],eps[0]['support_P2'],.1);project(ref,floors);q=episode(m,None,eps[0],'Reptile','P',floors,.1);same(m.state_dict(),ref.state_dict());assert q['inner_steps']==20 and q['outer_executed']
 m=copy.deepcopy(base);ref=copy.deepcopy(base);o=fomaml.outer_optimizer(m);ro=fomaml.outer_optimizer(ref)
 for e in eps[1:]:
  rr=fomaml.episode(ref,ro,e,'P2','Transfer');assert rr['status']=='completed';project(ref,floors,ro);qq=episode(m,o,e,'FOMAML','P',floors);same(m.state_dict(),ref.state_dict());np.testing.assert_array_equal(qq['gradient'],rr['gradient']);assert qq['inner_steps']==20 and qq['inner_reset'] and qq['query_unchanged']
  for i,v in o.state_dict()['state'].items():
   for k,x in v.items():assert torch.equal(x,ro.state_dict()['state'][i][k])
 assert all(float(v['step'])==2 for v in o.state.values())
