import copy,json
import numpy as np
import torch
from experiments.high_scale_meta_progression.runtime import meta_step,fresh_pack
from experiments.heldout_context_meta_scaling_10k.adapters import bind
from experiments.structured_overlap_meta import runtime as overlap
from experiments.high_scale_meta_progression.protocol import scaled_initial
from experiments.high_scale_meta_progression.common import HIGH,core,source_model,initialize_process
from experiments.context_experience_prior.meta import run_episode
from experiments.heldout_context_meta import fomaml
from experiments.meta_objective_balance.runtime import adapt_query,combine
from experiments.structured_overlap_meta.generator import family,episode
from experiments.structured_overlap_meta.runtime import meta_episode

def check(a,b):
 for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

def test_native_reptile_transfer_balanced_structured_gradients():
 initialize_process();base=scaled_initial(0,5);rr=json.loads((HIGH/'registry/reptile_s00.json').read_text())['episodes'][0];eps=json.loads((HIGH/'registry/fomaml_s00.json').read_text())['episodes'];s=next(e for e in eps if e['kind']=='shared');c=next(e for e in eps if e['kind']=='conflict')
 a=source_model(base);b=source_model(base);r=meta_step(a,None,rr,'R',1);run_episode(b,rr,rr['support_P2'],.1);check(a.state_dict(),b.state_dict());assert r['inner_updates']==20
 a=source_model(base);b=source_model(base);ao=fomaml.outer_optimizer(a);bo=fomaml.outer_optimizer(b);r=meta_step(a,ao,s,'T',1);q=fomaml.episode(b,bo,s,'P2','Transfer');check(a.state_dict(),b.state_dict());np.testing.assert_array_equal(r['gradient'],q['gradient'])
 a=source_model(base);b=source_model(base);ao=fomaml.outer_optimizer(a);bo=fomaml.outer_optimizer(b);pair=dict(pair=1,shared=s,conflict=c);r=meta_step(a,ao,pair,'B',1);gs,_,_=adapt_query(b,s);ga,gc,_=adapt_query(b,c);g=combine(gs,ga,gc,.5,0);bo.zero_grad(set_to_none=True)
 for p,v in zip(b.parameters(),g):p.grad=v.clone()
 torch.nn.utils.clip_grad_norm_(list(b.parameters()),1.);bo.step();check(a.state_dict(),b.state_dict());np.testing.assert_array_equal(r['gradient'],np.concatenate([v.numpy().ravel() for v in g]));assert r['inner_updates']==40 and r['episodes']==2
 e=episode(family(np.random.default_rng(99311),dict(X='A',Y='B',Q='C'),(0,1),0,5,set()),2)
 a=source_model(base);b=source_model(base);ao=fomaml.outer_optimizer(a);bo=fomaml.outer_optimizer(b);r=meta_step(a,ao,e,'O1',1)
 with bind(overlap,pack_state=fresh_pack):q=meta_episode(b,bo,e)
 check(a.state_dict(),b.state_dict());np.testing.assert_array_equal(r['gradient'],q['combined_gradient']);assert r['sequence_presentations']==40
