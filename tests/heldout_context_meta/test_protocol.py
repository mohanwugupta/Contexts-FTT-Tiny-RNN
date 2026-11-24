import copy
import pytest
from experiments.heldout_context_meta.protocol import make_block,validate

def test_shared_roles_marginals_and_leakage_negative():
 r=make_block(0,1,set());validate(r);bad=copy.deepcopy(r)
 e=next(x for x in bad['episodes'] if x['kind']=='shared');e['support']['Transfer'][0]['context']=e['query_context']
 with pytest.raises(ValueError):validate(bad)
 bad=copy.deepcopy(r);e=next(x for x in bad['episodes'] if x['kind']=='shared');e['support']['Seen']=[dict(x,context=e['others'][0]) for x in e['support']['Seen']]
 with pytest.raises(ValueError):validate(bad)
 for e in r['episodes']:
  if e['kind']=='conflict':assert e['support']['Transfer']==e['support']['Seen']
 assert r==make_block(0,1,set())
