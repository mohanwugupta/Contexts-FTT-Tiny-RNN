from experiments.high_scale_constrained_meta.registries import generate
from experiments.heldout_context_meta.protocol import support,queries

def test_native_counterbalance_and_global_diagnostic_exclusion():
 a=generate(0,'Reptile',1,set());reserved={tuple(a['episodes'][0]['paths'][0])};b=generate(0,'Reptile',1,reserved)
 assert len(b['episodes'])==240 and not any(tuple(p) in reserved for e in b['episodes'] for p in e['paths'])
 f=generate(0,'FOMAML',1,reserved)
 for e in f['episodes']:
  ss=support(e,'P2','Transfer');assert len(ss)==20
  if e['kind']=='shared':assert e['query_context'] not in {r['context'] for r in ss} and len(queries(e,'P2'))==1
  else:assert len(queries(e,'P2'))==4 and e['paths'][0][:3]==e['paths'][1][:3] and e['paths'][0][3]!=e['paths'][1][3]
