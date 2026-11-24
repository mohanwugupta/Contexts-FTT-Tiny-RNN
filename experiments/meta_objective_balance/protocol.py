"""Prespecified design and context-dependent query-prefix identification."""
import copy
import numpy as np
from experiments.heldout_context_meta import protocol as old
ARMS={'A':(.25,0.),'B':(.25,.5),'C':(.5,0.),'D':(.5,.5),'E':(.75,0.),'F':(.75,.5)}
SCREEN=list(range(6));CONFIRM=list(range(6,20));RECOVERY_EVERY=60

def critical_mask(ep,qs):
 m=np.zeros((len(qs),7),dtype=bool);groups={}
 for i,q in enumerate(qs):
  path=ep['paths'][q['sequence']]
  for t in range(7):groups.setdefault(tuple(path[:t+1]),[]).append((i,t,path[t+1],q['context']))
 for g in groups.values():
  if len({r[2] for r in g})>1 and len({r[3] for r in g})>1:
   for i,t,_,_ in g:m[i,t]=True
 return m

def pair_block(block):
 old.validate(block);s=[e for e in block['episodes'] if e['kind']=='shared'];c=[e for e in block['episodes'] if e['kind']=='conflict']
 return [dict(pair=(block['block']-5)*120+i+1,source_block=block['block'],shared=copy.deepcopy(a),conflict=copy.deepcopy(b)) for i,(a,b) in enumerate(zip(s,c))]

def validate_pairs(pairs):
 try:
  assert len(pairs)%120==0
  for off in range(0,len(pairs),120):
   chunk=pairs[off:off+120];eps=[p[k] for p in chunk for k in ['shared','conflict']];old.validate(dict(episodes=eps))
   for p in chunk:
    assert p['shared']['kind']=='shared' and p['conflict']['kind']=='conflict'
    e=p['conflict'];qs=old.queries(e,'P2');m=critical_mask(e,qs);assert m.sum()>0 and m.shape==(4,7)
    s=old.support(e,'P2','Transfer');mapping={}
    for x in s:mapping.setdefault(x['context'],set()).add(x['sequence'])
    assert len(mapping)==2 and all(len(v)==1 for v in mapping.values())
  return True
 except (AssertionError,KeyError,TypeError) as e:raise ValueError('Invalid paired stream/context schedule') from e
