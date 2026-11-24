"""Sealed balanced support/query roles; identical task libraries for both arms."""
import numpy as np
from experiments.context_experience_prior.registry import draw,allowed

def signature(ep):return (ep['kind'],tuple(sorted(tuple(p) for p in ep['paths'])))
def schedule(pair,order,mapping,pattern,kind,prior):
 x,y=pair[::1 if order==0 else -1];ctx=[x,y,x,y];seq=([mapping,1-mapping,mapping,1-mapping] if prior=='P2' else ([mapping,mapping,1-mapping,1-mapping] if pattern==0 else [mapping,1-mapping,1-mapping,mapping]))
 if kind=='shared':seq=[0]*4
 return [dict(inner_update=k+1,block=k//5+1,context=ctx[k//5],sequence=seq[k//5]) for k in range(20)]

def queries(ep,prior):
 if ep['kind']=='shared':return [dict(context=ep['query_context'],sequence=0)]
 s=ep['support']['Transfer'][prior]
 if prior=='P1':return [dict(context=c,sequence=j) for c in ep['pair'] for j in range(2)]
 cells=list(dict.fromkeys((r['context'],r['sequence']) for r in s));return [dict(context=c,sequence=j) for c,j in cells for _ in range(2)]

def make_block(seed,block,seen):
 seen=set(seen);rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261121,81841,seed,block])));spec=[]
 for qctx in 'ABC':
  for q in range(40):spec.append(dict(kind='shared',query_context=qctx,order=q%2,companion=(q//2)%2,seen_qfirst=(q//4)%2,mapping=0,pattern=0))
 for pair in ['AB','AC','BC']:
  for q in range(40):spec.append(dict(kind='conflict',pair=pair,order=q%2,mapping=(q//2)%2,pattern=(q//4)%2))
 rng.shuffle(spec);episodes=[];rejected=[]
 for i,item in enumerate(spec,1):
  while True:
   p=draw(rng);paths=[p] if item['kind']=='shared' else [p,draw(rng,p[:3],p[3])];ep=dict(episode=(block-1)*240+i,**item,paths=paths);sig=signature(ep)
   if sig not in seen:seen.add(sig);break
   rejected.append(dict(**ep,reason='duplicate_library_within_seed'))
  if item['kind']=='shared':
   others=[x for x in 'ABC' if x!=item['query_context']];ep['others']=others;tp=''.join(others);sp=item['query_context']+others[item['companion']];so=0 if item['seen_qfirst'] else 1
   ep['support']={'Transfer':schedule(tp,item['order'],0,0,'shared','P1'),'Seen':schedule(sp,so,0,0,'shared','P1')}
  else:
   s={p:schedule(item['pair'],item['order'],item['mapping'],item['pattern'],'conflict',p) for p in ['P1','P2']};ep['support']={'Transfer':s,'Seen':s}
  episodes.append(ep)
 r=dict(seed=seed,block=block,episodes=episodes,rejections=rejected);validate(r);return r

def support(ep,prior,arm):return ep['support'][arm] if ep['kind']=='shared' else ep['support'][arm][prior]

def validate(r):
 if len(r['episodes'])!=240:raise ValueError('240 episodes/block')
 counts={};qcounts={};exposure={a:{c:0 for c in 'ABC'} for a in ['Transfer','Seen']};seen_sig=set()
 for ep in r['episodes']:
  if not all(allowed(p) for p in ep['paths']):raise ValueError('Invalid/leaked sequence')
  sig=signature(ep)
  if sig in seen_sig:raise ValueError('Duplicate accepted task')
  seen_sig.add(sig);counts[ep['kind']]=counts.get(ep['kind'],0)+1
  if ep['kind']=='shared':
   q=ep['query_context'];qcounts[q]=qcounts.get(q,0)+1
   if q in {x['context'] for x in support(ep,'P1','Transfer')}:raise ValueError('Withheld query leaked')
   if q not in {x['context'] for x in support(ep,'P1','Seen')}:raise ValueError('Control query not seen')
  else:
   if ep['paths'][0][:3]!=ep['paths'][1][:3] or ep['paths'][0][3]==ep['paths'][1][3]:raise ValueError('No conflict')
   if ep['support']['Seen']!=ep['support']['Transfer']:raise ValueError('Arms differ on conflict')
   for p in ['P1','P2']:
    ss=support(ep,p,'Transfer');cells={}
    for x in ss:cells[x['context'],x['sequence']]=cells.get((x['context'],x['sequence']),0)+1
    if sorted(cells.values())!=([5]*4 if p=='P1' else [10]*2):raise ValueError('Wrong context information')
  for a in ['Transfer','Seen']:
   s=support(ep,'P1',a)
   if len(s)!=20 or len(set(x['context'] for x in s))!=2:raise ValueError('Support budget/context count')
   for k,x in enumerate(s):
    if x['inner_update']!=k+1 or x['block']!=k//5+1 or x['context']!=s[(k//5)%2*5]['context']:raise ValueError('Wrong support blocks')
    exposure[a][x['context']]+=1
 if counts!={'shared':120,'conflict':120} and counts!={'conflict':120,'shared':120}:raise ValueError('Type balance')
 if qcounts!={'A':40,'B':40,'C':40}:raise ValueError('Query role balance')
 if any(v!=1600 for a in exposure.values() for v in a.values()):raise ValueError('Marginal context exposures differ')
 return True
