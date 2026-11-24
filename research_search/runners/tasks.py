"""Trusted bank construction excludes every development/confirmation orbit."""
import copy,numpy as np
from experiments.context_experience_prior import registry
from research_search.evaluation.validity import background_allowed

def background(seed,episodes,forbidden,kind='P2'):
    rows=[];rng=np.random.default_rng(np.random.SeedSequence([202611230802,int(seed)]));attempts=0
    while len(rows)<episodes:
        r=registry.make_registry(int(rng.integers(0,2**31-1)))
        for ep,update in zip(r['episodes'],r['updates']):
            ep=copy.deepcopy(ep)
            while not background_allowed(ep['paths'][0],forbidden):
                attempts+=1
                if attempts>100000:raise RuntimeError('Fixed task-generation limit reached; retain unavailable bank')
                ep['paths'][0]=registry.draw(rng)
            if ep['kind']=='conflict':
                while not background_allowed(ep['paths'][1],forbidden) or ep['paths'][1][:3]!=ep['paths'][0][:3] or ep['paths'][1][3]==ep['paths'][0][3]:
                    attempts+=1
                    if attempts>100000:raise RuntimeError('Fixed task-generation limit reached; retain unavailable bank')
                    ep['paths'][1]=registry.draw(rng,ep['paths'][0][:3],ep['paths'][0][3])
            rows.append(dict(episode=len(rows)+1,paths=ep['paths'],updates=update[kind],kind=ep['kind'],pair=ep['pair']))
            if len(rows)==episodes:break
    return rows
