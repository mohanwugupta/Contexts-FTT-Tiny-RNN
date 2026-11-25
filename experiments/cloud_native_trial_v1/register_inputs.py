"""One-time pure task registration. No models, optimization or evaluation.

Run locally only when creating this version. Cloud consumes the resulting assets;
missing banks are not replaced by resampling or synthetic confirmation identities.
"""
import json
from pathlib import Path
from .control import HERE,ROOT,sha,save

def exclusion_lookup(forbidden):
    from experiments.context_experience_prior.registry import orbit,allowed
    full=set();seven=set()
    for p in forbidden:
        full.update(orbit(p));seven.update(orbit(p[:-1])|orbit(p[1:]))
    def permitted(path,actual):
        if actual is not forbidden:raise ValueError('Changed exclusion list')
        return bool(allowed(path) and not (orbit(path)&full or (orbit(path[:-1])|orbit(path[1:]))&seven))
    return permitted

def cached_background(seed,episodes,forbidden,kind):
    # Exact set-union distributivity; no RNG calls, schedule or native loop changes.
    from research_search.runners import tasks
    original=tasks.background_allowed
    try:
        tasks.background_allowed=exclusion_lookup(forbidden)
        return tasks.background(seed,episodes,forbidden,kind)
    finally:tasks.background_allowed=original

def shared_competence(seed,forbidden):
    import numpy as np
    from experiments.context_experience_prior import registry
    permitted=exclusion_lookup(forbidden)
    rng=np.random.default_rng(np.random.SeedSequence([202611230802,int(seed)+900000]))
    # Only shared competence is needed: no conflicting second path is generated.
    raw=registry.make_registry(int(rng.integers(0,2**31-1)))['episodes']
    attempts=0;cal=[]
    for i,pair in enumerate(['AB','AC','BC']):
        path=raw[i]['paths'][0]
        while not permitted(path,forbidden):
            attempts+=1
            if attempts>100000:raise RuntimeError('Fixed shared competence generation ceiling; retain unavailable bank')
            path=registry.draw(rng)
        cal.append(dict(episode=1,paths=[path],kind='shared',pair=pair,updates=[dict(inner_update=k+1,block=k//5+1,context=pair[(k//5)%2],sequence=0) for k in range(20)]))
    return cal

def register():
    background=cached_background
    from research_search.common import fingerprint
    bank=ROOT/'research_search/database/benchmark_banks'
    commitment=json.loads((ROOT/'research_search/audit/registered_bank_commitments.json').read_text())
    for rel,h in commitment['files'].items():
        if sha(bank/rel)!=h:raise ValueError('Registered bank changed: '+rel)
    dev=json.loads((bank/'development.json').read_text());private=json.loads((bank/'sealed/confirmation.json').read_text())
    seal=json.loads((ROOT/'research_search/audit/evaluator_seal.json').read_text())
    assets=HERE/'assets'
    if assets.exists():raise ValueError('Registered assets must not be overwritten')
    assets.mkdir()
    save(assets/'development.json',dev)
    forbidden=[t['path'] for t in dev['tasks']+private['tasks']]
    for seed in [21000,21001]:
        tasks=background(seed,1920,forbidden,'P2');extra=forbidden+[p for r in tasks for p in r['paths']]
        cal=shared_competence(seed,extra)
        import gzip
        data=json.dumps(dict(seed=seed,background=tasks,competence=cal),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        (assets/f'seed{seed}.json.gz').write_bytes(gzip.compress(data,mtime=0))
    save(assets/'provenance.json',dict(registered_bank_commitments=commitment['files'],runner_hash=seal['runner_hash'],
      generation='Exact native background() with equivalent cached exclusions; fixed shared-only competence registration v1; same paths/order across arms',
      competence_correction='Original unused conflict-path generation exceeded100000 attempts; failure retained, shared-only generation keeps20 updates and inherited criteria',private_confirmation_vectors_included=False,confirmation_evaluations=0,model_constructions=0,optimizer_steps=0))
    save(assets/'FILES.json',{p.name:sha(p) for p in assets.iterdir() if p.is_file()})
    return dict(registered=True,files=4,private_confirmation_data_exported=False)

if __name__=='__main__':print(json.dumps(register()))
