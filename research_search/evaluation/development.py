"""Deterministic benchmark registration, not a scientific experiment."""
import numpy as np
from experiments.context_experience_prior.registry import allowed,orbit
from research_search.common import write

CANONICAL=[0,1,2,0,3,1,2,3]
def make_banks(private_seed=0):
    # The default is a deterministic unit-fixture bank, never used for live
    # confirmation. Registration supplies and privately stores an OS seed.
    rng=np.random.default_rng(202611230801);paths=[CANONICAL];tasks=[]
    while len(tasks)<4:
        p=rng.integers(0,4,8).tolist()
        from .validity import equivalent
        if not allowed(p) or any(equivalent(p,q) for q in paths):continue
        # Preserve history dependence and allfour symbols, rather than an
        # unrelated one-step lookup problem.
        paths.append(p);tasks.append(dict(task_id=f'task{len(tasks):02}',path=p,context_assignment=['A','B'] if len(tasks)%2==0 else ['B','A']))
    dev=dict(bank_id='development-v1',tasks=tasks,seeds=list(range(21000,21006)),task_seed=202611230801,target_updates=330)
    rng=np.random.default_rng(private_seed);held=[]
    while len(held)<3:
        p=rng.integers(0,4,8).tolist()
        if not allowed(p) or any(equivalent(p,q) for q in paths):continue
        paths.append(p);held.append(dict(task_id=f'heldout{len(held):02}',path=p,context_assignment=['A','B'] if len(held)%2==0 else ['B','A']))
    seeds=[]
    while len(seeds)<6:
        seed=int(rng.integers(1000000,2**31-1))
        if seed not in seeds:seeds.append(seed)
    confirm=dict(bank_id='confirmation-v1',tasks=held+[dict(task_id='canonical',path=CANONICAL,context_assignment=['A','B'])],seeds=seeds,private_generation_seed=private_seed,target_updates=330,one_evaluation_per_frozen_family=True)
    return dev,confirm
def register(directory):
    from pathlib import Path
    import secrets
    from research_search.common import PACKAGE,read,sha
    p=Path(directory);commit=PACKAGE/'audit/registered_bank_commitments.json'
    if commit.exists():
        r=read(commit)
        for name,h in r['files'].items():
            if sha(p/name)!=h:raise ValueError('Registered bank changed; never resample confirmation')
        return read(p/'development.json'),read(p/'sealed/confirmation.json')
    a,b=make_banks(secrets.randbits(128));write(p/'development.json',a,True);write(p/'sealed/confirmation.json',b,True)
    write(commit,dict(files={n:sha(p/n) for n in ['development.json','sealed/confirmation.json']},development_tasks=4,development_seed_count=6,confirmation_tasks=4,confirmation_seed_count=6,confirmation_random_seed_private=True,canonical_reserved=True),True);return a,b
