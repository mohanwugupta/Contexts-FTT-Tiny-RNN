"""Prebehavioral sequence registry and exact relabeling/rotation classes."""
import itertools
from collections import Counter
import pandas as pd
from context_sequence.provenance.hashing import object_hash


def equivalence_class(sequence):
    s=tuple(sequence);classes=[]
    for k in range(len(s)):
        rotation=s[k:]+s[:k];mapping={};canonical=[]
        for v in rotation:
            if v not in mapping:mapping[v]=len(mapping)
            canonical.append(mapping[v])
        classes.append(tuple(canonical))
    return min(classes)


def sequence_registry():
    proposed=[('original',(0,1,2,0,3,1,2,3)),('S1',(0,1,0,1,2,3,2,3)),('S2',(0,1,0,2,1,3,2,3)),('S3',(0,1,2,0,3,2,1,3))]
    used=set();rows=[]
    for name,s in proposed:
        original=s
        if equivalence_class(s) in used:
            for candidate in itertools.product(range(4),repeat=8):
                if Counter(candidate)==Counter({i:2 for i in range(4)}) and all(a!=b for a,b in zip(candidate,candidate[1:])) and equivalence_class(candidate) not in used:
                    s=candidate;break
        used.add(equivalence_class(s))
        rows.append({'sequence_id':name,'tokens':list(s),'proposed_tokens':list(original),'replaced_equivalent_candidate':s!=original,'sequence_sha256':object_hash(list(s)),'equivalence_class':list(equivalence_class(s))})
    r={'version':'sequence_registry_v1','sequences':rows,'seeds':list(range(20))};r['registry_sha256']=object_hash(r)
    return r


def validate_sequences(registry):
    if object_hash({k:v for k,v in registry.items() if k!='registry_sha256'})!=registry['registry_sha256']:raise ValueError('Sequence registry hash mismatch')
    seen=set();rows=[]
    for r in registry['sequences']:
        s=tuple(r['tokens']);eq=equivalence_class(s)
        valid=(len(s)==8 and Counter(s)==Counter({i:2 for i in range(4)}) and all(a!=b for a,b in zip(s,s[1:])) and eq not in seen and object_hash(list(s))==r['sequence_sha256'])
        rows.append({**r,'valid':valid});seen.add(eq)
    return pd.DataFrame(rows)
