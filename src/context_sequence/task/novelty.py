"""Behavior-blind cue registry; this module cannot load models."""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import numpy as np
from context_sequence.provenance.hashing import object_hash

VARIANTS=('Novel30_A','Novel30_B','Novel60_A','Novel60_B')
GROUPS={'Novel30Mean':VARIANTS[:2],'Novel60Mean':VARIANTS[2:],'NovelMean':VARIANTS}


def registry_payload():
    r=math.sqrt(3)/2
    vectors={'A':[1.,0.,0.],'B':[0.,1.,0.],'Mixed':[.5,.5,0.],'NoContext':[0.,0.,0.],
             'Novel30_A':[r,-.5,0.],'Novel30_B':[-.5,r,0.],
             'Novel60_A':[.5,-r,0.],'Novel60_B':[-r,.5,0.]}
    cues={}
    for name,v in vectors.items():
        norm=float(np.linalg.norm(v));angle=math.degrees(math.atan2(v[1],v[0])) if norm else None
        da=math.degrees(math.acos(np.clip(v[0]/norm,-1,1))) if norm else None
        db=math.degrees(math.acos(np.clip(v[1]/norm,-1,1))) if norm else None
        cues[name]={'vector':v,'angle_degrees':angle,'nearest_familiar':('A' if da<=db else 'B') if norm else None,
                    'novelty_degrees':min(da,db) if norm else None,'norm':norm,
                    'in_convex_hull':bool(v[2]==0 and v[0]>=0 and v[1]>=0 and abs(v[0]+v[1]-1)<1e-12),
                    'in_familiar_sector':bool(norm and v[0]>=0 and v[1]>=0)}
    payload={'version':'novelty_registry_v1','created_at':datetime.now(timezone.utc).isoformat(),
             'definition':'Unit-norm out-of-support cues outside the familiar angular sector; not a novel semantic dimension.',
             'repository_state':'prospective registry; historical source snapshot in artifacts/manifests/pre_freeze_snapshot.json',
             'cues':cues,'aggregates':{k:list(v) for k,v in GROUPS.items()},
             'legacy_label_map':{**{f'Novel{i}':f'LegacyNovel{i}' for i in range(1,5)},'NovelMean':'LegacyNovelMean'}}
    payload['registry_sha256']=object_hash(payload)
    return payload


def verify_registry(payload):
    body={k:v for k,v in payload.items() if k!='registry_sha256'}
    if object_hash(body)!=payload['registry_sha256']:raise ValueError('Novel registry hash mismatch')
    expected=registry_payload()
    if payload['cues']!=expected['cues'] or payload['aggregates']!=expected['aggregates']:raise ValueError('Novel registry differs from prospectively specified geometry')
    return payload


def read_registry(path,expected_hash=None):
    payload=verify_registry(json.loads(Path(path).read_text()))
    if expected_hash and payload['registry_sha256']!=expected_hash:raise ValueError('Contract registry hash mismatch')
    return payload


def freeze_registry(path):
    path=Path(path)
    if path.exists():return read_registry(path)
    path.parent.mkdir(parents=True,exist_ok=True);payload=registry_payload()
    with path.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    return payload


def cues_from_registry(payload):
    verify_registry(payload)
    return {k:np.array(v['vector'],dtype=np.float64) for k,v in payload['cues'].items()}
