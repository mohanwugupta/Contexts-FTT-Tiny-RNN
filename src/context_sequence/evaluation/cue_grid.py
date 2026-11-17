"""Behavior-blind finite cue grid, norm/direction controls and exact anchors."""
import numpy as np
from context_sequence.provenance.hashing import object_hash

ALPHA=[-1.366025403784,-1.,-.5,0.,.5,1.,1.366025403784]
BETA=[-1.,-.366025403784,0.,.366025403784,1.]


def alpha_beta(cue):
    x,y,z=np.asarray(cue,dtype=float)
    if z!=0:raise ValueError('Third cue channel must be zero')
    return float(x-y),float(x+y)


def vector_from_coordinates(alpha,beta):
    return np.array([(alpha+beta)/2,(beta-alpha)/2,0.])


def cue_geometry(cue):
    c=np.asarray(cue,dtype=float);a,b=alpha_beta(c);norm=np.linalg.norm(c)
    angle=None if norm==0 else float(np.degrees(np.arccos(np.clip(max(c[0],c[1])/norm,-1,1))))
    # Distance to the familiar convex segment x+y=1, x,y>=0.
    t=np.clip((c[0]-c[1]+1)/2,0,1);endpoint=np.array([t,1-t,0])
    return {'cue_alpha':a,'cue_beta':b,'cue_norm':float(norm),'nearest_endpoint_angle_degrees':angle,
            'distance_to_familiar_segment':float(np.linalg.norm(c-endpoint))}


def grid_registry(novelty):
    cues=[]
    def add(name,kind,v):
        v=np.asarray(v,dtype=float).tolist()
        cues.append({'name':name,'kind':kind,'vector':v,'cue_sha256':object_hash(v),**cue_geometry(v)})
    for i,a in enumerate(ALPHA):
        for j,b in enumerate(BETA):
            add(f'grid_a{i}_b{j}','grid',vector_from_coordinates(a,b))
            # Preserve the declared decimal axes rather than floating-point cancellation in x±y.
            cues[-1]['cue_alpha']=a;cues[-1]['cue_beta']=b
    for name,value in novelty['cues'].items():add(name,'anchor',value['vector'])
    for side,v in [('A',np.array([1,0,0])),('B',np.array([0,1,0]))]:
        for scale in [.5,1.,1.5]:add(f'ray_{side}_{scale:g}','ray',scale*v)
    r={'version':'context_grid_v1','alpha':ALPHA,'beta':BETA,'third_channel':0.,'novelty_registry_sha256':novelty['registry_sha256'],'cues':cues}
    r['registry_sha256']=object_hash(r)
    return r


def verify_grid(registry):
    if object_hash({k:v for k,v in registry.items() if k!='registry_sha256'})!=registry['registry_sha256']:raise ValueError('Cue grid registry hash mismatch')
    if registry['alpha']!=ALPHA or registry['beta']!=BETA:raise ValueError('Registered cue axes changed')
    for c in registry['cues']:
        if object_hash(c['vector'])!=c['cue_sha256'] or c['vector'][2]!=0:raise ValueError('Cue hash/third channel mismatch: '+c['name'])
    return registry
