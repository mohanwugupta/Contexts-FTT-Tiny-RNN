"""Exactly matched familiar exposure, different within/between-sequence timing."""
import numpy as np


def schedule(arm):
    if arm not in ['B','I','S']:raise ValueError('Unregistered training arm')
    c=np.zeros((160,7,3),np.float32)
    for i in range(160):
        first=0 if ((i//10)%2==0 if arm=='B' else i%2==0) else 1
        c[i,:,first]=1
        if arm=='S':c[i,3:,first]=0;c[i,3:,1-first]=1
    validate_schedule(c,arm);return c


def validate_schedule(c,arm):
    c=np.asarray(c)
    if c.shape!=(160,7,3):raise ValueError('Exactly160 online sequences with seven cues required')
    isA=np.all(c==[1,0,0],axis=2);isB=np.all(c==[0,1,0],axis=2)
    if not (isA|isB).all():raise ValueError('protected/non-familiar cue detected in training')
    if not np.all(isA.sum(0)==80) or not np.all(isB.sum(0)==80):raise ValueError('Position-level exposure imbalance')
    switches=np.any(c[:,1:]!=c[:,:-1],axis=2)
    if arm=='S':
        expected=np.zeros((160,6),bool);expected[:,2]=True
        if not np.array_equal(switches,expected) or not np.array_equal(isA[:,0],np.arange(160)%2==0):raise ValueError('S must alternate A→B/B→A with one3→4 switch')
    else:
        expected=(np.arange(160)//10)%2==0 if arm=='B' else np.arange(160)%2==0
        if arm not in ['B','I'] or switches.any() or not np.array_equal(isA[:,0],expected):raise ValueError('Wrong blocked/interleaved order')
    return True
