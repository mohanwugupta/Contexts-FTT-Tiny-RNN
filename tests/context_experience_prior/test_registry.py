import numpy as np
import pandas as pd
import pytest
from experiments.context_experience_prior.registry import make_registry,validate,allowed,transition_counts

def test_target_dihedral_exclusion_and_history():
 target=[0,1,2,0,3,1,2,3]
 for x in [target,target[::-1],target[2:]+target[:2]]:assert not allowed(x)
 assert not allowed([0,1,2,3,0,1,2,3]) # one-step lookup suffices

def test_exact_episode_exposure_and_context_information():
 r=make_registry(0);validate(r);assert len(r['episodes'])==240
 e=pd.DataFrame(r['episodes']);assert e.groupby('pair').size().tolist()==[80]*3
 assert e.groupby(['pair','kind']).size().tolist()==[40]*6
 for ep in r['episodes']:
  rows=r['updates'][ep['episode']-1];p1=pd.DataFrame(rows['P1']);p2=pd.DataFrame(rows['P2'])
  assert p1.context.value_counts().sort_index().equals(p2.context.value_counts().sort_index())
  assert p1.sequence.value_counts().sort_index().equals(p2.sequence.value_counts().sort_index())
  if ep['kind']=='conflict':assert sorted(p1.groupby(['context','sequence']).size())==[5]*4 and sorted(p2.groupby(['context','sequence']).size())==[10]*2
  for path in ep['paths']:assert allowed(path) and transition_counts(path).sum()==7

def test_registry_deterministic_and_leakage_negative():
 r=make_registry(1);assert r==make_registry(1) and r!=make_registry(2)
 r['episodes'][0]['paths'][0]=[0,1,2,0,3,1,2,3]
 with pytest.raises(ValueError):validate(r)
