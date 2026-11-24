import copy
import numpy as np
import torch
import pytest
from experiments.high_scale_initialization.core import pair,step,trace,policy,optimizer,training_cue
from confirmatory_srn.training import state_hash,train_condition
from confirmatory_srn.config import smoke_config
from confirmatory_srn.model import ConfirmatoryElmanSRN

def test_canonical_initial_draw_and_pair_are_exact_without_normalization():
 torch.manual_seed(3);canonical=ConfirmatoryElmanSRN(init_scale=2.)
 arms=pair(3)
 for m in arms.values():
  assert state_hash(m)==state_hash(canonical)
  assert torch.linalg.matrix_norm(m.recurrent.weight_hh_l0,2)>1
  assert not torch.count_nonzero(m.recurrent.weight_ih_l0[:,6])
 assert [training_cue(k) for k in [1,10,11,20,170,330]]==['A','A','B','B','A','A']

def test_source_training_smoke_parity_and_protected_cue_guard():
 torch.set_num_threads(1);r=train_condition('High_monotonic',smoke_config(),seed=2)
 m=pair(2)['RNN-Tanh'];p=policy();o=optimizer(m,p);losses=[]
 for k in range(1,21):losses.append(step(m,p,o,training_cue(k))['ce_before'])
 assert state_hash(m,p)==state_hash(r.model,r.policy)
 np.testing.assert_array_equal(losses,r.sequence_level.sequence_CE)
 with pytest.raises(ValueError):step(m,p,o,'Mixed')

def test_linear_frozen_math_and_current_token_shape():
 m=pair(1)['RNN-Linear'];d={c:trace(m,c,dtype=torch.float64) for c in ['A','B','Mixed']}
 np.testing.assert_allclose(d['Mixed']['hidden'],(d['A']['hidden']+d['B']['hidden'])/2,atol=1e-7,rtol=1e-12)
 assert d['Mixed']['ce'].mean()<=(d['A']['ce'].mean()+d['B']['ce'].mean())/2+1e-7
 assert d['A']['hidden'].shape==(7,8)
