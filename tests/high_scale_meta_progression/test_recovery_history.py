import copy
import numpy as np
import pytest
import torch
from experiments.high_scale_meta_progression.train import verify_saved_checkpoint
from experiments.high_scale_meta_progression.common import atomic_save,dump
from experiments.heldout_context_meta_scaling_10k.storage import Journal

def test_atomic_journal_rejects_skips_and_checkpoint_tampering(tmp_path):
 j=Journal(tmp_path/'j.sqlite');j.initialize(dict(episode=0,x='start'));j.commit(dict(episode=1,x='state'),dict(episode=1,gradient=np.ones(3)),registered=True);assert j.latest()['episode']==1 and j.checkpoint(1)['x']=='state'
 with pytest.raises(ValueError):j.commit(dict(episode=1),dict(episode=1))
 with pytest.raises(ValueError):j.commit(dict(episode=3),dict(episode=3))
 j.close()
 p=tmp_path/'c.pt';h=atomic_save(p,dict(model={'w':torch.ones(2)},optimizer=None,episode=0));dump(p.with_suffix('.json'),dict(sha256=h));verify_saved_checkpoint(p);p.write_bytes(b'wrong')
 with pytest.raises(AssertionError):verify_saved_checkpoint(p)

def test_source_float32_interpolation_arithmetic():
 a=torch.linspace(-23,15,172);b=torch.linspace(9,-17,172)
 for e in [.1,.01,.065732]:np.testing.assert_array_equal((a+e*(b-a)).numpy(),a.numpy()+e*(b.numpy()-a.numpy()))
