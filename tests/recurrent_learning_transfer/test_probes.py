import numpy as np
import pandas as pd
import pytest
from experiments.recurrent_learning_transfer.probes import validate_seal,pca_familiar,geometry
from experiments.recurrent_learning_transfer.design import ARCHS,DIAG
from experiments.recurrent_learning_transfer.network import pair,trace


def test_no_protected_evaluation_of_incomplete_or_selected_runs():
 r=pd.DataFrame([dict(seed=s,architecture=a,checkpoint=k,status='completed',checkpoint_path=f'{s}_{a}_{k}',checkpoint_sha='x') for s in range(20) for a in ARCHS for k in DIAG]);seal=dict(all40_attempts_registered=True,all320_checkpoint_slots_registered=True,scientific_updates_planned=13200,diagnostic_updates_planned=640,checkpoint_hashes=dict(zip(r.checkpoint_path,r.checkpoint_sha)))
 assert validate_seal(seal,r)
 for bad in [r.iloc[:-1],pd.concat([r,r.iloc[[0]]])]:
  with pytest.raises(ValueError):validate_seal(seal,bad)
 bad=dict(seal);bad['all40_attempts_registered']=False
 with pytest.raises(ValueError):validate_seal(bad,r)


def test_pca_uses_familiar_only_and_does_not_classify_learning():
 m=pair(1)['RNN-Tanh'];d={c:trace(m,c) for c in ['A','B']};p=pca_familiar(d);assert p['coordinates'].shape==(14,2)
 assert geometry(d)['hidden_centroid_distance']>=0
 with pytest.raises(ValueError):pca_familiar({**d,'Mixed':trace(m,'Mixed')})
 assert 'functional_split' not in geometry(d)
