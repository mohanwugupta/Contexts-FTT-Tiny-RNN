import numpy as np
import pytest
from experiments.architecture_averaging_factorial.audit import validate_legacy,validate_cue_registry
from experiments.architecture_averaging_factorial.protocol import CUES
import pandas as pd


def test_cue_registry_not_desired_midpoints():
 assert validate_cue_registry(CUES)
 bad=dict(CUES);bad['Mixed']=[1.,1.,0]
 with pytest.raises(ValueError):validate_cue_registry(bad)
 bad=dict(CUES);bad['Novel30_A']=[.5,.5,0]
 with pytest.raises(ValueError):validate_cue_registry(bad)


def test_existing_d1_roundoff_not_relabelled_mechanism():
 f=pd.DataFrame({'hidden_identity_residual':[1e-14],'affine_residual':[1e-14],'observed_float32_residual':[1e-7],'roundoff_bound':[1e-4]});assert validate_legacy(f)
 f.loc[0,'observed_float32_residual']=1
 with pytest.raises(ValueError):validate_legacy(f)
