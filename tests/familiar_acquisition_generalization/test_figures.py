from pathlib import Path
import pandas as pd
from experiments.familiar_acquisition_generalization.figures import validate_inputs
import pytest

def test_figures_do_not_silently_drop_models_or_duplicate_originals():
 f=pd.DataFrame([dict(seed=s,arm=a) for s in range(20) for a in ['Original','B','I','S']]);assert validate_inputs(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]])]:
  with pytest.raises(ValueError):validate_inputs(bad)
