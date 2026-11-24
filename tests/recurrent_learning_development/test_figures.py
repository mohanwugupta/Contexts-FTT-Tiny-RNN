import pandas as pd
import pytest

def test_figures_require_all_20_seeds_and_all_blocks():
 from experiments.recurrent_learning_development.figures import validate
 b=pd.DataFrame([dict(seed=s,architecture=a,end=e) for s in range(20) for a in ['RNN-Linear','RNN-Tanh'] for e in range(10,331,10)])
 assert validate(b)
 with pytest.raises(ValueError):validate(b.iloc[:-1])
 with pytest.raises(ValueError):validate(pd.concat([b,b.iloc[:1]]))
