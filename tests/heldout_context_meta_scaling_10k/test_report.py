from experiments.heldout_context_meta_scaling_10k.report import QUESTIONS,interpret_interval

def test_nine_questions_and_inconclusive_is_not_equivalence():
 assert len(QUESTIONS)==9
 assert interpret_interval(-1,1)=='uncertain direction; neither benefit nor equivalence established'
 assert interpret_interval(-2,-1)=='decrease supported by this pointwise interval'
 assert interpret_interval(1,2)=='increase supported by this pointwise interval'

def test_sample_denominator_uses_actual_interval_schema_without_guessing():
 import pandas as pd
 import pytest
 from experiments.heldout_context_meta_scaling_10k.report import sample_n
 assert sample_n(pd.Series({'n_seeds':20}))==20
 assert sample_n(pd.Series({'n':0}))==0
 with pytest.raises(ValueError):sample_n(pd.Series({'mean':0.}))
 with pytest.raises(ValueError):sample_n(pd.Series({'n':20,'n_seeds':19}))
