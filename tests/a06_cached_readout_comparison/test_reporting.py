import pandas as pd
from experiments.a06_cached_readout_comparison import reporting
def test_position_explanation_names_new_errors_and_retains_wrong_positions():
 f=pd.DataFrame([dict(position=2,transition='became_incorrect',ce_contribution=.2),dict(position=3,transition='stayed_incorrect',ce_contribution=-19.8),dict(position=1,transition='stayed_correct',ce_contribution=0.)])
 r=reporting.transitions(f);assert r['became_incorrect']==[2] and r['stayed_incorrect']==[3] and r['largest_loss_decrease_position']==3
def test_missing_endpoint_is_not_zero_or_failure():
 assert reporting.fmt(None)=='unavailable'
 assert reporting.fmt(0.)=='0'
