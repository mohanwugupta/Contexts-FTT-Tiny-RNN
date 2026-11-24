from experiments.heldout_context_meta_scaling_10k.figures import MAIN,DETAIL,DOSES,plotting_cases

def test_fixed_panels_no_selected_dose_and_historical_alias_is_reference():
 assert MAIN==[f'figure_{i}' for i in range(1,7)]
 assert DETAIL==[960,3840,7680,10000]
 assert DOSES==[0,240,960,1920,3840,5760,7680,10000]
 assert plotting_cases(0)==[('Historical',0)]
 assert plotting_cases(10000)==[('Seen',10000),('Transfer',10000)]
