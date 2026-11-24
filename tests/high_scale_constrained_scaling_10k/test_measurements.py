import pandas as pd
from experiments.high_scale_constrained_scaling_10k.measurements import measure,all_registry
from experiments.high_scale_constrained_scaling_10k.common import OUT

def test_native_signed_measurements_and_constraint_labels():
 r=pd.read_csv(OUT/'registry/reused_targets.csv').query("phase=='R' and dose==1920").iloc[0].to_dict();t=measure(r,cache=False);assert set(t['outcomes'].condition)=={'U'} and set(t['outcomes'].phase)=={'R'} and set(t['outcomes']['update'])=={0,100,170,330,660,990};assert len(t['switch_costs'])==99 and len(t['position_switch_costs'])==693
 assert set(t['matched_acquisition'].horizon)=={330,990}
 assert all((t['outcomes'].gap-(t['outcomes'].C_ce-t['outcomes'].familiar_ce)).abs()<1e-12)

def test_new_reference_path_survives_missing_reuse_column():
 r=pd.read_csv(OUT/'registry/B_targets.csv').iloc[0].to_dict();r['reused']=float('nan');t=measure(r,cache=False);assert set(t['outcomes'].condition)=={'P'} and set(t['outcomes'].phase)=={'R'}
