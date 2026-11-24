import pandas as pd,numpy as np
from experiments.high_scale_meta_progression.analysis import analyze_one,paired_effects
from experiments.high_scale_meta_progression.common import OUT

def test_full_horizon_all_positions_and_original_late_gates():
 r=pd.read_csv(OUT/'registry/fresh.csv').iloc[0].to_dict();a=analyze_one(r);assert len(a['switch_costs'])==99 and len(a['blocks'])==99 and len(a['balanced_cycles'])==49 and len(a['position_switch_costs'])==99*7
 assert a['switch_costs'].iloc[-1].reacquisition_status=='end_of_run_censored'
 assert set(a['functional_splitting']['update'])=={170,330}
 assert set(a['outcomes']['update'])=={0,100,170,330,660,990};assert a['outcomes'].seed.nunique()==1
 for r in a['outcomes'].itertuples():assert abs(r.gap-(r.C_ce-r.familiar_ce))<1e-12

def test_paired_comparison_preserves_missing_seeds_and_signed_values():
 f=pd.DataFrame([dict(seed=s,scale=4,phase=a,dose=d,update=330,x=(s if a=='R' else -s)) for s in range(6) for a,d in [('R',1920),('T',240)]])
 raw,ci=paired_effects(f,['x']);g=raw[raw.contrast.eq('T240-R1920')];assert len(g)==6 and (g.effect<=0).all();assert ci[ci.contrast.eq('T240-R1920')].iloc[0]['n']==6
