import pytest
from experiments.heldout_context_meta_scaling_10k.final_checks import expected_counts,validate_grid

def test_all_original_seeds_and_doses_required():
 c=expected_counts();assert c['physical_histories']==1200 and c['frozen_cells']==2383200 and c['new_target_positions']==11121600 and c['new_optimizer_steps']==30763200
 good=[dict(seed=s,condition=k,arm=a,dose=d) for s in range(20) for k in ['P1-I','P1-C','P2-I','P2-C'] for d in [0,240,960,1920,3840,5760,7680,10000] for a in (['Historical'] if d==0 else ['Seen','Transfer'])]
 assert validate_grid(good)
 with pytest.raises(ValueError):validate_grid(good[:-1])
 with pytest.raises(ValueError):validate_grid(good+[good[0]])

def test_frozen_grid_rejects_shifted_time_and_wrong_cue():
 import pandas as pd
 from experiments.heldout_context_meta_scaling_10k.final_checks import validate_frozen_grid
 cues=['A','B','C','Mixed-average','Mixed-both','NoContext']
 f=pd.DataFrame([dict(seed=0,condition='P1-I',arm='Transfer',dose=d,update=k,cue=c) for d in [960,10000] for k in range(3) for c in cues]);assert validate_frozen_grid(f,last=2)
 bad=f.copy();bad.loc[0,'update']=3
 with pytest.raises(ValueError):validate_frozen_grid(bad,last=2)
 bad=f.copy();bad.loc[bad.cue=='NoContext','cue']='Novel'
 with pytest.raises(ValueError):validate_frozen_grid(bad,last=2)
 with pytest.raises(ValueError):validate_frozen_grid(pd.concat([f,f.iloc[[0]]]),last=2)
