import numpy as np
import pandas as pd
import pytest
from experiments.meta_objective_balance.selection import choose
from experiments.meta_objective_balance.statistics import interval

def fixture():
 spec={'A':(.05,.65),'B':(.08,.7),'C':(.05,.6),'D':(.14,.72),'E':(.3,.9),'F':(.18,.8)}
 return pd.DataFrame([dict(seed=s,arm=a,gap=g,conflict_accuracy=v,conflict_ce=1-v,strong_familiar=1,D_A=.2,D_B=.1,nonfloor_both=True) for a,(g,v) in spec.items() for s in range(6)])
def test_selection_is_seed_prespecified_tolerance_and_multivariate_not_best_seed():
 f=fixture();result,summary=choose(f);assert result['selected']==['D','A'] and result['tolerance_met']
 f.loc[f.arm!='C','gap']+=.5;r,_=choose(f);assert r['selected']==['A','E'] and not r['tolerance_met']
 f=fixture();f.loc[(f.arm=='D')&(f.seed==0),'strong_familiar']=0;r,_=choose(f);assert 'D' not in r['selected']
 f=fixture();f.loc[0,'seed']=6
 with pytest.raises(ValueError):choose(f)
def test_whole_seed_intervals_preserve_missing_signed_effects_and_medians():
 r=interval(np.array([-1.,2.,np.nan]),[6,7,8]);assert r['n']==2 and r['n_planned']==3 and r['mean']==.5 and r['median']==.5 and r['ci_low']<=0<=r['ci_high']
 r=interval(np.full(14,np.nan),list(range(6,20)));assert r['n']==0 and np.isnan(r['mean']) and np.isnan(r['ci_low'])

def test_archived_baseline_is_reference_not_seventh_screen_arm():
 from experiments.meta_objective_balance.selection import screen_candidates
 f=fixture().assign(dose=480);baseline=f[f.arm.eq('C')].assign(arm='Baseline',dose=0);all_rows=pd.concat([f,baseline],ignore_index=True)
 q=screen_candidates(all_rows);assert len(q)==36 and set(q.arm)==set('ABCDEF')
 r,_=choose(q);assert r['selected']==['D','A']
