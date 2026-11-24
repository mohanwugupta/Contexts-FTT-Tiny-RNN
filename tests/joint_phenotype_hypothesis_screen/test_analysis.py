import copy
import numpy as np
import pandas as pd
from experiments.joint_phenotype_hypothesis_screen import analysis
from experiments.joint_phenotype_hypothesis_screen.common import PROBES,TIMES

def frame():
 rows=[]
 for n in TIMES:
  for c in PROBES:
   ce=.1 if n>=100 else (10.-min(n,60)*.1+(1 if (n//10)%2 and c=='B' else 0))
   rows.append(dict(update=n,cue=c,ce=ce,correct_count=7,margin=1.,gap=0.,useful=True,strict=True,historical_strict_gap_rule=True))
 return pd.DataFrame(rows)
def test_summary_preserves_same_weight_signs_and_integer_acquisition():
 f=frame();ref=dict(resolved=True,Q_A=.1,Q_B=.1,E_normalized=.01);r=analysis.seed_summary(f,'A01',0,'pilot',ref)
 assert r['own_A']==1 and r['cross_A']==0 and r['D_A']==1 and r['Q_A']==1
 assert r['own_B']==2 and r['cross_B']==1 and r['Q_B']==.5
 g=f.copy();g.loc[(g['update']==330)&(g.cue=='A'),'correct_count']=5
 assert not analysis.seed_summary(g,'A01',0,'pilot',ref)['familiar']
 g=f.copy();g.loc[g.cue=='NoContext','ce']=100.
 rr=analysis.seed_summary(g,'A01',0,'pilot',ref);assert rr['early']==r['early'] and rr['joint']==r['joint']

def test_nonlearning_not_shared_and_negative_control_qualified():
 f=frame();f.loc[f.cue=='A','ce']=.01;r=analysis.seed_summary(f,'B01',0,'pilot',dict(resolved=True,Q_A=.1,Q_B=.1,E_normalized=.01))
 assert r['floor_A'] and not np.isfinite(r['Q_A']) and not r['joint']
 r.update(joint=False,familiar=False,C_ce=1.,C_correct=5,E_normalized=0.,Q_A=np.nan,Q_B=0.)
 control=dict(r);assert analysis.classify(r,control,False)=='budget_inconclusive'
