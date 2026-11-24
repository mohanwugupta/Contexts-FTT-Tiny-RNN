import pandas as pd
from experiments.a06_novel_cue_followup import reporting as r
def test_joint_counts_do_not_combine_different_seeds():
 f=pd.DataFrame([dict(condition='A06',seed=0,screen_joint=True,original_early_joint_not_interference_dominated=True,C_useful=True,familiar=True),dict(condition='A06',seed=1,screen_joint=False,original_early_joint_not_interference_dominated=False,C_useful=False,familiar=True)])
 q=r.counts(f);assert q.iloc[0].screen_joint==1 and q.iloc[0].familiar==2 and q.iloc[0].n==2
