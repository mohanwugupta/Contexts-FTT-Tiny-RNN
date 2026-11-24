import numpy as np
import pandas as pd
from experiments.joint_phenotype_hypothesis_screen import report,figures

def test_no_confirmation_is_not_four_failures():
 s=pd.DataFrame([dict(candidate='A01',seed=0,role='pilot',familiar=True,joint=True,strict_joint=True)])
 r=report.confirmation_summary(s,[]);assert r==[]
 assert figures.confirmation_message([]).startswith('No candidates qualified')

def test_confirmation_requires_four_acquisition_and_three_same_seed_joint():
 s=pd.DataFrame([dict(candidate='A01',seed=i,role='confirmation',familiar=True,joint=i!=5,strict_joint=i==2) for i in [2,3,4,5]])
 r=report.confirmation_summary(s,['A01'])[0];assert r['familiar_passes']==4 and r['joint_passes']==3 and r['reproducible_feasibility']
 s.loc[s.seed.eq(5),'familiar']=False
 assert not report.confirmation_summary(s,['A01'])[0]['reproducible_feasibility']
 assert report.confirmation_summary(s.iloc[:3],['A01'])[0]['status']=='incomplete_confirmation'
