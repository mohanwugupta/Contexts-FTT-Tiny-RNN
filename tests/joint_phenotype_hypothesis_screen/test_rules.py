import copy
import numpy as np
import pandas as pd
import pytest
from experiments.joint_phenotype_hypothesis_screen import rules

def reference():
 return {'Q_A':.8,'Q_B':.6,'E_normalized':.2,'resolved':True}

def valid(seed=0):
 return dict(seed=seed,Q_A=.6,Q_B=.5,E=2.,E_normalized=.15,net_A=1.,net_B=1.,familiar=True,C_useful=True,Mixed_useful=True,C_strict=True,C_ce=.08,C_correct=7,early=True,joint=True,strict_joint=True)

def test_specificity_signed_gain_and_floor():
 a=rules.direction(10.,1.,2.,'A');assert a['D_A']==-1 and a['Q_A']==-1
 b=rules.direction(10.,2.,-1.,'B');assert b['D_B']==3 and b['Q_B']==1 and b['interference_B']
 for start,own in [(.05,.01),(1.,.05),(1.,0.),(1.,-1.)]:assert not np.isfinite(rules.direction(start,own,0.,'A')['Q_A'])

def test_integer_accuracy_gate_and_reference_not_replaceable():
 assert rules.useful(.5,6) and not rules.useful(.5,5)
 x=valid();assert rules.early_gate(x,reference())
 for key,value in [('net_A',0.),('Q_B',.29),('E',0.)]:
  y=dict(x,**{key:value});assert not rules.early_gate(y,reference())
 assert not rules.early_gate(x,dict(reference(),resolved=False))

def test_selection_requires_both_seeds_and_fixed_routes():
 base=pd.DataFrame([dict(valid(i),C_ce=.5,joint=False,strict_joint=False) for i in [0,1]])
 x=pd.DataFrame([valid(i) for i in [0,1]])
 r=rules.route(x,base,'A01');assert r['eligible'] and r['transfer'] and r['rescue']
 y=x.copy();y.loc[1,'joint']=False;assert not rules.route(y,base,'A01')['eligible']
 y=x.copy();y.loc[1,'C_correct']=5;assert not rules.route(y,base,'A01')['eligible']
 assert not rules.route(x,base,'A06',x)['eligible']
 z=x.copy();z['C_ce']+=.1;assert rules.route(x,base,'A06',z)['eligible']

def test_rescue_preservation_and_noncompensatory_ranking():
 b=pd.DataFrame([valid(i) for i in [0,1]]);b.loc[1,'joint']=False
 a=b.copy();a['joint']=True;a['strict_joint']=True;a.loc[0,'E_normalized']=.01
 assert not rules.route(a,b,'A02')['eligible']
 rows=[dict(id='B01',strict_count=2,retention=.5,C_ce=.001,eligible=False),dict(id='A03',strict_count=1,retention=.6,C_ce=.2,eligible=True),dict(id='A01',strict_count=1,retention=.6,C_ce=.2,eligible=True)]
 assert rules.rank(rows)==['A01','A03']

def test_rescue_accuracy_preservation_applies_to_already_joint_controls():
 b=pd.DataFrame([dict(valid(i),joint=False) for i in [0,1]])
 a=pd.DataFrame([dict(valid(i),C_correct=6,C_ce=.4,C_strict=False,strict_joint=False) for i in [0,1]])
 assert rules.route(a,b,'A01')['eligible']

def test_switch_indexing_same_weight_and_signed_adjustment():
 rows=[]
 for n in [0,10,20,30,40,50,60]:
  for c,loss in [('A',2.+n/100),('B',4.+n/50)]:rows.append(dict(update=n,cue=c,ce=loss))
 d=rules.switches(pd.DataFrame(rows));assert d.loc[0,'incoming']=='B' and d.loc[1,'incoming']=='A'
 np.testing.assert_allclose(d.adjusted_switch,[.1,-.2,.3,-.4,.5,-.6])
 with pytest.raises((ValueError,KeyError)):rules.switches(pd.DataFrame(rows).iloc[1:])
