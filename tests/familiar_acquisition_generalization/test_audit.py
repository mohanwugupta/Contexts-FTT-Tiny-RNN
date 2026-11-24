import numpy as np
import pandas as pd
import pytest
from experiments.familiar_acquisition_generalization.audit import validate, aggregate, compare, bootstrap, classify, separation


def fixture():
 rows=[]
 for seed in range(20):
  for arm,step in [('Original',170),('B',330),('I',330),('S',330)]:
   for cue in ['A','B','Mixed','NoContext','Novel30_A','Novel30_B','Novel60_A','Novel60_B']:
    for t in range(1,8):
     y=[0,1,2,0,3,1,2,3][t];z=np.zeros(4);z[y]=2
     rows.append(dict(seed=seed,arm=arm,step=step,cue=cue,transition=t,token=[0,1,2,0,3,1,2,3][t-1],target=y,raw_ce=np.log(np.exp(2)+3)-2,checkpoint_sha='x',**{f'logit_{i}':z[i] for i in range(4)},**{f'hidden_{i}':float(cue=='A') for i in range(8)}))
 return pd.DataFrame(rows)


def test_complete_indices_no_duplicate_or_missing():
 f=fixture();assert validate(f)
 for bad in [f.iloc[:-1],pd.concat([f,f.iloc[[0]]])]:
  with pytest.raises(ValueError):validate(bad)
 f.loc[0,'target']=0
 with pytest.raises(ValueError):validate(f)


def test_accuracy_gap_novelmean_and_separation():
 f=fixture();q=aggregate(f);assert len(q)==80 and q.accuracy_A.eq(1).all()
 assert np.allclose(q.gap_Mixed,0) and np.allclose(q.ce_NovelMean,q.ce_Novel30_A)
 assert np.allclose(q.hidden_separation,np.sqrt(8)) and np.allclose(q.logit_separation,0)
 # Invariance to class-common logit shifts; mean of per-position distances.
 a=f[(f.seed==0)&(f.arm=='Original')&(f.cue=='A')];b=f[(f.seed==0)&(f.arm=='Original')&(f.cue=='B')].copy()
 for i in range(4):b[f'logit_{i}']+=3
 assert separation(a,b)[1]==pytest.approx(0,abs=1e-12)


def test_paired_changes_do_not_confuse_gap_and_absolute_loss():
 q=aggregate(fixture());q.loc[q.arm!='Original','familiar_ce']-=.5;q.loc[q.arm!='Original','ce_Mixed']-=.1;q['gap_Mixed']=q.ce_Mixed-q.familiar_ce
 c=compare(q);assert len(c)==60
 assert np.allclose(c.delta_familiar_ce,-.5) and np.allclose(c.delta_ce_Mixed,-.1) and np.allclose(c.delta_gap_Mixed,.4)


def test_seed_bootstrap_rejects_pseudoreplication_and_missing():
 q=pd.DataFrame({'seed':range(20),'effect':range(20)});assert bootstrap(q)['mean']==9.5
 for bad in [pd.concat([q,q]),q.iloc[:-1]]:
  with pytest.raises(ValueError):bootstrap(bad)


def test_classification_requires_both_contexts_and_both_separations():
 q=dict(ce_A=.1,ce_B=.7,accuracy_A=1.,accuracy_B=1.,ce_Mixed=2.,accuracy_Mixed=.2,gap_Mixed=1.6,hidden_retention=1.,logit_retention=1.)
 assert not classify(q)['strong_familiar']
 q['ce_B']=.1;assert classify(q)['strong_familiar'] and classify(q)['strong_retained_poor']
 q['logit_retention']=.1;assert not classify(q)['strong_retained_poor']
