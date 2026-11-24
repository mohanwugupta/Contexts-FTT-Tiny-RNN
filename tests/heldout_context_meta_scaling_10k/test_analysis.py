import numpy as np
import pandas as pd
from experiments.heldout_context_meta_scaling_10k.analysis import scaling_effects,early_updates,primary_filter

def test_signed_early_same_weight_gains_and_floor():
 rows=[]
 for k,a,b in [(0,2.,3.),(1,1.,2.5),(2,.01,.02),(3,.009,.019)]:
  for cue,ce in [('A',a),('B',b)]:rows.append(dict(seed=0,condition='P1-I',arm='Transfer',dose=10000,update=k,cue=cue,ce=ce))
 x=early_updates(pd.DataFrame(rows));assert x.iloc[0].own_gain==1 and x.iloc[0].cross_gain==.5 and x.iloc[0].specificity==.5
 assert x.iloc[2].floor_at_start

def test_paired_changes_keep_seeds_and_primary_cannot_pick_checkpoint():
 f=pd.DataFrame([dict(seed=s,condition=c,arm=a,dose=d,update=330,cue='C',gap=s+(d/10000 if a=='Seen' else -d/10000)) for s in range(20) for c in ['P1-I','P2-I','P1-C','P2-C'] for a in ['Seen','Transfer'] for d in [960,1920,10000]])
 e=scaling_effects(f,['update','cue'],['gap']);p=primary_filter(e)
 assert len(p)==80 and set(p.condition)=={'P1-I','P2-I'} and set(p.contrast)=={'Transfer:10000-960','Transfer-Seen'}
 v=p[p.contrast=='Transfer:10000-960'];np.testing.assert_allclose(v.effect,-.904)

 # Secondary scaling interaction does not replace the four primary comparisons.
 interaction=e[(e.dose==10000)&e.contrast.eq("query_arm_scale_interaction:10000-960")]
 assert len(interaction)==80
 np.testing.assert_allclose(interaction.effect,-1.808)

def test_boolean_accuracy_contrasts_are_signed_numeric_differences():
 f=pd.DataFrame([dict(seed=s,condition='P1-I',arm=a,dose=d,update=330,position=1,correct=bool((s%2==0)==(a=='Transfer'))) for s in range(20) for a in ['Seen','Transfer'] for d in [960,10000]])
 e=scaling_effects(f,['update','position'],['correct']);x=e[e.contrast.eq('Transfer-Seen')&e.dose.eq(10000)].set_index('seed').reindex(range(20))
 assert len(x)==20
 np.testing.assert_array_equal(x.effect,[1. if s%2==0 else -1. for s in range(20)])
