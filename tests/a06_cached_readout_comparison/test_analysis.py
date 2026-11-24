import numpy as np
import pandas as pd
import pytest
from experiments.a06_cached_readout_comparison import analysis as a

def test_static_preserves_each_checkpoint_bias_not_temperature():
 z=np.arange(2*5*7*4,dtype=float).reshape(2,5,7,4);b=np.array([[1.,-2.,3.,.5],[7.,2.,-1.,4.]])
 got=a.static_logits(z,b);np.testing.assert_array_equal(got,.25*(z-b[:,None,None,:])+b[:,None,None,:])
 assert not np.array_equal(got,z*.25)
 assert not np.array_equal(got[1],.25*(z[1]-b[0])+b[0])
 with pytest.raises(ValueError):a.static_logits(z,b[:1])

def test_explicit_labels_and_common_checkpoints_no_interpolation():
 f=pd.DataFrame([dict(update=k,cue=c) for k in [0,20] for c in ['A','B']])
 assert a.layout(f,(2,2,7,4),[0,20],['A','B'])==([0,20],['A','B'])
 with pytest.raises(ValueError):a.layout(f,(2,2,7,4),[0,10],['A','B'])
 with pytest.raises(ValueError):a.layout(pd.concat([f,f.iloc[:1]]),(2,2,7,4),[0,20],['A','B'])
 assert a.common_updates({'A00':[0,10,330],'A06':[0,20,330]})==[0,330]

def test_positions_correct_counts_signed_contributions_and_confident_errors():
 z=np.zeros((7,4));z[np.arange(7),a.core.TARGET]=2.;z[0]=[100,0,0,0]
 p=a.score_rows(z,seed=1,update=330,cue='C',arm='A00',source='fixture')
 zz=z.copy();zz[0]=[5,0,0,0];zz[1]=[1,0,0,0]
 q=a.score_rows(zz,seed=1,update=330,cue='C',arm='A06',source='fixture')
 f=pd.DataFrame(p+q);d=a.position_differences(f)
 t=d[d.contrast.eq('A06_minus_A00')];assert len(t)==7
 assert t.iloc[0].transition=='stayed_incorrect' and t.iloc[1].transition=='became_incorrect'
 assert t.ce_contribution.sum()==pytest.approx((f[f.arm.eq('A06')].ce.mean()-f[f.arm.eq('A00')].ce.mean()))
 assert sum(r['correct'] for r in p)==6 and sum(r['correct'] for r in q)==5
 assert t.iloc[0].ce_contribution<0 and t.iloc[1].ce_contribution>0

def test_native_masks_all_positions_and_no_accuracy_from_rounded_fraction():
 z=np.zeros((7,4));p=a.score_rows(z,0,0,'NoContext','A00','fixture')
 assert len(p)==7 and {r['position'] for r in p}==set(range(1,8))
 assert [r['target'] for r in p]==a.core.TARGET.tolist()
 f=a.aggregate(pd.DataFrame(p));assert f.iloc[0].correct_count==1
 assert f.iloc[0].ce==pytest.approx(np.log(4))

def test_original_classifications_copied_without_new_gates():
 f=pd.DataFrame([dict(candidate='A06',seed=1,classification='partial_signal',early=False,joint=False,Q_A=.2,Q_B=.3,floor_A=False)])
 g=a.preserve_early(f);assert g.iloc[0].classification=='partial_signal' and not g.iloc[0].joint and not g.iloc[0].early
 assert g.iloc[0].Q_A==.2 and g.iloc[0].has_practice_history

def test_missing_root_metrics_can_use_labeled_run_metrics():
 f=pd.DataFrame([dict(update=330,cue='C',ce=1.,correct_count=6)])
 q=a.registered_fallback([(0,'A06',f)])
 assert q.iloc[0].seed==0 and q.iloc[0].candidate=='A06' and q.iloc[0]['update']==330

def test_incompatible_static_view_keeps_original_name_and_flags():
 f=pd.DataFrame([dict(candidate='A06_static',seed=1,classification='frozen_expression_control',early=False,joint=False)])
 g=a.preserve_early(f,{1:False});assert g.iloc[0].arm=='original_A06_static' and not g.iloc[0].joint and not g.iloc[0].has_practice_history
