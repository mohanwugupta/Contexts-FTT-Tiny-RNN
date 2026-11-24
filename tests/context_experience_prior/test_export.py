import csv
import pytest
from experiments.context_experience_prior.export_correction import decode_training,NEW_ORDER
from experiments.context_experience_prior.core import ordered_frame

def test_new_record_decoding_preserves_actual_logged_scalars(tmp_path):
 p=tmp_path/'bad.csv';row=['0','I','P1','P1-I','1','A','False',5.,1.,.4,5.08,8.,3.,True,'completed',True]
 with p.open('w') as f:
  w=csv.writer(f);w.writerow(['seed','organization','prior','condition','update','training_cue','archive_reused','archived_update_reused','trajectory_kind','status','optimizer_step_executed','ce_before','execution_loss','total_loss','accuracy_before','hidden_absmax','gradient_norm','amplitude_fraction_above_95','amplitude_fraction_above_99','new_optimizer_step_executed']);w.writerow(row)
 x=decode_training(p).iloc[0];assert x.ce_before==5 and x.execution_loss==.4 and x.total_loss==5.08 and x.status=='completed' and x.new_optimizer_step_executed

def test_append_requires_same_schema_and_reorders():
 import pandas as pd
 f=pd.DataFrame([{'b':2,'a':1}]);assert ordered_frame(f,['a','b']).iloc[0].tolist()==[1,2]
 with pytest.raises(ValueError):ordered_frame(f,['a','missing'])
