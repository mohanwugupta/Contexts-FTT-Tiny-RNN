import numpy as np
import pytest
from experiments.structured_overlap_meta.pilot_verify import validate_steps

def test_capacity_cannot_reset_optimizer_or_train_withheld_context():
 schedule=[dict(update=k,role='X' if (k-1)//10%2==0 else 'Y',cue='A' if (k-1)//10%2==0 else 'B',optimizer_step_executed=True,total_loss=1.1,ce_before=1.,execution_loss=.5) for k in range(1,331)]
 assert validate_steps(schedule,dict(X='A',Y='B',Q='C'))
 wrong=[dict(r) for r in schedule];wrong[10]['cue']='C'
 with pytest.raises(ValueError):validate_steps(wrong,dict(X='A',Y='B',Q='C'))
 wrong=[dict(r) for r in schedule];wrong[-1]['optimizer_step_executed']=False
 with pytest.raises(ValueError):validate_steps(wrong,dict(X='A',Y='B',Q='C'))
