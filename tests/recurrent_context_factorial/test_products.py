import pandas as pd
import pytest
from experiments.recurrent_context_factorial.figures import validate_training

def test_actual_loss_plot_requires_all20_and_allupdates():
 f=pd.DataFrame([dict(seed=s,arm=a,update=k,training_cue='A' if (k-1)//10%2==0 else 'B',ce_before=1.) for s in range(20) for a in ['HI','HC','NI','NC'] for k in range(1,331)])
 validate_training(f)
 with pytest.raises(ValueError):validate_training(f[f.seed!=19])
 f.loc[f['update']==11,'training_cue']='A'
 with pytest.raises(ValueError):validate_training(f)

def test_complete_scientific_artifacts_without_training():
 from experiments.recurrent_context_factorial.verify import verify
 result=verify(sealed=False)
 assert result['part_a_evaluations']==79440
 assert result['part_b_evaluations']==158880
 assert result['new_optimizer_steps']==26300 and result['archived_updates_reused']==100
 assert result['part_b_states']==26480 and result['training_replayed']==False
