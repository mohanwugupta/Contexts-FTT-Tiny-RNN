import copy,numpy as np,pytest
from experiments.high_scale_constrained_meta.validation import validate_record

def fixture():return dict(inner_steps=20,inner_steps_array=np.full(8,20),inner_reset=True,policy_reset=True,query_unchanged=True,moments_preserved=True,outer_executed=True,phase='Reptile',arm='U',updates=[dict(inner_update=i+1,block=i//5+1,context='A' if i//5%2==0 else 'B',sequence=0,status='completed',optimizer_step_executed=True,ce_before=1.,execution_loss=.1,total_loss=1.02) for i in range(20)],after=np.zeros((3,172)),unprojected=np.zeros((3,172)),projection=[dict(enabled=False) for _ in range(5)])
def test_negative_schedule_objective_and_moment_fixtures():
 e={'support_P2':[{k:u[k] for k in ['inner_update','block','context','sequence']} for u in fixture()['updates']]};validate_record(fixture(),e)
 for change in ['schedule','objective','moments','inner']:
  r=fixture()
  if change=='schedule':r['updates'][4]['context']='B'
  if change=='objective':r['updates'][0]['total_loss']=1.
  if change=='moments':r['after'][1,0]=.1
  if change=='inner':r['inner_steps_array'][0]=19
  with pytest.raises(ValueError):validate_record(r,e)
