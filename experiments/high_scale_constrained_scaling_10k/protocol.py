"""Actual fresh RMS floors, exact native projection and Adam continuation."""
from .common import *
from experiments.high_scale_constrained_meta.initialization import groups,rms,project,TOL_ABS,TOL_REL
from experiments.heldout_context_meta.fomaml import outer_optimizer

def floors(state):return {k:rms(x) for k,x in groups(source_model(state)).items()}
def projection(m,f,o,condition):
 if condition not in ['U','P']:raise ValueError('Registered constraint condition required')
 return project(m,f,o,enabled=condition=='P')

def validate_checkpoint(b,arm,dose):
 count=dose//2 if arm=='B' else dose
 if b.get('arm')!=arm or b.get('meta_episode')!=dose or b.get('episode')!=count:raise ValueError('Wrong objective-specific meta cursor')
 if not all(k in b for k in ['torch_rng','numpy_rng','python_rng','registry_sha','code','model','optimizer']):raise ValueError('Incomplete restart metadata')
 if not all(torch.isfinite(v).all() for v in b['model'].values()):raise ValueError('Nonfinite source model')
 if arm!='R':
  opt=b['optimizer'];s=opt['state'] if opt else {}
  if len(s)!=6 or any(float(v['step'])!=count or not torch.isfinite(v['exp_avg']).all() or not torch.isfinite(v['exp_avg_sq']).all() for v in s.values()):raise ValueError('Missing or wrong outer Adam moments/counters')
 return True

def restore_outer(b,arm,dose):
 validate_checkpoint(b,arm,dose);m=source_model(b['model']);o=outer_optimizer(m);o.load_state_dict(copy.deepcopy(b['optimizer']));return m,o

def plan():return dict(new_meta_episodes=1336320,new_inner_updates=26726400,new_outer_Adam_updates=1139040,new_Reptile_interpolations=34560,new_target_runs=846,new_target_updates=386100,new_target_clones=10152,new_conflict_runs=504,new_conflict_updates=120960,new_scientific_optimizer_updates=28382652,existing_U_meta_runs=90,existing_target_histories=288,existing_conflict_references=90)
