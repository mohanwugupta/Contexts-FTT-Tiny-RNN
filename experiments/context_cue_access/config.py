from pathlib import Path
from context_sequence.provenance.hashing import object_hash
from experiments.output_compatible_trajectories.config import SEQUENCE,PROBES,REGISTRY_SHA
ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'results/output_compatible_trajectories/v1_20261111'
DEFAULT=ROOT/'results/context_cue_access/v1'
CONTRACT={'version':'v1','cohort':'High_monotonic','seeds':list(range(20)),'probes':list(PROBES),'s1_k':[1,3,5],'s2_k':[3,5],
 'grid_points':13,'cue_min':-1.5,'cue_max':1.5,'third_channel':0.,'starts':6,'steps':80,'lr':.04,'betas':[.9,.999],'adam_epsilon':1e-8,
 'start_order':['A','B','Mixed','NoContext','target','best_fixed_candidate'],'boundary_initialization':'atanh(nextafter(1,0)) with sign; requested and actual values retained',
 'selection':'lowest double-surrogate iterate per start; original float32 source replay selects best start; fixed candidates reported separately',
 'sobol_n':512,'sobol_scramble':True,'sobol_seed':20261112,'independent_nesting':['target,u','u,u','u,target'],
 'domain_tolerance':1e-12,'bootstrap_seed':20260925,'bootstrap_n':10000,'rank_absolute':1e-8,'rank_relative':1e-4,'finite_difference_steps':[1e-3,1e-4],
 's3_schedules':['N0','R_oracle'],'s3_fd_poor_abs':.02,'s3_fd_poor_rel':.1,'saturation_derivative':1e-6,
 'subgroup':'any prior feasible P3 selected restart viable under imported criterion AND same-k native fails; far-positive separately recorded',
 'no_model_training':True,'no_hidden_placement':True,'cue_oracle_future_targets':True,'viability_implementation':'experiments.output_compatible_trajectories.continuation.future_summary'}
CONTRACT['sha256']=object_hash(CONTRACT)
