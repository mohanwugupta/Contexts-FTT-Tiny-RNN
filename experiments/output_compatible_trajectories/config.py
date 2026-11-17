"""Behavior-independent budgets. Never rewrite after protected outcomes."""
from pathlib import Path
import json
import hashlib
import numpy as np
from context_sequence.provenance.hashing import object_hash

VERSION='v1_20261111'
SEQUENCE=(0,1,2,0,3,1,2,3)
PROBES=('Mixed','NoContext','Novel30_A','Novel30_B','Novel60_A','Novel60_B')
K=(1,3,5)
REGISTRY_SHA='2b28095562ea69a33d96c521817dcd0743f3d66658385ac425212f6951e00727'
CONTRACT={'version':VERSION,'sequence':list(SEQUENCE),'seeds':list(range(20)),'k':list(K),'probes':list(PROBES),
 'p1_p2_cohorts':['High_monotonic','C0_canonical','C1_contrast_only','C2_closure_only','C3_full'],
 'p3_p4_cohorts':['High_monotonic'],'p2_directions':24,'p2_signs':[-1,1],'p2_radii':[.1,.3],
 'p2_natural_directions':12,'p2_natural_radii':[.1],'box_margin':1e-6,'far_relative_distance':.25,
 'viability_slack_nats':.10,'practical_improvement_nats':.20,'reporting_tolerance_nats':.01,
 'null_tolerance':1e-10,'softmax_tolerance':1e-10,'ce_equivalence_tolerance':1e-10,
 'box_tolerance':1e-10,'historical_ce_tolerance':.002,'historical_hidden_tolerance':.0002,'historical_logit_tolerance':.002,
 'oracle_starts':9,'oracle_random_starts':8,'oracle_steps':150,'oracle_learning_rate':.02,
 'oracle_adam_betas':[.9,.999],'oracle_adam_epsilon':1e-8,'near_radius_fraction':.5,
 'oracle_random_radii':[.1,.1,.1,.1,.3,.3,.3,.3],
 'oracle_selection':'lowest smooth-double objective iterate per fixed restart; replay every selected state with actual float32 source recurrence; best restart selected by source future CE',
 'oracle_constraints':'kernel-coordinate line backoff to box and near ball; never clamp hidden coordinates',
 'infeasible_starts':'retain each planned start and flag invalid; no replacement, hidden clipping or extra restarts',
 'oracle_stall_tolerance':1e-8,'direction_rng_seed':20261111,'bootstrap_seed':20260925,'bootstrap_draws':10000,
 'smoke':{'seeds':[0,1,2,3],'k':[1,3],'probes':['Mixed','NoContext'],'directions':2,'oracle_starts':2,'oracle_steps':10},
 'p4_sources':['A','B',*PROBES],'p4_branches':['native_source','A','B','Mixed','zero','target_native'],
 'p4_baseline':'target_native pulse from the identical source prestate; persistent schedules are separate estimands',
 'no_model_training':True,'oracle_is_future_target_informed':True}
CONTRACT['contract_sha256']=object_hash(CONTRACT)


def rng_for(*identity):
    digest=hashlib.sha256(json.dumps([CONTRACT['direction_rng_seed'],*identity],separators=(',',':')).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8],'little'))


def subset(records,stage,smoke=False):
    allowed=CONTRACT['p3_p4_cohorts'] if stage in ('p3','p4') else CONTRACT['p1_p2_cohorts']
    seeds=CONTRACT['smoke']['seeds'] if smoke else CONTRACT['seeds']
    return [r for r in records if r['condition'] in allowed and r['seed'] in seeds]
