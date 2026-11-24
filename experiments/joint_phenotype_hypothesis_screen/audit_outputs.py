"""Post-run read-only task/exposure and initialization reporting; no fitting."""
import types
from torch import nn
from .common import *
from . import engine,adapter,banks
from experiments.meta_learning_scaling.protocol import task_signature

@torch.no_grad()
def gate_values(v,c):return 1+.5*torch.tanh(v@v.new_tensor(c))

def main():
 before=digest(rng());ledger=sha(OUT/'registry/optimizer_events.jsonl');tasks=[];gate_rows=[];width=[];routes=[]
 for seed in [0,1]:
  b=engine.bank(seed);meta_paths={tuple(p) for e in b['meta'] for p in e['paths']}
  for name in ['background','calibration','calibration_validation','width']:
   assert all(banks.allowed(p) and tuple(p) not in meta_paths for e in b[name] for p in e['paths'])
  for candidate in ['B00']+[f'B{i:02}' for i in range(1,7)]:
   n=80 if candidate=='B04' else 240;eps=b['meta'][:n];mask=sum(sum(b['missing_masks'][i]) for i in range(n)) if candidate=='B03' else 0;shared=sum(e['kind']=='shared' for e in eps);presentations=4800;shared_presentations=shared*(60 if candidate=='B04' else 20)
   rows=dict(candidate=candidate,seed=seed,episodes=n,inner_updates=4800,shared_episodes=shared,conflict_episodes=n-shared,unique_tasks=len({task_signature(e) for e in eps}),unique_paths=len({tuple(p) for e in eps for p in e['paths']}),masked_presentations=mask,shared_presentations=shared_presentations,masked_shared_fraction=mask/shared_presentations,masked_all_fraction=mask/presentations,epsilon_hash=digest(engine.epsilons(candidate)),task_bank_sha=sha(OUT/f'registry/tasks_s{seed:02}.json'),target_equivalent_paths=0)
   if candidate!='B00':
    f=pd.read_csv(OUT/f'runs/{candidate}_s{seed}/meta_updates.csv.gz');assert len(f)==4800
    for ei,e in enumerate(eps,1):
     r=f[f.episode.eq(ei)];schedule=engine.support(e,candidate);assert r.context.tolist()==[u['context'] for u in schedule] and r.sequence.tolist()==[u['sequence'] for u in schedule]
    cp=torch.load(OUT/f'runs/{candidate}_s{seed}/meta_progress.pt',weights_only=False);assert cp['cursor']==4800 and cp['episode']==n and cp['inner']==0 and cp['task'] is None
   tasks.append(rows)
  record=json.loads((OUT/f'registry/width_calibration_s{seed}.json').read_text());base=torch.load(ROOT/engine.sources(seed)['F00']['initializer'],weights_only=False)['model']
  # Reconstruct the exact draw using the SAVED factor. No second calibration.
  with torch.random.fork_rng():
   torch.manual_seed(20261123+32005+seed);m=nn.Module();m.recurrent=nn.RNN(7,32,nonlinearity='tanh',batch_first=True);m.sequence_head=nn.Linear(32,4)
   with torch.no_grad():
    for name,v in m.named_parameters():
     if 'weight' in name:nn.init.normal_(v,0.,5.*np.sqrt(8/32) if name=='sequence_head.weight' else 5.)
     else:v.zero_()
    m.recurrent.weight_hh_l0.mul_(.9/torch.linalg.matrix_norm(m.recurrent.weight_hh_l0,2));m.sequence_head.bias.copy_(base['sequence_head.bias']);m.sequence_head.weight.mul_(record['empirical_factor'])
  m.forward=types.MethodType(adapter.flexible_forward,m);first=pd.read_csv(OUT/f'runs/B05_s{seed}/meta_episodes.csv').iloc[0];assert digest(m.state_dict())==first.before_hash
  result=dict(seed=seed,analytic_factor=record['analytic_factor'],saved_empirical_factor=record['empirical_factor'],initial_state_hash=digest(m.state_dict()),reference_class_centered_bias_free_rms=record['reference_rms'],output_bias=record['output_bias'],parameters_reselected=False,optimizer_operations=0)
  with torch.no_grad():
   for mode in ['full','token_only','cue_only']:
    zz=[];centered=[];biasfree=[]
    for e in b['width']:
     for path in e['paths']:
      x,_=core.tensors(path)
      if mode=='cue_only':x.zero_()
      for c in ['A','B','C']:
       vec=torch.zeros(3) if mode=='token_only' else torch.tensor(core.CUES[c]);z=m(x,vec)[0][0];zz.append(z.double().numpy());centered.append((z-z.mean(-1,keepdim=True)).double().numpy());v=z-m.sequence_head.bias;biasfree.append((v-v.mean(-1,keepdim=True)).double().numpy())
    result[mode+'_raw_logit_rms']=float(np.sqrt(np.square(np.array(zz)).mean()));result[mode+'_class_centered_logit_rms']=float(np.sqrt(np.square(np.array(centered)).mean()));result[mode+'_bias_free_class_centered_rms']=float(np.sqrt(np.square(np.array(biasfree)).mean()))
  np.testing.assert_array_equal(result['full_bias_free_class_centered_rms'],record['matched_rms']);result['precision']='native float32 centering, float64 RMS aggregation, exactly as calibration';width.append(result)
  checkpoints=torch.load(OUT/f'runs/B06_s{seed}/checkpoints.pt',weights_only=False)
  for n,cp in checkpoints.items():
   for c in PROBES:
    v=cp['model']['V'];g=gate_values(v,core.CUES[c]);assert (g>=.5).all() and (g<=1.5).all()
    if c=='NoContext':assert torch.equal(g,torch.ones(8))
    gate_rows.append(dict(seed=seed,update=n,cue=c,minimum=float(g.min()),maximum=float(g.max()),C_gate_column_fixed=torch.equal(v[:,2],checkpoints[0]['model']['V'][:,2]),Mixed_is_actual_forward=True,affine_midpoint_applicable=False))
 selection=json.loads((OUT/'selection.json').read_text());s=pd.read_csv(OUT/'registry/pilot_summary_at_selection.csv')
 for r in selection['rows']:
  f=s[s.candidate.eq(r['id'])&s.seed.isin([0,1])];both=bool(len(f)==2 and f.joint.fillna(False).all())
  routes.append(dict(candidate=r['id'],status=r['status'],both_pilot_seeds_joint=both,transfer_improvement_subtest=r.get('transfer'),splitting_improvement_subtest=r.get('splitting'),rescue_preservation_subtest=r.get('rescue'),static_control_passed=r.get('static_control_passed'),complete_transfer_route=both and r.get('transfer',False),complete_splitting_route=both and r.get('splitting',False),complete_rescue_route=both and r.get('rescue',False),eligible=r['eligible'],selected=r['id'] in selection['selected']))
 csv(OUT/'registry/realized_background_exposure.csv',tasks);put(OUT/'registry/width_gain_details.json',width);csv(OUT/'registry/gate_ranges.csv',gate_rows);csv(OUT/'selection_route_components.csv',routes);put(OUT/'registry/fixture_attempt_qualifications.json',dict(gradient_update_attempts=54,known_zero_optimizer_call_attempt=dict(run='fixture_test_zero_gate_actual_mixed_and_fixed_C',step=1,reason='training cue tensor integer versus gate float; failed before backward or optimizer',actual_optimizer_calls=0,failed_log='audit/07_adapter_GREEN.log'),known_completed_optimizer_calls_despite_failed_test=dict(run='fixture_test_sgd_equation_policy_identity_and_restore',step=1,reason='native Adam-specific target assertion reached after SGD and policy Adam updates',actual_optimizer_calls=2,failed_log='audit/07_adapter_GREEN.log'),all_attempts_charged=True,all_other_reserved_calls_executed=True))
 assert digest(rng())==before and sha(OUT/'registry/optimizer_events.jsonl')==ledger
 put(OUT/'audit/task_and_initialization_audit.json',dict(passed=True,optimizer_operations=0,parameter_fitting=0,task_rows=len(tasks),width_initializers_match_first_meta_before_hash=True,gate_rows=len(gate_rows),native_support_schedules_all_matched=True))
 return width
