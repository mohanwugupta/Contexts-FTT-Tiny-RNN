"""Frozen source chain, actual weight RMS histories, new initial-state provenance."""
import shutil
from .common import *
from .initialization import initialize,groups,rms
from experiments.meta_learning_scaling.common import source_initial
from experiments.heldout_context_meta.common import initial as historical_fomaml

def metrics(state):
 m=source_model(state);r={name:rms(x) for name,x in groups(m).items()};r.update(recurrent_rms=rms(m.recurrent.weight_hh_l0),recurrent_spectral_norm=float(torch.linalg.matrix_norm(m.recurrent.weight_hh_l0.detach().double(),2)),bias_rms=float(np.sqrt(np.mean(np.concatenate([p.detach().numpy().reshape(-1) for n,p in m.named_parameters() if 'bias' in n])**2))));return r

def main():
 initialize_process();start=time.monotonic();manifest=PARENT/'manifest.json';b=json.loads(manifest.read_text());core.hashes(PARENT,b['outputs']);core.hashes(ROOT,b['code']);core.hashes(ROOT,b['inputs']);refs={str((PARENT/k).relative_to(ROOT)):v for k,v in b['outputs'].items()};refs.update(b['code']);refs.update(b['inputs']);refs[str(manifest.relative_to(ROOT))]=sha(manifest)
 explicit=['confirmatory_srn/model.py','experiments/high_scale_initialization/REPORT.md']
 for name in ['confirmatory_srn/model.py','experiments/context_experience_prior/core.py','experiments/context_experience_prior/meta.py','experiments/meta_learning_scaling/continuation.py','experiments/meta_learning_scaling/protocol.py','experiments/heldout_context_meta/fomaml.py','experiments/heldout_context_meta/protocol.py','experiments/recurrent_learning_transfer/measures.py','results/high_scale_initialization/v1/REPORT.md']:
  refs[name]=sha(ROOT/name)
 dump(OUT/'registry/parent_hashes.json',refs);rows=[];initial_rows=[];reserved=set();reuse=[]
 for s in range(20):
  # All20 historical models remain, distinct from the six new initializations.
  p=core.OUT/f'meta/seed_{s}_P2_I.pt';obj=torch.load(p,weights_only=False);r0=obj['records'][0]['meta'];r240=obj['records'][240]['meta'];assert all(torch.equal(v,core.base_pair(s)['I'].state_dict()[k]) for k,v in r0.items())
  states=[('Reptile',0,r0,p),('Reptile',240,r240,p)]
  for dose in [960,1920]:
   st,path,_=source_initial(s,'P2','I',dose);states.append(('Reptile',dose,st,ROOT/path))
  for dose in [240,960]:
   st,path=historical_fomaml(s,'P2','I','Transfer',dose);states.append(('FOMAML',dose,st,path))
  for phase,dose,state,path in states:rows.append(dict(seed=s,phase=phase,episode=dose,source=str(path.relative_to(ROOT)),source_sha=sha(path),**metrics(state)))
  d=ROOT/f'results/heldout_context_meta/v1/registry/heldout_v2_seed_{s}.json';eps=json.loads(d.read_text())['episodes'];reserved.update(tuple(p) for e in eps for p in e['paths'])
  if s in SEEDS:
   shutil.copyfile(d,OUT/f'registry/heldout_v2_seed_{s}.json');high,low,floors,raw=initialize(s)
   for arm,m in [('High',high),('Low',low)]:
    path=OUT/f'initial/s{s:02d}_{arm}.pt';h=atomic_save(path,dict(seed=s,arm=arm,model=copy.deepcopy(m.state_dict()),raw_reference=raw.state_dict(),floors=floors,initialization_sd=2. if arm=='High' else .02,recurrent_spectral_normalization=.9,biases_zero=True,code=codes()));initial_rows.append(dict(seed=s,arm=arm,path=str(path.relative_to(ROOT)),sha256=h,**metrics(m.state_dict()),**{'floor_'+k:v for k,v in floors.items()}));reuse.append(dict(seed=s,arm=arm,original_raw_P0_state_identical=all(torch.equal(v,r0[k]) for k,v in m.state_dict().items()),native_full_target_reusable=False,reason='All3nonzero cue columns; high minimum floors or low SD differ from old P0 target initialization'))
  print('Source scales audited seed'+str(s),flush=True)
 assert not any(r['original_raw_P0_state_identical'] for r in reuse),'Exact branch match: stop audit and implement reuse before any target steps'
 write_frame(OUT/'tables/historical_scale_trajectories.csv',rows);write_frame(OUT/'registry/initializations.csv',initial_rows);write_frame(OUT/'registry/preflight_reuse_audit.csv',reuse);dump(OUT/'registry/diagnostic_paths.json',sorted(reserved));dump(OUT/'audit/source_audit.json',dict(passed=True,all20_historical_seeds=True,historical_checkpoints=len(rows),new_seeds=SEEDS,original_tanh_SD=2.,original_context_AB_SD=2.,original_C_neutral=True,original_recurrence_normalization=False,new_Linear_original_recurrence_normalization=.9,recurrent_floor_not_applied=True,parent_manifest_sha=sha(manifest),wall_seconds=time.monotonic()-start));seal={str(p.relative_to(OUT)):sha(p) for p in (OUT/'initial').glob('*.pt')};seal.update({str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('*') if p.is_file()});dump(OUT/'audit/initial_registry_seal.json',dict(passed=True,files=seal,preregistration_sha=sha(OUT/'registry/PREREGISTRATION.md'),code=codes()))
 append('## Source/reuse audit complete\nAll20 prior P2-I trajectories at Reptile0/240/960/1920 and FOMAML240/960 measured by actual group RMS, never by initialization SD. Original Tanh SD2 has unnormalized recurrence and a neutral C channel; the current Linear source normalizes R once to.9 and uses all3nonzero columns. Every prediction tensor can change in both meta algorithms; policy is reset and not meta-updated. New six-seed High/Low initial states and per-group floors sealed; exact raw P0 branch matching rejected as inapplicable before target execution. No optimizer steps in audit.')
 print('Source and initial registry audit PASSED',flush=True)
if __name__=='__main__':main()
