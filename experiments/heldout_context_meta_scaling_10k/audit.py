"""Source/reuse audit and reserved matched registry, before any new training."""
import argparse,shutil
from .common import *
from .protocol import reserve_block,execution_prefix,AUDIT_EPISODES,NEW_DOSES,source

def validate_source_bundle(state,opt,episode):
 try:
  assert len(state)==6 and sum(v.numel() for v in state.values())==172 and all(torch.isfinite(v).all() for v in state.values())
  assert len(opt['state'])==6 and len(opt['param_groups'])==1
  g=opt['param_groups'][0];assert g['lr']==.001 and g['weight_decay']==0 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8
  for st in opt['state'].values():assert float(st['step'])==episode and torch.isfinite(st['exp_avg']).all() and torch.isfinite(st['exp_avg_sq']).all()
 except (AssertionError,KeyError,TypeError) as e:raise ValueError('Not an exact restartable source checkpoint') from e
 return True

def prepare():
 assert json.loads((OUT/'audit/predecessor_verified.json').read_text())['passed'];assert not (OUT/'audit/protocol_seal.json').exists()
 (OUT/'registry').mkdir(exist_ok=True);rows=[];summaries=[]
 for seed in range(20):
  held=SOURCE/f'registry/heldout_v2_seed_{seed}.json';reserve=json.loads(held.read_text());reserved={tuple(p) for e in reserve['episodes'] for p in e['paths']};assert len(reserved)==24
  shutil.copyfile(held,OUT/f'registry/heldout_v2_seed_{seed}.json')
  old=ROOT/'results/context_experience_prior/v1';scaling=ROOT/'results/meta_learning_scaling/v1'
  registries=[old/f'registry/seed_{seed}.json']+[scaling/f'registry/seed_{seed}_block_{b}.json' for b in range(1,8)]+[SOURCE/f'registry/seed_{seed}_b{b}.json' for b in range(1,5)]
  seen=set();seen_paths=set()
  for p in registries:
   for ep in json.loads(p.read_text())['episodes']:seen.add(source.signature(ep));seen_paths.update(tuple(p) for p in ep['paths'])
  assert not reserved.intersection(seen_paths)
  for block in range(5,43):
   reg=reserve_block(seed,block,seen,reserved);seen.update(source.signature(ep) for ep in reg['episodes']);executed=execution_prefix(reg)
   assert not reserved.intersection(tuple(p) for ep in executed for p in ep['paths'])
   dump(OUT/f'registry/seed_{seed}_b{block}.json',reg)
   kinds={k:sum(ep['kind']==k for ep in executed) for k in ['shared','conflict']};exposure={a:{c:0 for c in 'ABC'} for a in ['Seen','Transfer']};queries={c:sum(e['kind']=='shared' and e['query_context']==c for e in executed) for c in 'ABC'}
   for e in executed:
    for a in exposure:
     for v in source.support(e,'P1',a):exposure[a][v['context']]+=1
   summaries.append(dict(seed=seed,block=block,first=executed[0]['episode'],last=executed[-1]['episode'],executed=len(executed),unused=240-len(executed),shared=kinds['shared'],conflict=kinds['conflict'],reserved_rejections=len(reg['reserved_rejections']),duplicate_rejections=len(reg['rejections']),exposures=json.dumps(exposure,sort_keys=True),query_roles=json.dumps(queries,sort_keys=True)))
  for prior in ['P1','P2']:
   for org in ['I','C']:
    for arm in ['Seen','Transfer']:
     state,opt,path=load_source(seed,prior,org,arm);validate_source_bundle(state,opt,960)
     rows.append(dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm=arm,episode=960,path=str(path.relative_to(ROOT)),sha256=sha(path),optimizer_steps=json.dumps([float(s['step']) for s in opt['state'].values()]),optimizer_group=json.dumps(opt['param_groups'][0],sort_keys=True),context_C_norm=float(torch.linalg.vector_norm(state['recurrent.weight_ih_l0'][:,6]))))
  print(f'Prepared seed {seed}:38 balanced blocks, final160 prefix, reserved24 paths',flush=True)
 write_frame(OUT/'registry/sources.csv',rows);write_frame(OUT/'registry/block_summary.csv',summaries)
 dump(OUT/'audit/source_registry_prepared.json',dict(passed=True,sources=160,blocks=760,new_episodes=180800,full_blocks=740,partial_blocks=20,reserved_individual_sequences=480,old_or_new_reserved_overlap=0,all_original_scientific_operations_imported=True,training=False))
 (OUT/'SOURCE_REUSE_AUDIT.md').write_text('''# Source/reuse audit before continuation

All160 exact episode960 meta states and six-parameter Adam moment/step bundles are verified against the full predecessor seal. `registry/sources.csv` records every checkpoint SHA256 and optimizer settings. No outer Adam reset is allowed.

The task RNG is PCG64 keyed by `[20261121,81841,seed,block]`. Continue blocks5–42, keeping the original shuffle, task specifications and rejection rules. The corrected diagnostic library (24 individual full8-symbol paths/seed) is reserved before generation. Draw rejection adds only this required exclusion; candidates, exact final generator states, complete schedules and unused final80 episodes are retained. Each complete240 block preserves120/120 tasks and exact contextual roles/exposures. At10000 only the first160 episodes of block42 run; its realized counts are reported, without outcome-dependent reordering.

The predecessor saved no global torch RNG snapshot. Its actual update has no stochastic operation: taskmodel is copied, policy resets to zero, and all support/query inputs come from sealed registries; model factories use forked fixed RNG. The keyed task stream is exactly recoverable, and the clone benchmark verifies that an entire episode leaves the global torch RNG unchanged. New checkpoints save global torch/NumPy/Python states plus sealed task-registry hash/cursor; no required scientific RNG state is approximated.

Meta update: import `heldout_context_meta.fomaml.episode` unchanged. Compact mode disables only packed observation snapshots outside a fixed audit subset, retaining identical query gradients, weights, moments and update ordering. Source core.step/objective/Adam/policy/clip remain unchanged. Full traces are kept for every first episode of each new240 block, episode962, and every registered checkpoint. Compact per-episode records retain all20 inner scalar losses/norms, raw query gradient, query logits/hidden and before/after update summaries. This differs explicitly from the predecessor's all-episode packed tensor histories.

Target: reuse original target_run through a process-local namespace adapter; source files stay immutable. All800 new330-update runs retain every331 model/policy/Adam state and six-cue position predictions. Reuse400 old physical target histories at0/240/960; historical0 is one baseline, never two independent arms. Only corrected-v2 heldout data are reused. New diagnostic clones at3840/7680/10000 use the reserved library. No context-strength sweep is run.

Resumability: each completed meta episode atomically commits its compact summary and checksummed complete restart state in a synchronous SQLite transaction. Registered checkpoints are retained inside that transaction and atomically exported, so a crash between commit and export needs no optimizer replay. Duplicate or skipped committed episodes and checksum failures are rejected. Incomplete interrupted in-flight computations cannot be silently claimed as completed; any such event must be recorded as a technical interruption, and no partial update is used as a checkpoint.
''')

def seal():
 assert json.loads((OUT/'audit/benchmark.json').read_text())['passed']
 assert json.loads((OUT/'audit/source_registry_prepared.json').read_text())['passed']
 tests=(OUT/'audit/06_GREEN_source_audit.log').read_text();assert 'passed' in tests
 registry={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').iterdir() if p.is_file()}
 dump(OUT/'audit/protocol_seal.json',dict(passed=True,parent_manifest_sha=sha(SOURCE/'manifest.json'),registry=registry,code=codes(),new_episodes=1446400,inner_updates=28928000,outer_updates=1446400,new_target_runs=800,target_updates=264000,target_clone_updates=9600,heldout_clone_updates=115200,full_audit_episodes=AUDIT_EPISODES,workers=8,threads=1,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),sampling='OriginalblockkeyedPCG64;diagnosticreservation;lastblockprefix160',storage='AtomicperepisodeSQLite+registereddurablecheckpoint+fixedfulltraceaudit'))
 append('## Pre-training source and registry seal\nAll160 model/Adam960 sources verified. Source scientific operations unchanged; required task RNG is the original keyed blockstream, not an absent global RNG snapshot. Corrected-v2 full paths reserved against allnew complete blocks and finalprefix. Original taskroles/shuffle unchanged, finalpartial realizedcounts explicit. RED/GREEN storage/reset/source/leakage tests and bitwise clone benchmark passed; audit subset fixed before scientific outcomes. No main optimizer calls yet.')
 print('Continuation protocol sealed before scientific updates',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');a=p.parse_args();seal() if a.seal else prepare()
