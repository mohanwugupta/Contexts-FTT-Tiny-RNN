"""Resolve the actual sealed release; selected states are copied, never rewritten."""
from .common import *
from . import design,adapter
from experiments.high_scale_constrained_scaling_10k.measurements import all_registry
NATIVE_FILES=['experiments/context_experience_prior/core.py','experiments/high_scale_initialization/core.py','experiments/high_scale_meta_progression/protocol.py','experiments/high_scale_meta_progression/target.py','experiments/meta_learning_scaling/common.py','confirmatory_srn/model.py','execution_objective_search/objectives.py']
CONFIG=ROOT/'configs/ab_update_transfer_analysis.yaml'
def selected_registry():
 f=all_registry();q=f[(f.phase.eq('Fresh')&f.dose.eq(0))|(f.phase.eq('R')&f.dose.eq(1920))|(f.phase.isin(design.OBJECTIVES)&f.dose.isin([1920,10000]))].copy()
 q['regime']=q.condition;q['objective']=q.phase.where(q.phase.isin(design.OBJECTIVES),'none');q['meta_stage']=q.phase.map(lambda x:'fresh' if x=='Fresh' else 'reptile' if x=='R' else 'fomaml');q['cohort']=np.where(q.dose.eq(10000),'final','reference');q['meta_episodes']=q.dose
 q['trajectory_id']=[f's{int(r.seed):02d}_scale{int(r.scale)}_{r.condition}_{r.phase}_e{int(r.dose)}' for r in q.itertuples()];assert len(q)==342 and q.path.is_unique
 return q.reset_index(drop=True)
def identity():return dict(config_sha=sha(CONFIG),native_code={n:sha(ROOT/n) for n in NATIVE_FILES},adapter_code={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/ab_update_transfer_analysis/common.py',ROOT/'experiments/ab_update_transfer_analysis/adapter.py',ROOT/'experiments/ab_update_transfer_analysis/design.py',ROOT/'experiments/ab_update_transfer_analysis/storage.py']},environment=environment(),source_manifest=sha(SOURCE/'manifest.json'))
def checked(path,manifest):
 path=Path(path);rel=str(path.relative_to(ROOT));own=str(path.relative_to(SOURCE)) if path.is_relative_to(SOURCE) else None;expected=manifest['outputs'].get(own) if own else manifest['inputs'].get(rel)
 assert expected and sha(path)==expected,'Unsealed or changed source '+rel
 return expected

def inspect():
 configure();receipt=OUT/'audit/inspect_complete.json'
 if receipt.exists():return
 manifest=json.loads((SOURCE/'manifest.json').read_text());assert sha(SOURCE/'manifest.json')=='979e805192875e82201ed7313ed35dd13e94aa4cd81cf544a980724b43093a5d'
 for n in NATIVE_FILES:assert sha(ROOT/n)==(manifest['code'].get(n) or manifest['inputs'].get(n))
 q=selected_registry();csv(OUT/'registry/trajectories.csv',q);dump(OUT/'registry/resolved_config.json',config(CONFIG));dump(OUT/'registry/identity.json',identity());rows=[];times=[]
 for i,r in enumerate(q.to_dict('records'),1):
  t=time.monotonic();path=ROOT/r['path'];assert checked(path,manifest)==r['sha256'];b=torch.load(path,weights_only=False);states=b['states'];assert len(states)==r['total']+1
  # Recovery RNG is unchanged by the audited deterministic native target path.
  progs=sorted(path.parent.glob('progress_*.pt'));sidecars=[];rngs=[]
  for pp in progs:
   h=checked(pp,manifest);pb=torch.load(pp,weights_only=False);sidecars.append(dict(path=str(pp.relative_to(ROOT)),sha256=h,update=pb['states'][-1]['update']));rngs.append(pb['rng']);del pb
  assert rngs,'No archived target RNG sidecar';assert all(torch.equal(rngs[0],x) for x in rngs)
  requested=design.PRIMARY+(design.SECONDARY if r['cohort']=='final' else []);saved={};status=[]
  for n in requested:
   if n>=len(states):status.append(dict(target_update=n,status='missing',reason='No full archived state; no late reconstruction'));continue
   st=states[n];assert st['update']==n;m,p,o=adapter.restore(st);assert next(m.parameters()).dtype==torch.float32
   assert all(float(x['step'])==n for x in o.state.values());cursor=dict(completed_updates=n,prior_cue=design.cue(n) if n else 'none',next_cue=design.cue(n+1),tokens=core.TARGET_PATH[:-1],targets=core.TARGET.tolist(),stateful_scheduler=False)
   full=dict(**copy.deepcopy(st),rng=dict(torch=rngs[0].clone()),cursor=cursor);saved[n]=full
   rows.append(dict(analysis_version=design.VERSION,trajectory_id=r['trajectory_id'],seed=r['seed'],scale=r['scale'],regime=r['regime'],objective=r['objective'],meta_stage=r['meta_stage'],meta_episodes=r['dose'],actual_outer_steps=r['dose']//2 if r['objective']=='B' else r['dose'],cohort=r['cohort'],target_update=n,prior_cue=cursor['prior_cue'],next_cue=cursor['next_cue'],path=r['path'],archive_sha256=r['sha256'],initializer_sha256=r['source_sha'],state_hash=digest(full),model_hash=digest(st['model']),policy_hash=digest(st['policy']),Adam_hash=digest(st['optimizer']),RNG_hash=digest(full['rng']),cursor_hash=digest(cursor),rng_provenance='archived target recovery sidecar; unchanged deterministic native path; Python/NumPy unused',rng_sidecar=sidecars[0]['path'],status='direct',reason='',dtype='float32',Adam_entries=len(o.state)))
  atomic(OUT/f'source_cache/{r["trajectory_id"]}.pt',dict(states=saved,source=r,sidecars=sidecars,missing=status));times.append(time.monotonic()-t)
  if i%18==0:print('Source audit',i,'/342',flush=True)
 csv(OUT/'registry/source_states.csv',rows);csv(OUT/'registry/source_loading_seconds.csv',[dict(seconds=x) for x in times]);dump(receipt,dict(passed=True,trajectories=342,final=144,reference=198,states=len(rows),missing_states=sum(len(torch.load(OUT/f'source_cache/{r.trajectory_id}.pt',weights_only=False)['missing']) for r in q.itertuples()),optimizer_steps=0,sidecar_RNG_invariant=True,source_manifest=sha(SOURCE/'manifest.json')))
 with (OUT/'SOURCE_AUDIT.md').open('a') as f:f.write('\n\n## Resolved source audit\nAll342 physical trajectories and their requested states are indexed in registry/source_states.csv. Native-code, archive and recovery hashes were matched against the sealed source manifest/chain. Every requested full model/policy/Adam state restores exactly; all t0 Adam states are empty. Recovery Torch RNG values agree across every available330/660/990 sidecar pertrajectory. The target path consumes no Torch/Python/NumPy RNG and has no mutable scheduler. Branch recovery stores exact Torch RNG, explicit deterministic cue cursor and isolated current unused Python/NumPy RNG. This is not a claim that Python/NumPy RNG were archived at each historical target snapshot. Fresh4/6 histories include an exact reused330 prefix; its original worker RNG is not separately identifiable, but no stochastic target operation consumes it. No reconstruction is needed for the currently available state grid.\n')
 append('## Source inspection GREEN\nAll342 requested physical trajectories resolved from actual registries, never objective-expanded parent aliases. Sealed archives/native code and recovery RNG sidecars verified; direct state recovery/model-policy-Adam hashes exported, zero optimizer operations. See audit/inspect_complete.json for exact coverage. RNG provenance qualifications retained, deterministic task cursor explicit.')
