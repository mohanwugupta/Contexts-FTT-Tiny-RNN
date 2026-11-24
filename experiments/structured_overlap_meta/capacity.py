"""Mandatory fixed pilot, with a preregistered learnability stop gate."""
import argparse,concurrent.futures,multiprocessing
from .common import *
from .batch import forward_pair,step_batch

def criteria(ce,correct,mask):
 ce=np.asarray(ce,float);correct=np.asarray(correct,bool);mask=np.asarray(mask,bool);assert ce.shape==correct.shape==mask.shape==(4,7)
 cx=float(ce[:2].mean());cy=float(ce[2:].mean());ax=float(correct[:2].mean());ay=float(correct[2:].mean());strong=bool(np.isfinite(ce).all() and cx<=.5 and cy<=.5 and ax>=6/7 and ay>=6/7);critical=bool(correct[mask].all());per_trial=bool(((ce.mean(1)<=.5)&(correct.mean(1)>=6/7)).all())
 return dict(CE_X=cx,CE_Y=cy,accuracy_X=ax,accuracy_Y=ay,strong_contexts=strong,all_critical_correct=critical,critical_accuracy=float(correct[mask].mean()) if mask.any() else np.nan,critical_ce=float(ce[mask].mean()) if mask.any() else np.nan,per_subtrial_strong=per_trial,success=bool(strong and critical),strict_subtrial_success=bool(per_trial and critical))

def decision(f):
 expected={(s,k,i) for s in PILOT_SEEDS for k in range(3) for i in range(12)};actual={(int(r.seed),int(r.n_exceptions),int(r.task)%12) for r in f.itertuples()};complete=actual==expected and len(f)==72;two=f[f.n_exceptions.eq(2)];n=len(two);passed=int(two.reached.astype(bool).sum());numerical=bool(f.status.eq('completed').all());return dict(passed=bool(complete and n==24 and passed>=20 and numerical),cases=n,successes=passed,required=20,rate=passed/n if n else None,threshold=.8,initializations=PILOT_SEEDS,complete_library=complete,numerical_integrity=numerical,capacity_unit='task-initialization case; not independent network',meta_training_allowed=bool(complete and n==24 and passed>=20 and numerical))

def observe(m,e,k):
 scores=[];contexts=[];positions=[]
 for role in ['X','Y']:
  with torch.no_grad():z,h,y=forward_pair(m,e['paths'][role],e['roles'][role])
  for j in range(2):
   o=core.score(z[j].numpy().astype(float),y[j].numpy());scores.append(o);contexts.append(dict(update=k,role=role,cue=e['roles'][role],subtrial=j,ce=float(o['ce'].mean()),accuracy=float(o['accuracy'].mean())))
   for t in range(7):positions.append(dict(update=k,role=role,cue=e['roles'][role],subtrial=j,position=t+1,token=e['paths'][role][j][t],target=int(y[j,t]),ce=o['ce'][t],correct=bool(o['accuracy'][t]),margin=o['margin'][t],critical=bool(e['critical_mask'][j+(2 if role=='Y' else 0)][t]),shared_eligible=bool(e['novel_mask'][j][t]),**{f'logit_{v}':float(z[j,t,v]) for v in range(4)},**{f'probability_{v}':o['probabilities'][t,v] for v in range(4)},**{f'hidden_{v}':float(h[j,t,v]) for v in range(8)}))
 r=criteria(np.stack([s['ce'] for s in scores]),np.stack([s['accuracy'] for s in scores]),e['critical_mask']);return dict(update=k,status='completed',**r),contexts,positions

def run(job):
 seed,task=job;initialize_process();torch.set_num_threads(json.loads((OUT/'audit/benchmark.json').read_text())['selected']['threads']);reg=json.loads((OUT/'registry/pilot.json').read_text());e=reg['episodes'][task];dest=OUT/f'pilot/s{seed:02d}_{e["pilot_id"]}';dest.mkdir(parents=True);state,outer,path=source(seed);m=source_model(state);policy=core.policy();opt=core.optimizer(m,policy);assert not opt.state;start=time.monotonic();dump(dest/'started.json',dict(seed=seed,task=task,episode=e,source=str(path.relative_to(ROOT)),source_sha=sha(path),original_outer_step=960,registry_sha=sha(OUT/'registry/pilot.json'),code=codes(),time=time.time()));states=[];curves=[];contexts=[];positions=[];training=[];count=0;failed=None;first=None
 for k in range(331):
  if failed:
   states.append(dict(update=k,status=failed,model=None,policy=None,optimizer=None));curves.append(dict(update=k,status=failed,success=False,CE_X=np.nan,CE_Y=np.nan,accuracy_X=np.nan,accuracy_Y=np.nan,critical_accuracy=np.nan,critical_ce=np.nan));continue
  states.append(dict(update=k,status='completed',model=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(policy.state_dict()),optimizer=copy.deepcopy(opt.state_dict())));r,c,p=observe(m,e,k);curves.append(r);contexts.extend(c);positions.extend(p)
  if r['success'] and first is None:first=k
  if k<330:
   role='X' if k//10%2==0 else 'Y';q=step_batch(m,policy,opt,e['roles'][role],e['paths'][role]);count+=int(q['optimizer_step_executed']);training.append(dict(update=k+1,role=role,cue=e['roles'][role],**q));frozen=r['CE_X'] if role=='X' else r['CE_Y'];np.testing.assert_allclose(q['ce_before'],frozen,rtol=1e-5,atol=1e-4)
   if q['status']!='completed':failed=q['status'];atomic_save(dest/'failure_state.pt',dict(model=m.state_dict(),policy=policy.state_dict(),optimizer=opt.state_dict(),update=k+1,reason=failed))
 atomic_save(dest/'states.pt',dict(seed=seed,task=task,source=str(path.relative_to(ROOT)),source_sha=sha(path),episode=e,original_outer_optimizer=outer,outer_updates_executed=0,states=states,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
 for n,data in [('learning',curves),('subtrial_evaluations',contexts),('training',training)]:write_frame(dest/f'{n}.csv',data)
 write_frame(dest/'position_predictions.csv.gz',positions);last=curves[-1];row=dict(seed=seed,task=task,pilot_id=e['pilot_id'],n_exceptions=e['n_exceptions'],status=failed or 'completed',reached=first is not None,first_pass=first,final_success=bool(last['success']),final_CE_X=float(last['CE_X']) if np.isfinite(last['CE_X']) else None,final_CE_Y=float(last['CE_Y']) if np.isfinite(last['CE_Y']) else None,updates_executed=count,outer_updates=0,directory=str(dest.relative_to(OUT)),states_sha=sha(dest/'states.pt'),wall_seconds=time.monotonic()-start);dump(dest/'complete.json',row);return row

def main():
 assert json.loads((OUT/'audit/source_audit.json').read_text())['passed'] and json.loads((OUT/'audit/benchmark.json').read_text())['passed'];seal=json.loads((OUT/'audit/pilot_registry_seal.json').read_text());core.hashes(OUT,seal['files']);assert sha(ROOT/'experiments/structured_overlap_meta/generator.py')==seal['generator_sha'];jobs=[(s,t) for s in PILOT_SEEDS for t in range(36)];rows=[];start=time.monotonic();dump(OUT/'audit/pilot_started.json',dict(jobs=jobs,code=codes(),criterion=seal['required_passes'],time=time.time()))
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run,jobs):rows.append(r);print(f'Capacity cases {len(rows)}/72 complete',flush=True)
 write_frame(OUT/'tables/pilot_cases.csv',rows);f=pd.DataFrame(rows);d=decision(f);dump(OUT/'audit/capacity_decision.json',d);dump(OUT/'audit/pilot_complete.json',dict(passed=True,cases=len(rows),updates=int(f.updates_executed.sum()),outer_updates=0,wall_seconds=time.monotonic()-start,capacity_passed=d['passed']));append('## Capacity preflight complete\nAllfixed72task-initializationcasesretained,full330updateswherefinite,original20/24two-exceptiongateevaluatedwithoutretuning. Actualcapacitygate: '+str(d)+'. No scientific meta-training before this decision.')
 print(json.dumps(d),flush=True)
if __name__=='__main__':main()
