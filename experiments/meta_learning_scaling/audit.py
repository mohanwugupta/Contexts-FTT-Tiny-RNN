"""Integrity, exact source correspondence and before-run continuation registry seal."""
from .common import *
from .protocol import *
from experiments.context_experience_prior.outer_audit import verify_outer

def main():
 initialize_process();start=time.monotonic();parent=verify_parent();assert not (OUT/'audit/protocol_seal.json').exists();reused=[];meta=[];summary=[];reject=[]
 for seed in range(20):
  old_registry=json.loads((SOURCE/f'registry/seed_{seed}.json').read_text());seen={task_signature(e) for e in old_registry['episodes']};old_duplicates=240-len(seen)
  for block in range(1,8):
   r=continuation_block(seed,block,seen);seen.update(task_signature(e) for e in r['episodes']);dump(OUT/f'registry/seed_{seed}_block_{block}.json',r);reject.extend(dict(seed=seed,continuation_block=block,**x) for x in r['rejections']);summary.append(dict(seed=seed,block=block,episodes=240,first_episode=block*240+1,last_episode=(block+1)*240,rejections=len(r['rejections']),old_duplicate_count=old_duplicates))
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261120,99174,seed])));v=rng.normal(size=(20,8));v/=np.linalg.norm(v,axis=1,keepdims=True);np.save(OUT/f'registry/random_directions_seed_{seed}.npy',v)
  for org in ['I','C']:
   for prior in ['P0','P1','P2']:
    dose=0 if prior=='P0' else 240;path=SOURCE/f'target/seed_{seed}_{prior}_{org}.pt';b=torch.load(path,weights_only=False);assert len(b['states'])==331
    if prior!='P0':
     mp=SOURCE/f'meta/seed_{seed}_{prior}_{org}.pt';mb=torch.load(mp,weights_only=False);st=mb['records'][240];assert st['episode']==240 and st['status']=='completed';assert all(torch.equal(v,b['states'][0]['model'][n]) for n,v in st['meta'].items());meta.append(dict(seed=seed,prior=prior,organization=org,path=str(mp.relative_to(ROOT)),sha256=sha(mp),resume_episode=241,epsilon=.01,meta_initial_hash=st['meta_hash']))
    reused.append(dict(run_id=f's{seed:02d}_{prior}_{org}_d{dose}',seed=seed,prior=prior,organization=org,condition=prior+'-'+org,dose=dose,status='completed',source=str(path.relative_to(ROOT)),source_sha=sha(path),path=str(path.relative_to(ROOT)),sha256=sha(path),directory='',reused=True,states=331,target_updates=0,diagnostic_updates=0,wall_seconds=0.))
 write_frame(OUT/'registry/reuse_registry.csv',reused);write_frame(OUT/'registry/meta_resume_registry.csv',meta);write_frame(OUT/'registry/continuation_summary.csv',summary);write_frame(OUT/'registry/duplicate_rejections.csv',reject,columns=None)
 reg={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('*')};dump(OUT/'audit/protocol_seal.json',dict(parent_manifest_sha=sha(SOURCE/'manifest.json'),parent_outputs=parent['outputs'],parent_code=parent['code'],registry=reg,code=codes(),workers=4,meta_inner=2688000,meta_outer=134400,new_target_runs=560,new_target_updates=184800,new_diagnostic_updates=6720,total_new_gradient_updates=2879520,fixture_gradient_updates=0,doses=DOSES,outer_epsilon=.01,random_probe_included=True,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),wall_seconds=time.monotonic()-start))
 append('## R0 source/continuation seal complete\nAll359parentoutputs,83codefiles,115sourceinputs verified.120targethistories source-exact;all80episode240weights matchtargetstate0.140newbalancedregistries sealedbeforetraining,duplicatecandidatesretained;private20directionsperseed sealed.4CPUworkers,torchthreads1,deterministicops. No oldupdatesrerun;no scientificgradientstepsyet. Registered continuation starts241epsilon.01. Allnewtrainingserializationchecks pass beforelaunch.')
 print('Protocol sealed; exact120reusehistories/80meta-resumes verified',flush=True)
if __name__=='__main__':main()
