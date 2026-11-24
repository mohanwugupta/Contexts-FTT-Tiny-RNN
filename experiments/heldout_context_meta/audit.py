"""Source integrity and matched training/diagnostic registry seal before FOMAML."""
from .common import *
from .protocol import *
from experiments.context_experience_prior.registry import make_registry as original_registry

def main():
 initialize_process();start=time.monotonic();parent=parent_verify();assert not (OUT/'audit/protocol_seal.json').exists();rows=[];summary=[];allreject=[]
 for seed in range(20):
  oldreg=json.loads((OLD/f'registry/seed_{seed}.json').read_text());seen={signature(e) for e in oldreg['episodes']}
  for b in range(1,8):seen.update(signature(e) for e in json.loads((SOURCE/f'registry/seed_{seed}_block_{b}.json').read_text())['episodes'])
  for b in range(1,5):
   r=make_block(seed,b,seen);seen.update(signature(e) for e in r['episodes']);dump(OUT/f'registry/seed_{seed}_b{b}.json',r);allreject.extend(dict(seed=seed,block=b,**e) for e in r['rejections']);summary.append(dict(seed=seed,block=b,episodes=240,shared=120,conflict=120,rejections=len(r['rejections']),inner_exposures_per_cue=1600,query_shared_per_cue=40))
  rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261121,81842,seed])));held=[]
  for pair in ['AB','AC','BC']:
   for order in [0,1]:
    for mapping in [0,1]:
     while True:
      p=draw(rng);ep=dict(episode=len(held)+1,kind='conflict',pair=pair,order=order,mapping=mapping,pattern=0,paths=[p,draw(rng,p[:3],p[3])]);sig=signature(ep)
      if sig not in seen:seen.add(sig);break
      allreject.append(dict(seed=seed,block='heldout',**ep,reason='duplicate_library_within_seed'))
     ss=schedule(pair,order,mapping,0,'conflict','P2');ep['support']={a:{p:ss for p in ['P1','P2']} for a in ['Transfer','Seen']};held.append(ep)
  dump(OUT/f'registry/heldout_seed_{seed}.json',dict(seed=seed,episodes=held,stable_identifiable_for_all_priors=True))
  # Reuse the already sealed sensitivity directions exactly.
  v=np.load(SOURCE/f'registry/random_directions_seed_{seed}.npy');np.save(OUT/f'registry/random_directions_seed_{seed}.npy',v)
  for prior in ['P1','P2']:
   for org in ['I','C']:
    state,path=initial(seed,prior,org);tp=SOURCE/f'target/s{seed:02d}_{prior}_{org}_d1920/states.pt';tb=torch.load(tp,weights_only=False);assert len(tb['states'])==331 and all(torch.equal(w,tb['states'][0]['model'][n]) for n,w in state.items());rows.append(dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm='Historical',dose=0,run_id=f's{seed:02d}_{prior}_{org}_Historical_d0',meta_path=str(path.relative_to(ROOT)),meta_sha=sha(path),path=str(tp.relative_to(ROOT)),sha256=sha(tp),directory=str(tp.parent.relative_to(ROOT)),status='completed',reused=True,states=331))
 write_frame(OUT/'registry/historical.csv',rows);write_frame(OUT/'registry/task_summary.csv',summary);write_frame(OUT/'registry/rejections.csv',allreject)
 reg={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('*')};dump(OUT/'audit/protocol_seal.json',dict(parent_manifest_sha=sha(SOURCE/'manifest.json'),registry=reg,code=codes(),meta_runs=160,meta_episodes=153600,meta_inner_updates=3072000,outer_updates=153600,target_runs=320,target_updates=105600,target_diagnostic_updates=3840,heldout_diagnostic_updates=96000,total_new_optimizer_steps=3431040,fixture_inner_updates=120,fixture_outer_updates=6,workers=4,threads=1,python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),wall_seconds=time.monotonic()-start));append('## Source/registryprotocolseal\nWholepredecessormanifest verified;80source1920metastates matchhistoricaltargetstate0. Bothnewarmssamecopiedweights;80newtrainingblocksand20heldoutcollectionssealedwithstrictlibraryexclusions/rolebalance. Originalobjective/Adam/clip/sourcehelpersunchanged. FOMAMLindependentreferencepasses(afterretainedbatchedprojectionfixturecorrection);fixture120inner+6outerseparate. No scientificmetatrainingyet. FourdeterministicCPUworkers,completepackedinner/outerhistories.')
 print('Source, role-balance, duplicate and gradient gates passed; protocol sealed',flush=True)
if __name__=='__main__':main()
