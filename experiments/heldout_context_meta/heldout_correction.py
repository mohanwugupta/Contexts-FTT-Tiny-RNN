"""Versioned correction: unseen task pairs are insufficient for fresh sequences."""
import shutil
from .common import *
from .protocol import schedule,signature
from experiments.context_experience_prior.registry import draw

def fresh_collection(seed,seen_paths):
 seen=set(seen_paths);rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261121,81842,seed])));held=[];reject=[]
 for pair in ['AB','AC','BC']:
  for order in [0,1]:
   for mapping in [0,1]:
    while True:
     p=draw(rng);paths=[p,draw(rng,p[:3],p[3])];ep=dict(episode=len(held)+1,kind='conflict',pair=pair,order=order,mapping=mapping,pattern=0,paths=paths)
     if len({tuple(p) for p in paths})==2 and not any(tuple(p) in seen for p in paths):seen.update(tuple(p) for p in paths);break
     reject.append(dict(**ep,reason='individual_sequence_seen_or_duplicate'))
    ss=schedule(pair,order,mapping,0,'conflict','P2');ep['support']={a:{p:ss for p in ['P1','P2']} for a in ['Transfer','Seen']};held.append(ep)
 return dict(seed=seed,episodes=held,stable_identifiable_for_all_priors=True,individual_sequence_exclusion=True,rejections=reject)

def main():
 invalid=OUT/'audit/invalid_heldout_v1';invalid.mkdir();files=[]
 for p in (OUT/'tables').glob('heldout*.csv'):shutil.copyfile(p,invalid/p.name);files.append(str(p.relative_to(OUT)))
 for name in ['FINAL_HELDOUT_CONTEXT_META_REPORT.md','RESEARCHER_HANDOFF.md']:
  p=OUT/name
  if p.exists():shutil.copyfile(p,invalid/name)
 for p in (OUT/'figures').glob('mechanism_heldout_conflicts.*'):shutil.copyfile(p,invalid/p.name)
 newhash={}
 for seed in range(20):
  regs=[json.loads((OLD/f'registry/seed_{seed}.json').read_text())]+[json.loads((SOURCE/f'registry/seed_{seed}_block_{b}.json').read_text()) for b in range(1,8)]+[json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text()) for b in range(1,5)];seen={tuple(p) for r in regs for e in r['episodes'] for p in e['paths']};r=fresh_collection(seed,seen);v=[tuple(p) for e in r['episodes'] for p in e['paths']];assert len(v)==24 and len(set(v))==24 and not seen.intersection(v);p=OUT/f'registry/heldout_v2_seed_{seed}.json';dump(p,r);newhash[str(p.relative_to(OUT))]=sha(p)
 dump(OUT/'audit/heldout_correction_seal.json',dict(reason='46individualsequencesseen;oldpair-onlyexclusioninsufficient',invalid_initial_updates=96000,corrected_planned_updates=96000,extra_technical_cost=96000,actual_total_optimizer_steps=3527040,registry=newhash,selection=False,meta_and_targets_unchanged=True,invalid_tables_preserved=files))
 # Invalidate firstsummary exports only after byte-identical copies; rawv1 remains untouched.
 for name in ['heldout_seed','heldout_summary','heldout_block_seed']:(OUT/'tables'/f'{name}.csv').unlink()
 print('Fresh individual-sequence v2 registry sealed; v1 diagnostic preserved as invalid',flush=True)
if __name__=='__main__':main()
