from .common import *
from .protocol import scaled
from experiments.high_scale_constrained_meta.initialization import initialize,groups,rms
from experiments.high_scale_constrained_meta.runtime import scale_metrics

def build(seeds,sources,scales):
 rows=[]
 for seed in seeds:
  for src in sources:
   p=PARENT/f'initial/s{seed:02d}_High.pt' if src=='Fresh' else PARENT/f'meta/s{seed:02d}_{src}/fomaml_e1920.pt'
   if seed>=6:
    assert src=='Fresh';base=initialize(seed)[0].state_dict();origin='original deterministic High generator, new seed';origin_sha=None
   else:base=torch.load(p,weights_only=False)['model'];origin=str(p.relative_to(ROOT));origin_sha=sha(p)
   for s in scales:
    d=scaled(base,s);path=OUT/f'initial/s{seed:02d}_{src}_{s}.pt';h=atomic_save(path,dict(seed=seed,source=src,nominal_scale=s,alpha=s/2,model=d,source_path=origin,source_sha=origin_sha,code=codes()));m=source_model(d);q=scale_metrics(m);rows.append(dict(seed=seed,source=src,scale=s,alpha=s/2,path=str(path.relative_to(ROOT)),sha256=h,source_path=origin,source_sha=origin_sha,recurrent_spectral_norm=q['recurrent_spectral_norm'],effective_contrast=float(q['effective_contrast'].mean()),centered_contrast=float(q['centered_contrast'].mean()),**{k:rms(v) for k,v in groups(m).items()}))
 return rows

def main():
 m=json.loads((PARENT/'manifest.json').read_text());core.hashes(PARENT,m['outputs']);core.hashes(ROOT,m['code']);core.hashes(ROOT,m['inputs']);refs={str((PARENT/k).relative_to(ROOT)):v for k,v in m['outputs'].items()};refs.update(m['code']);refs.update(m['inputs']);refs[str((PARENT/'manifest.json').relative_to(ROOT))]=sha(PARENT/'manifest.json');dump(OUT/'registry/parent_hashes.json',refs);rows=build(range(6),['Fresh','U','P'],SCALES);write_frame(OUT/'registry/screen_initializations.csv',rows)
 pre=pd.read_csv(PARENT/'registry/preflight_targets.csv');old=pd.read_csv(PARENT/'registry/targets.csv');reuse=[]
 for seed in range(6):
  for src in ['Fresh','U','P']:
   r=(pre[pre.seed.eq(seed)&pre.arm.eq('High')] if src=='Fresh' else old[old.seed.eq(seed)&old.arm.eq(src+'-F')&old.dose.eq(1920)]).iloc[0].to_dict();path=ROOT/r['path'];b=torch.load(path,weights_only=False);new=torch.load(OUT/f'initial/s{seed:02d}_{src}_2.pt',weights_only=False)['model'];assert all(torch.equal(v,b['states'][0]['model'][k]) for k,v in new.items());r.update(arm=src,dose=2,source=src,scale=2,reused=True,directory=str((PARENT/r['directory']).relative_to(ROOT)));reuse.append(r)
 write_frame(OUT/'registry/reused_scale2.csv',reuse);dump(OUT/'audit/source_audit.json',dict(passed=True,all_six_paired_sources_available=True,scale2_bitwise_identity=True,exact_histories_reused=18,parameters_scaled=['token','A','B','C','output'],no_meta_training=True));dump(OUT/'audit/screen_initial_seal.json',dict(passed=True,files={r['path']:r['sha256'] for r in rows},preregistration_sha=sha(OUT/'registry/PREREGISTRATION.md')));append('## Source audit complete\nFull parent/source chain verified.144 initial copies prepared; eighteen scale2 initial states bitwise match archived target starts and histories reused, never retrained. All dimensions/recurrence/biases unchanged; actual RMS and class-centered contrasts archived. No meta-training authorized or executed.')
if __name__=='__main__':main()
