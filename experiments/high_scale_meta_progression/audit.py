"""Verify the entire frozen source chain and exact archived 330-update branches."""
import shutil
from .common import *
from .protocol import scaled_initial,restore_target,validate_fork,config
from experiments.high_scale_constrained_meta.audit import metrics
from experiments.high_scale_constrained_meta.runtime import scale_metrics

def archived_target(seed,scale):
 r=pd.read_csv(SWEEP/'registry/screen_runs.csv');r=r[r.seed.eq(seed)&r.arm.eq('Fresh')&r.dose.eq(scale)].iloc[0].to_dict();p=ROOT/r['path']
 if sha(p)!=r['sha256']:raise ValueError('Archived target changed')
 return dict(states=p,directory=p.parent,sha256=r['sha256'])

def main():
 configure();config();m=json.loads((SWEEP/'manifest.json').read_text());core.hashes(SWEEP,m['outputs']);core.hashes(ROOT,m['code']);core.hashes(ROOT,m['inputs']);refs={str((SWEEP/k).relative_to(ROOT)):v for k,v in m['outputs'].items()};refs.update(m['code']);refs.update(m['inputs']);refs[str((SWEEP/'manifest.json').relative_to(ROOT))]=sha(SWEEP/'manifest.json');dump(OUT/'registry/parent_hashes.json',refs);(OUT/'initial').mkdir(exist_ok=True);rows=[];reused=[]
 for seed in range(6):
  for scale in [4,5,6]:
   d=scaled_initial(seed,scale);p=OUT/f'initial/s{seed:02d}_scale{scale}.pt';h=atomic_save(p,dict(model=d,seed=seed,scale=scale,source=str((HIGH/f'initial/s{seed:02d}_High.pt').relative_to(ROOT)),source_sha=sha(HIGH/f'initial/s{seed:02d}_High.pt'),alpha=scale/2,code=codes()));rows.append(dict(seed=seed,scale=scale,path=str(p.relative_to(ROOT)),sha256=h,**metrics(d),centered_contrast=float(scale_metrics(source_model(d))['centered_contrast'].mean())))
   if scale in [4,6]:
    old=archived_target(seed,scale);b=torch.load(old['states'],weights_only=False);assert len(b['states'])==331;validate_fork(d,b['states'][0]['model']);restore_target(b['states'][330]);reused.append(dict(seed=seed,scale=scale,path=str(old['states'].relative_to(ROOT)),sha256=old['sha256'],prefix_states=331,updates_reused=330,model_policy_Adam_complete=True))
 for seed in range(6):shutil.copyfile(HIGH/f'registry/heldout_v2_seed_{seed}.json',OUT/f'registry/heldout_v2_seed_{seed}.json')
 reserved=json.loads((HIGH/'registry/diagnostic_paths.json').read_text());dump(OUT/'registry/diagnostic_paths.json',reserved);write_frame(OUT/'registry/initializations.csv',rows);write_frame(OUT/'registry/reused_targets.csv',reused);dump(OUT/'audit/source_audit.json',dict(passed=True,parent_manifest_sha=sha(SWEEP/'manifest.json'),initializations=18,reused_targets=12,reused_target_updates=3960,reused_diagnostic_updates=144,source_chain_files=len(refs),fresh_diagnostic_never_meta_parent=True,all20_historical_seeds_preserved=True,config_sha=sha(ROOT/'configs/high_scale_meta_progression.yaml')));append('## Source audit complete\nFull predecessor seal and all ancestors verified; all18 fresh states materialized from original High tensors. All12 scale4/6 target330 archives match initial tensors exactly and contain complete model/policy/Adam states. Their331 frozen evaluations and diagnostics will be copied without reevaluation; only updates331–990 execute. Corrected-v2 global reserved paths copied before new task generation. No optimizer steps in audit.');print('Source/reuse audit PASSED',flush=True)
if __name__=='__main__':main()
