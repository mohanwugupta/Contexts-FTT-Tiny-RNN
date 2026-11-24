"""Resolve exact native parents, read-only controls, references and task banks."""
from .common import *
from . import banks,rules

def main(dry_run=False):
 configure();receipt=OUT/'registry/inspection.json'
 if receipt.exists():
  r=json.loads(receipt.read_text())
  for p,h in r['source_files'].items():assert sha(ROOT/p)==h,p
  print(json.dumps(dict(registered=True,pilots=r['pilot_seeds'],confirmation=r['confirmation_seeds'],ceilings=CAPS,physical_rows=len(r['runs']),optimizer_operations=0)));return
 manifests={p:json.loads((ROOT/p).read_text()) for p in ['results/high_scale_constrained_scaling_10k/v1/manifest.json','results/high_scale_meta_progression/v1/manifest.json','results/ab_update_transfer_analysis/v1/manifest.json']};expected={}
 for name,m in manifests.items():
  directory=Path(name).parent
  expected.update(m.get('inputs',{}));expected.update(m.get('code',{}));expected.update({str(directory/k):v for k,v in m['outputs'].items()})
 files={name:sha(ROOT/name) for name in manifests}
 def pin(p):
  p=Path(p);rel=str(p.relative_to(ROOT));h=sha(p)
  if rel in expected:assert h==expected[rel],rel
  else:raise ValueError('Source not pinned by releases '+rel)
  files[rel]=h;return h
 native=['experiments/context_experience_prior/core.py','experiments/context_experience_prior/registry.py','experiments/context_experience_prior/meta.py','experiments/high_scale_initialization/core.py','experiments/high_scale_meta_progression/protocol.py','experiments/high_scale_meta_progression/target.py','experiments/meta_learning_scaling/common.py','experiments/recurrent_learning_transfer/measures.py','experiments/linear_scale_sweep/analysis.py','confirmatory_srn/model.py','confirmatory_srn/config.py','execution_objective_search/objectives.py','experiments/ab_update_transfer_analysis/adapter.py']
 for n in native:pin(ROOT/n)
 registry=PREVIOUS/'registry/trajectories.csv';pin(registry);f=pd.read_csv(registry);assert f.seed.drop_duplicates().tolist()==SEEDS
 rtargets=pd.read_csv(PROGRESSION/'registry/reptile_targets.csv');pin(PROGRESSION/'registry/reptile_targets.csv');up=SOURCE/'registry/U_checkpoints.csv';pin(up);parents=pd.read_csv(up)
 runs=[];controls={};refrows=[]
 for seed in SEEDS:
  taskfile=PROGRESSION/f'registry/R_s{seed:02}.json';pin(taskfile);bank=banks.make(seed);put(OUT/f'registry/tasks_s{seed:02}.json',bank)
  controls[seed]={}
  for label,phase,dose in [('F00','Fresh',0),('A00','R',1920),('B00','R',240)]:
   row=(f[(f.seed==seed)&(f.scale==5)&(f.phase==phase)&(f.dose==dose)&(f.regime.isin(['U','Shared']))].iloc[0].to_dict() if label!='B00' else rtargets[(rtargets.seed==seed)&(rtargets.scale==5)&(rtargets.phase=='R')&(rtargets.dose==240)].iloc[0].to_dict());path=ROOT/row['path'];pin(path);directory=path.parent
   evaluations=directory/'evaluations.csv';training=directory/'training.csv';position=directory/'position_predictions.csv.gz'
   for p in [evaluations,training,position]:pin(p)
   initializer=PROGRESSION/(f'initial/s{seed:02}_scale5.pt' if label=='F00' else f'meta/s{seed:02}_scale5/R_e{dose}.pt');pin(initializer);b=torch.load(initializer,weights_only=False)
   if label!='F00':
    matches=parents[(parents.seed==seed)&(parents.scale==5)&(parents.phase=='R')&(parents.dose==dose)];assert len(matches)==1 and matches.iloc[0].sha256==sha(initializer);assert b['arm']=='R' and b['episode']==dose and b['optimizer'] is None and b['registry_sha']==sha(taskfile)
   controls[seed][label]=dict(initializer=str(initializer.relative_to(ROOT)),initializer_sha=sha(initializer),states=str(path.relative_to(ROOT)),evaluations=str(evaluations.relative_to(ROOT)),training=str(training.relative_to(ROOT)),positions=str(position.relative_to(ROOT)),meta_episodes=dose,meta_inner_steps=dose*20,source_sha=row['source_sha'],state_sha=sha(path))
   assert row['source_sha']==sha(initializer),'Target initializer lineage mismatch'
   if label=='F00':
    ef=pd.read_csv(evaluations,float_precision='round_trip');metrics=rules.summarize(ef[ef.cue.isin(PROBES)]);refrows.append(dict(seed=seed,**metrics,source=str(evaluations.relative_to(ROOT)),sha256=sha(evaluations)))
   # Only pilot controls are materialized; reserved controls remain read-only sources.
   role='reference' if label=='F00' else ('pilot' if seed<2 else 'reserved_control')
   runs.append(dict(id=f'{label}_s{seed}',candidate=label,seed=seed,role=role,stage='archived',status='available_reuse',source=str(path.relative_to(ROOT))))
  for candidate in [f'A{i:02}' for i in range(1,8)]+[f'B{i:02}' for i in range(1,7)]:runs.append(dict(id=f'{candidate}_s{seed}',candidate=candidate,seed=seed,role='pilot' if seed<2 else 'reserved',stage='checkpoint' if candidate[0]=='A' else 'pretraining',status='registered_pilot' if seed<2 else 'not_selected',source=controls[seed]['A00' if candidate[0]=='A' else 'F00']['initializer']))
 ref={k:float(np.median([r[k] for r in refrows if np.isfinite(r[k])])) if any(np.isfinite(r[k]) for r in refrows) else None for k in ['Q_A','Q_B','E_normalized']};ref['resolved']=all(v is not None and v>0 for v in ref.values());ref['values']=refrows;ref['retention']=.5;ref['rules_config_sha']=sha(ROOT/'configs/joint_phenotype_hypothesis_screen.yaml');put(OUT/'reference_thresholds.json',ref)
 put(OUT/'registry/control_sources.json',controls);csv(OUT/'candidate_registry_initial.csv',runs);put(OUT/'registry/resolved_config.json',json.loads((ROOT/'configs/joint_phenotype_hypothesis_screen.yaml').read_text()))
 precision=SOURCE/'tables/strict_precision_exceptions.csv';pin(precision);csv(OUT/'registry/inherited_precision_exceptions.csv',pd.read_csv(precision))
 r=dict(passed=True,source_files=files,source_releases={name:sha(ROOT/name) for name in manifests},seed_order=SEEDS,pilot_seeds=[0,1],confirmation_seeds=[2,3,4,5],runs=runs,ceilings=CAPS,bank_hashes={str(p.relative_to(OUT)):sha(p) for p in (OUT/'registry').glob('tasks_*.json')},reference_sha=sha(OUT/'reference_thresholds.json'),task_exclusions='native allowed() full/cyclic/reverse and seven-window target exclusions; validation full paths additionally disjoint from240meta and other banks',native_functions=native,git_commit='ef43aedaad2009cd168801faa0cfa7150c271d88')
 put(receipt,r)
 with (OUT/'SOURCE_AUDIT.md').open('a') as h:h.write('\n\nResolved: six exact U/Reptile1920 parents, six exact240 parents and six Fresh histories, all source/checkpoint/function hashes match sealed release registries. First240 P2 tasks and nativeepsilon decay match archived B00, so B00 initialization AND330 target histories can be reused exactly; do not spend duplicate meta/target updates. Seeds0/1 pilot,2–5 reserved. All task/augmentation/width/calibration/background banks and Fresh reference values frozen before candidate outcomes. Native strict exceptions copied, not repaired. See registry/inspection.json and control_sources.json for exact hashes, lineage, and exclusions.\n')
 append('## Inspect GREEN, before candidate outcomes\nAll source-native functions/lineages resolved. F00 six-seed reference medians, complete240-task/augmentation and independent validation/calibration/width banks sealed; A00 and B00 exact archived target histories available, no reconstruction. B00 is an exact first240 ordinary native prefix with0.1→0.01 outer schedule, reusable rather than new training. No new model execution or optimizer call in inspection.')
 print(json.dumps(dict(registered=True,reference={k:ref[k] for k in ['Q_A','Q_B','E_normalized','resolved']},controls='A00/B00/F00 archived',ceilings=CAPS,physical_rows=len(runs),optimizer_operations=0)),flush=True)
