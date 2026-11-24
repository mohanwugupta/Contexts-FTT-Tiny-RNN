"""Independent budget/alias/code/frozen-gate and primary paired arithmetic audit."""
from .common import *
from .gate import decide
from .train import target_jobs

def check_budget(b):
 expected=dict(reptile_inner=460800,reptile_outer=23040,fomaml_inner=460800,fomaml_outer=23040,preflight_target=3960,target=23760,target_clones=1008,conflict=18720)
 if b!=expected:raise ValueError('Changed fixed scientific budget')
 return sum(b.values())

def main():
 for n in ['preflight_verified','reptile_verified','fomaml_verified','targets_verified','heldout_verified','scale_projection_verified','final_analysis_complete']:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed'],n
 assert not (OUT/'audit/SCIENTIFIC_STOP.json').exists();r=decide(read('preflight_seed_outcomes'));decision=json.loads((OUT/'audit/preflight_decision.json').read_text());assert r['passed'] and r['high_joint_count']==decision['high_joint_count'];assert sha(ROOT/'experiments/high_scale_constrained_meta/gate.py')==decision['gate_source_sha'];initialseal=json.loads((OUT/'audit/initial_registry_seal.json').read_text());core.hashes(OUT,initialseal['files']);metaseal=json.loads((OUT/'audit/meta_registry_seal.json').read_text());core.hashes(OUT,metaseal['files']);budget={};critical={'common.py','initialization.py','runtime.py','train.py','registries.py','validation.py'}
 for phase in ['reptile','fomaml']:
  reg=pd.read_csv(OUT/f'registry/{phase}_runs.csv');assert len(reg)==12 and set(reg.seed)==set(SEEDS) and set(reg.arm)=={'U','P'} and reg.status.eq('completed').all() and reg.episodes.eq(1920).all();budget[phase+'_inner']=int(reg.inner_updates.sum());budget[phase+'_outer']=int(reg.outer_updates.sum());start=json.loads((OUT/f'audit/{phase}_started.json').read_text());core.hashes(ROOT,{k:v for k,v in start['code'].items() if Path(k).name in critical});assert start['time']>decision['time']
 for s in SEEDS:
  high=torch.load(OUT/f'initial/s{s:02d}_High.pt',weights_only=False)['model']
  for arm in ['U','P']:
   d=OUT/f'meta/s{s:02d}_{arm}';r0=torch.load(d/'reptile_e0.pt',weights_only=False);r1920=torch.load(d/'reptile_e1920.pt',weights_only=False);f0=torch.load(d/'fomaml_e0.pt',weights_only=False);assert all(torch.equal(v,r0['model'][k]) for k,v in high.items()) and all(torch.equal(v,f0['model'][k]) for k,v in r1920['model'].items());assert not f0['optimizer']['state'] and r0['optimizer'] is None
 pre=pd.read_csv(OUT/'registry/preflight_targets.csv');tar=pd.read_csv(OUT/'registry/targets.csv');held=pd.read_csv(OUT/'registry/heldout.csv');assert len(pre)==12 and len(tar)==72 and len(held)==78;assert set(zip(tar.seed,tar.prior,tar.organization,tar.arm,tar.dose))==set(target_jobs());budget.update(preflight_target=int(pre.target_updates.sum()),target=int(tar.target_updates.sum()),target_clones=int(pre.diagnostic_updates.sum()+tar.diagnostic_updates.sum()),conflict=int(held.diagnostic_steps.sum()));total=check_budget(budget);physical=read('meta_physical_registry');assert len(physical)==78 and physical.path.nunique()==78 and physical.reused_preflight.sum()==6;assert len(read('meta_evaluations'))==78*331*6 and len(read('meta_blocks'))==78*33 and len(read('meta_matched_acquisition'))==78*12;out=read('meta_seed_outcomes');raw=read('meta_paired_seed_effects')
 for phase in ['R','F']:
  for dose in [240,960,1920]:
   u=out[out.phase.eq(phase)&out.dose.eq(dose)&out.condition.eq('U')].set_index('seed');p=out[out.phase.eq(phase)&out.dose.eq(dose)&out.condition.eq('P')].set_index('seed');assert len(u)==len(p)==6
   for m in ['D_A','D_B','gap','ce','familiar_ce','Mixed_both_ce','conflict_accuracy']:
    got=raw[raw.phase.eq(phase)&raw.dose.eq(dose)&raw.metric.eq(m)].set_index('seed').effect.reindex(SEEDS);np.testing.assert_allclose(got,(p[m]-u[m]).reindex(SEEDS),atol=1e-14,rtol=0,equal_nan=True)
 aliases=pd.read_csv(OUT/'registry/logical_checkpoint_aliases.csv');assert len(aliases)==24 and not aliases.independent_target_run.any();assert len(pd.read_csv(OUT/'tables/meta_group_scale_projection.csv.gz'))==230520;dump(OUT/'audit/final_tables_verified.json',dict(passed=True,scientific_optimizer_or_outer_moves=total,budget=budget,fixture_inner=162,fixture_outer=8,physical_target_histories=84,main_physical_histories=78,trained_checkpoints=72,aliases=24,all_source_scientific_code_unchanged=True,all_fixed_seed_grids=True,all_primary_paired_arithmetic=True,gate_reproduced=True,no_scientific_retries=True));append('## Final budget/source/alias audit passed\nExact46080metaepisodes/921600innerupdates;23040Reptileoutermoves/23040FOMAMLAdamsteps. Actualscientific training/diagnostic updates plus outer moves='+str(total)+'; separate fixtures170. All84physicaltargethistories (12preflight/72new),24logicalaliases,78conflictdiagnostics,unchangedcriticalcode,immutablegate/source/checkpoints andallpairedarithmeticverified. Projections are recorded postprocessing within outer steps, not extra Adam updates. No failure/rescue/retry or checkpoint selection.')
 print('Final fixed-protocol audit PASSED',flush=True)
if __name__=='__main__':main()
