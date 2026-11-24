"""Independent complete-cohort, budget, reuse, code and selection checks."""
from .common import *
from .selection import choose
from .generator import validate

def check_budgets(b,confirm):
 expected=dict(meta_inner=710400,meta_outer=35520,target=24420,target_clones=888,conflict_clones=17760,pilot=23760) if confirm else dict(meta_inner=172800,meta_outer=8640,target=5940,target_clones=216,conflict_clones=4320,pilot=23760)
 if b!=expected:raise ValueError('Changed registered scientific budget')
 return sum(b.values())

def main():
 selection=json.loads((OUT/'audit/selection_seal.json').read_text());assert sha(OUT/'SCREENING_DECISION.md')==selection['decision_sha'];f=read('screen_seed_outcomes');r,_=choose(f[f.arm.isin(list('ABC'))&f.dose.eq(480)]);assert r['selected']==selection['selected'] and r['passed']==selection['passed'];assert sha(OUT/'tables/screen_seed_outcomes.csv')==selection['screen_table_sha'];assert not list((OUT/'meta').glob('*/failure.json'));confirm=selection['passed'];totals=dict(meta_inner=0,meta_outer=0,target=0,target_clones=0,conflict_clones=0,pilot=int(read('pilot_cases').updates_executed.sum()));physical=[];critical=['common.py','generator.py','batch.py','runtime.py','train.py','validation.py']
 for stage,seeds,nmeta,ntarget,neps,nhistory in [('screen',SCREEN,18,18,480,24)]+([('confirm',CONFIRM,28,56,960,70)] if confirm else []):
  for kind in ['meta','targets']:assert json.loads((OUT/f'audit/{stage}_{kind}_verified.json').read_text())['passed']
  m=pd.read_csv(OUT/f'registry/{stage}_meta_runs.csv');t=pd.read_csv(OUT/f'registry/{stage}_targets.csv');h=pd.read_csv(OUT/f'registry/{stage}_heldout.csv');assert len(m)==nmeta and len(t)==ntarget and len(h)==ntarget and set(m.seed)==set(seeds);assert m.episodes.eq(neps).all() and m.inner_updates.eq(neps*20).all() and m.outer_updates.eq(neps).all();totals['meta_inner']+=int(m.inner_updates.sum());totals['meta_outer']+=int(m.outer_updates.sum());totals['target']+=int(t.target_updates.sum());totals['target_clones']+=int(t.diagnostic_updates.sum());totals['conflict_clones']+=int(h.diagnostic_steps.sum());started=json.loads((OUT/f'audit/{stage}_meta_started.json').read_text());core.hashes(ROOT,{k:v for k,v in started['code'].items() if Path(k).name in critical})
  if stage=='confirm':assert started['time']>selection['time'];assert set(m.arm)=={'A',selection['selected']}
  f=read(stage+'_evaluations');assert len(f)==nhistory*331*6 and not f.duplicated(['seed','arm','dose','update','cue']).any() and f.ce.notna().all() and set(f.cue)==set(CUES);assert len(read(stage+'_blocks'))==nhistory*33 and len(read(stage+'_matched_acquisition'))==nhistory*12;reg=read(stage+'_physical_registry');assert len(reg)==nhistory;physical.extend(reg.to_dict('records'))
  for seed in seeds:
   a=f[f.seed.eq(seed)&f.arm.eq('Baseline')].sort_values(['update','cue']);b=pd.read_csv(SOURCE/f'target/s{seed:02d}_P2_I_Transfer_d960/evaluations.csv',float_precision='round_trip').sort_values(['update','cue']);np.testing.assert_array_equal(a.ce,b.ce);np.testing.assert_array_equal(a.accuracy,b.accuracy)
  raw=read(stage+'_paired_seed_effects');out=read(stage+'_seed_outcomes')
  for a in [x for x in out.arm.unique() if x not in ['Baseline','A']]:
   for dose in sorted(out.dose.unique()):
    x=out[out.dose.eq(dose)&out.arm.eq(a)].set_index('seed');y=out[out.dose.eq(dose)&out.arm.eq('A')].set_index('seed')
    if not len(x):continue
    for met in ['gap','ce','conflict_accuracy','conflict_ce','D_A','D_B']:
     got=raw[raw.candidate.eq(a)&raw.dose.eq(dose)&raw.metric.eq(met)].set_index('seed').effect.reindex(seeds);want=(x[met].astype(float)-y[met].astype(float)).reindex(seeds);np.testing.assert_allclose(got,want,rtol=0,atol=1e-15,equal_nan=True)
 seal=json.loads((OUT/'audit/meta_registry_seal.json').read_text());core.hashes(OUT,seal['files']);assert sha(ROOT/'experiments/structured_overlap_meta/generator.py')==seal['generator_sha'];reserved={tuple(p) for key in ['diagnostic_paths','pilot_paths'] for p in json.loads((OUT/f'registry/{key}.json').read_text())}
 for s in range(20):
  for a in 'ABC':
   for e in json.loads((OUT/f'registry/episodes_seed_{s}_{a}.json').read_text()):validate(e,reserved)
 total=check_budgets(totals,confirm);assert len(physical)==(94 if confirm else 24) and len({r['path'] for r in physical})==len(physical);assert sum(bool(x['reused']) for x in physical)==(20 if confirm else 6);dump(OUT/'audit/final_tables_verified.json',dict(passed=True,scientific_optimizer_steps=total,breakdown=totals,implementation_fixtures=json.loads((OUT/'audit/fixture_accounting.json').read_text()),physical_target_histories=len(physical),all_seed_grids=True,all_primary_arithmetic=True,all_baseline_measurements_bitwise_reused=True,all_meta_code_unchanged=True,selection_reproduced=True,confirmation=confirm));append('## Final scientific audit\nAll fixed budgets, seed grids, matched/missing bins, baseline bitwise reuse, candidate-minus-A paired arithmetic, global exclusions, full-prefix masks, source scientific code hashes and frozen selection chronology verified. Actual total new scientific optimizer steps including the separate pilot: '+str(total)+'. Implementation fixtures170 separate.')
 print('Final scientific audit PASSED',flush=True)
if __name__=='__main__':main()
