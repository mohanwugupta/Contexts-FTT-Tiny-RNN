"""Independent final grid, budget, baseline reuse, pairing and code audit."""
from .common import *
from .protocol import ARMS,SCREEN,CONFIRM,critical_mask,old
from .selection import choose,screen_candidates

def check_budgets(b):
 expected=dict(meta_inner=1958400,meta_outer=48960,target=39600,target_clones=1440,conflict_clones=28800)
 if b!=expected:raise ValueError('Changed registered scientific budget')
 return sum(b.values())

def main():
 selection=json.loads((OUT/'audit/selection_seal.json').read_text());assert sha(OUT/'SCREENING_DECISION.md')==selection['decision_sha'];r,s=choose(screen_candidates(read('screen_seed_outcomes')));assert r['selected']==selection['selected'] and r['tolerance_met']==selection['tolerance_met'];assert sha(OUT/'tables/screen_seed_outcomes.csv')==selection['screen_table_sha'];assert not list((OUT/'meta').glob('*/failure.json'))
 totals=dict(meta_inner=0,meta_outer=0,target=0,target_clones=0,conflict_clones=0);critical=['common.py','protocol.py','runtime.py','train.py','validation.py'];physical=[]
 for stage,seeds,nmeta,ntarget,npairs in [('screen',SCREEN,36,36,240),('confirm',CONFIRM,42,84,960)]:
  for kind in ['meta','targets']:
   assert json.loads((OUT/f'audit/{stage}_{kind}_verified.json').read_text())['passed']
  m=pd.read_csv(OUT/f'registry/{stage}_meta_runs.csv');t=pd.read_csv(OUT/f'registry/{stage}_targets.csv');h=pd.read_csv(OUT/f'registry/{stage}_heldout.csv');assert len(m)==nmeta and len(t)==ntarget and len(h)==ntarget and set(m.seed)==set(seeds);assert m.pairs.eq(npairs).all() and m.inner_updates.eq(npairs*40).all() and m.outer_updates.eq(npairs).all();totals['meta_inner']+=int(m.inner_updates.sum());totals['meta_outer']+=int(m.outer_updates.sum());totals['target']+=int(t.target_updates.sum());totals['target_clones']+=int(t.diagnostic_updates.sum());totals['conflict_clones']+=int(h.diagnostic_steps.sum())
  started=json.loads((OUT/f'audit/{stage}_meta_started.json').read_text());core.hashes(ROOT,{k:v for k,v in started['code'].items() if Path(k).name in critical})
  f=read(stage+'_evaluations');expected=42 if stage=='screen' else 98;assert len(f)==expected*331*6 and not f.duplicated(['seed','arm','dose','update','cue']).any() and f.ce.notna().all() and set(f.cue)==set(CUES);assert len(read(stage+'_blocks'))==expected*33 and len(read(stage+'_matched_acquisition'))==expected*12
  reg=read(stage+'_physical_registry');assert len(reg)==expected;physical.extend(reg.to_dict('records'))
  for seed in seeds:
   a=f[f.seed.eq(seed)&f.arm.eq('Baseline')].sort_values(['update','cue']);b=pd.read_csv(SOURCE/f'target/s{seed:02d}_P2_I_Transfer_d960/evaluations.csv',float_precision='round_trip').sort_values(['update','cue']);np.testing.assert_array_equal(a.ce,b.ce);np.testing.assert_array_equal(a.accuracy,b.accuracy)
  # Independently reconstruct primary per-seed endpoint differences.
  raw=read(stage+'_paired_seed_effects');out=read(stage+'_seed_outcomes')
  for a in [x for x in out.arm.unique() if x not in ['Baseline','C']]:
   for d in sorted(out.dose.unique()):
    x=out[out.dose.eq(d)&out.arm.eq(a)].set_index('seed');y=out[out.dose.eq(d)&out.arm.eq('C')].set_index('seed')
    if not len(x):continue
    for met in ['gap','ce','conflict_accuracy','conflict_ce','D_A','D_B']:
     got=raw[raw.candidate.eq(a)&raw.dose.eq(d)&raw.metric.eq(met)].set_index('seed').effect.reindex(seeds);want=(x[met].astype(float)-y[met].astype(float)).reindex(seeds);np.testing.assert_allclose(got,want,rtol=0,atol=1e-15,equal_nan=True)
 for seed in range(20):
  reg=json.loads((OUT/f'registry/pairs_seed_{seed}.json').read_text());reserved={tuple(p) for e in json.loads((OUT/f'registry/heldout_v2_seed_{seed}.json').read_text())['episodes'] for p in e['paths']}
  for p in reg['pairs']:
   assert not reserved.intersection(tuple(path) for kind in ['shared','conflict'] for path in p[kind]['paths']);mask=critical_mask(p['conflict'],old.queries(p['conflict'],'P2'));assert mask[:,2].all() and mask.sum()==4
 total=check_budgets(totals);assert len(physical)==140 and len({r['path'] for r in physical})==140;assert sum(bool(x['reused']) for x in physical)==20;dump(OUT/'audit/final_tables_verified.json',dict(passed=True,scientific_optimizer_steps=total,breakdown=totals,fixture_inner_updates=240,fixture_outer_updates=6,physical_target_histories=140,new_target_histories=120,reused_baselines=20,all_seed_grids=True,all_primary_arithmetic=True,all_baseline_measurements_bitwise_reused=True,all_meta_code_unchanged=True,selection_reproduced=True,reserved_critical_positions=True));append('## Final scientific audit\nExact97,920episodes/48,960pairedouterupdates,1,958,400innerupdates;120newtargets/20reusedbaselines;39,600target+1,440targetclone+28,800conflictcloneupdates. Total2,077,200newscientificoptimizersteps,240inner+6outerimplementationfixturesseparate. All140physicalhistories/331states/sixcues,33blocks,12matchedbinrecordsperhistory,criticalmasks,primarypairedarithmeticandoldbaselinebitwiseCE/accuracyreuseverified. Scientificmetacodeunchanged;originalselectionreproducesexactly.')
 print('Final scientific audit PASSED',flush=True)
if __name__=='__main__':main()
