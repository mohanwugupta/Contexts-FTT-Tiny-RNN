"""Complete-grid, budget, historical reuse and paired arithmetic checks, no training."""
from .common import *
from .protocol import DOSES,NEW_DOSES,source

def expected_counts():return dict(physical_histories=1200,frozen_cells=2383200,new_target_positions=11121600,new_target_updates=264000,new_target_states=264800,training_rows=396000,early_update_rows=24000,matched_rows=14400,new_optimizer_steps=30763200)
def validate_grid(rows):
 expected={(s,c,a,d) for s in range(20) for c in CONDS for d in DOSES for a in (['Historical'] if d==0 else ['Seen','Transfer'])};actual=[(r['seed'],r['condition'],r['arm'],r['dose']) for r in rows]
 if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Missing, duplicated or changed seed/condition/dose')
 return True

def validate_frozen_grid(f,last=330):
 keys=['seed','condition','arm','dose'];full=keys+['cue']
 try:
  assert set(f.cue)==set(CUES) and not f.duplicated(full+['update']).any()
  assert f['update'].notna().all() and f['update'].mod(1).eq(0).all()
  g=f.groupby(full)['update'].agg(['size','min','max','nunique'])
  assert len(g)==len(f[keys].drop_duplicates())*6
  assert g['size'].eq(last+1).all() and g['min'].eq(0).all() and g['max'].eq(last).all() and g['nunique'].eq(last+1).all()
 except (AssertionError,KeyError,TypeError) as e:raise ValueError('Incomplete or mislabeled frozen state/cue grid') from e
 return True

def main():
 counts=expected_counts();r=read('physical_run_registry');validate_grid(r.to_dict('records'));f=read('evaluations');validate_frozen_grid(f);validate_grid(f[['seed','condition','arm','dose']].drop_duplicates().to_dict('records'));assert len(f)==counts['frozen_cells'] and not f.duplicated(['seed','condition','arm','dose','update','cue']).any();assert len(read('training'))==counts['training_rows'];assert len(read('matched_acquisition'))==counts['matched_rows'];assert len(read('early_updates'))==counts['early_update_rows'];assert len(r[r.reused])==400 and len(r[~r.reused])==800
 old=pd.read_csv(SOURCE/'tables/evaluations.csv',float_precision='round_trip');keys=['seed','condition','arm','dose','update','cue'];old=old.set_index(keys).sort_index();reused=f[f.reused].set_index(keys).sort_index();assert old.index.equals(reused.index)
 for n in ['ce','accuracy','familiar_ce','gap','strong_familiar','strict_all']:np.testing.assert_array_equal(old[n].to_numpy(),reused[n].to_numpy())
 effects=read('primary_seed_effects');assert len(effects)==80
 ep=read('endpoints');p=ep[ep.cue.eq('C')&ep['update'].eq(330)].pivot(index=['seed','condition'],columns=['arm','dose'],values='gap')
 for row in effects.itertuples():
  expected=p.loc[(row.seed,row.condition),('Transfer',10000)]-p.loc[(row.seed,row.condition),('Transfer',960) if row.contrast=='Transfer:10000-960' else ('Seen',10000)];np.testing.assert_allclose(row.effect,expected,atol=1e-12,rtol=1e-12,equal_nan=True)
 # All genuinely unseen diagnostic paths remain absent throughout all new registry slots.
 for seed in range(20):
  reserved={tuple(p) for e in json.loads((OUT/f'registry/heldout_v2_seed_{seed}.json').read_text())['episodes'] for p in e['paths']};assert len(reserved)==24;seen=set()
  for b in range(5,43):
   reg=json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text());assert source.validate(reg);paths={tuple(p) for e in reg['episodes'] for p in e['paths']};assert not reserved.intersection(paths)
   for e in reg['episodes']:assert source.signature(e) not in seen;seen.add(source.signature(e))
 meta=json.loads((OUT/'audit/meta_complete.json').read_text());target=json.loads((OUT/'audit/targets_verified.json').read_text());held=json.loads((OUT/'audit/heldout_complete.json').read_text());assert meta['inner_updates']==28928000 and meta['outer_updates']==1446400;assert target['target_updates']==264000 and target['diagnostic_updates']==9600 and target['position_rows']==11121600;assert held['diagnostic_steps']==115200
 actual=meta['inner_updates']+meta['outer_updates']+target['target_updates']+target['diagnostic_updates']+held['diagnostic_steps'];assert actual==counts['new_optimizer_steps'];af=read('acquisition');assert af.loc[af.status.eq('never_acquired'),'post_acquisition_gap'].isna().all();assert af.loc[af.status.eq('never_acquired'),'first_familiar'].isna().all()
 start=json.loads((OUT/'audit/meta_started.json').read_text());critical={n:h for n,h in start['code'].items() if n.endswith(('/common.py','/protocol.py','/runtime.py','/storage.py','/train.py'))};core.hashes(ROOT,critical);parent=json.loads((SOURCE/'manifest.json').read_text());core.hashes(SOURCE,parent['outputs']);core.hashes(ROOT,parent['code']);core.hashes(ROOT,parent['inputs'])
 dump(OUT/'audit/final_tables_verified.json',dict(passed=True,**counts,all20_seeds_and_all_prespecified_doses=True,reused_histories_bitwise_equal=400,diagnostic_full_path_overlap=0,primary_pair_arithmetic=True,never_acquired_missing=True,source_and_meta_implementation_unchanged=True,optimizer_steps_in_verification=0));append('## Final scientific arithmetic and integrity checks\nAll1200physicalhistories/2383200frozencells/396000practices/24000earlyupdates/14400matchedbinrecordsretained. Old400historiesbitwiseequalonCE/accuracy/gap/flags. Fourprimarycomparisonsverifiedfromseed-levelendpoints. Exactnewoptimizerbudget30763200;180inner+9outerfixturesseparate. Correctedfullpathsreservedacrossall760newfullregistries;unusedfinal80slotsnottrained. Scientificmetaimplementation/source/predecessorsealsunchanged. No modelupdatesinverification.')
 print('All grids, paired effects, budgets, source hashes and reservations verified',flush=True)
if __name__=='__main__':main()
