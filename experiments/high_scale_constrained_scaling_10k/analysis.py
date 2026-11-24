"""Saved numerical measurements and original paired seed inference, never optimization."""
from .common import *
from .measurements import all_registry,collect
from .statistics import summarize,comparisons,time_summary
METRICS=['D_A','D_B','own_A','cross_A','own_B','cross_B','early_cost','sustained_cost','A_to_B','B_to_A','A_to_B_raw','B_to_A_raw','familiar_ce','strong_familiar','C_ce','gap','C_accuracy','strict_C','C_success','Mixed_average_ce','Mixed_average_accuracy','Mixed_average_success','Mixed_both_ce','Mixed_both_accuracy','Mixed_both_success','NoContext_ce','NoContext_accuracy','NoContext_gap','NoContext_minus_C','first_familiar','functional_split','initial_ce_A','initial_ce_B','initial_accuracy_A','initial_accuracy_B']
GEOMETRY_METRICS=['token','context_A','context_B','context_C','output','recurrent_rms','recurrent_spectral_norm','bias_rms','centered_contrast']

def same_state_geometry(out,weights):
 keys=['seed','scale','condition','phase','dose','update'];f=out.merge(weights,on=keys,how='outer',validate='one_to_one',indicator=True)
 assert len(f)==len(out) and f['_merge'].eq('both').all(),'Missing or mismatched frozen target state'
 return f.drop(columns='_merge')

def phase_C():
 receipt=OUT/'audit/phase_C_comparison.json'
 if receipt.exists():return
 assert json.loads((OUT/'audit/phase_C_verified.json').read_text())['passed'] and json.loads((OUT/'audit/C_targets_complete.json').read_text())['passed'];reg=all_registry(require_complete=False);reg=reg[(reg.phase.eq('Fresh'))|reg.dose.eq(1920)];assert len(reg)==198;tables=collect(reg);out=tables['outcomes'];fixed=out[out['update'].eq(330)];assert len(fixed)==198 and fixed.seed.nunique()==6;summary=summarize(fixed,['scale','condition','phase','dose','update'],METRICS);raw,ci=comparisons(fixed,METRICS);(OUT/'tables').mkdir(exist_ok=True)
 for n,f in [('outcomes',out),('summary',summary),('paired_seeds',raw),('paired_intervals',ci)]:write_frame(OUT/f'tables/phase_C_{n}.csv',f)
 (OUT/'PHASE_C_COMPARISON.md').write_text('# Registered1920 checkpoint comparison\n\nAll198 fixed Fresh/Reptile1920/FOMAML1920 physical references retained, including exact historical U branches. The P-U comparison table uses target330; the original P1920 target horizon is330, while historical U1920 extended rows remain separately available. Every objective continues regardless of behavioral sign or success. This is a numerical-integrity/protocol checkpoint, not condition selection. See tables/phase_C_summary.csv and phase_C_paired_intervals.csv for all signed outcomes, means/medians and six-seed paired intervals. No objective, scale, dose, threshold or budget was selected from these results.\n');dump(receipt,dict(passed=True,registered_physical_cases=198,target_update_comparison=330,all_six_seeds=True,outcome_gate=False,all_144_FOMAML_models_continue=True,selected_conditions=None,new_optimization=0));append('## Phase C comparison complete\nAll198 fixed source/reptile/final1920 cases measured from saved tables; original signed/gate/bin definitions, all success/failure signs and six-seed P-U intervals retained. Behavioral outcomes do not gate continuation: all144 U/P objective trajectories proceed to10000. No new model evaluations or optimization.');print('Phase C protocol checkpoint PASSED; no outcome selection',flush=True)

def main():
 if (OUT/'audit/analysis_complete.json').exists():return
 assert json.loads((OUT/'audit/conflicts_complete.json').read_text())['passed'];reg=all_registry()
 from .cache import prepare
 prepare(reg)
 tables=collect(reg);out=tables['outcomes'];tables['cohort_summary']=summarize(out,['scale','condition','phase','dose','update'],METRICS);tables['paired_seed_effects'],tables['paired_intervals']=comparisons(out,METRICS)
 same=same_state_geometry(out,tables['target_checkpoint_weights']);tables['same_state_behavior_geometry']=same;tables['target_geometry_summary']=summarize(same,['scale','condition','phase','dose','update'],GEOMETRY_METRICS);tables['target_geometry_paired_seeds'],tables['target_geometry_paired_intervals']=comparisons(same,GEOMETRY_METRICS)
 matched=tables['matched_acquisition'];tables['matched_summary']=summarize(matched,['scale','condition','phase','dose','horizon','performance_range','stratum'],['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE']);mr=[];mc=[]
 for key,g in matched.groupby(['horizon','performance_range','stratum'],dropna=False):
  raw,ci=comparisons(g.assign(update=key[0]),['specificity','practiced_gain','unpracticed_gain','start_A_CE','start_B_CE'])
  for f,dest in [(raw,mr),(ci,mc)]:f['horizon']=key[0];f['performance_range']=key[1];f['stratum']=key[2];dest.append(f)
 tables['matched_paired_seeds']=pd.concat(mr,ignore_index=True);tables['matched_paired_intervals']=pd.concat(mc,ignore_index=True);h=pd.read_csv(OUT/'tables/conflict_seed.csv',float_precision='round_trip');assert len(h)==612;hm=['conflict_accuracy','critical_ce','ce','context_benefit','conflict_context_benefit','appropriate_context_change','own_gain','cross_gain','specificity'];tables['conflict_summary']=summarize(h,['scale','condition','phase','dose'],hm);tables['conflict_paired_seeds'],tables['conflict_paired_intervals']=comparisons(h.assign(update=-1),hm);tables['joint_components']=out.merge(h,on=['seed','scale','condition','phase','dose'],how='left',suffixes=('','_conflict'))
 # Only the registered final target curves require dense bootstrap bands for Figure2.
 final=tables['training'][tables['training'].dose.eq(10000)];tables['final_training_intervals']=time_summary(final,['scale','condition','phase'],'update',['ce_before'],SEEDS)
 for name,f in tables.items():write_frame(OUT/f'tables/{name}.csv',f)
 from .mechanism import main as mechanism
 mechanism()
 dump(OUT/'audit/analysis_complete.json',dict(passed=True,physical_target_histories=1134,paired_seeds=6,whole_seed_resamples=10000,source_measure_cache=True,missing_horizons_not_imputed=True,original_gate=[170,330],no_gate_990=True,optimization=0));append('## Final saved-table analysis complete\nAll1134 physical histories, constraints/objectives/doses and original signed block/switch/gate/bin measures retained. P-U and own-parent/Fresh/1920-to10000/scale/objective paired comparisons use identical seed and observed target horizon, with unregistered comparisons explicitly missing rather than extrapolated. Six planned seeds,10000 fixed resamples; no score or causal mediation.');print('Final analysis PASSED',flush=True)
if __name__=='__main__':main()
