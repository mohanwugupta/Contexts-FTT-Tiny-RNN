"""Resume only missing extensions after the retained Boolean-accuracy export failure."""
from .common import *
from .analysis import extended_tables

def main():
 assert not (OUT/'audit/analysis_complete.json').exists()
 blocker=json.loads((OUT/'audit/pipeline_blocker.json').read_text());assert blocker['stage']=='36_scientific_analysis'
 assert '3 passed' in (OUT/'audit/65_GREEN_boolean_accuracy.log').read_text()
 names=['evaluations','endpoints','block_transfer','joint_outcomes','acquisition','geometry','learning_effects','matched_acquisition','C_positions'];before={str(p.relative_to(OUT)):sha(p) for p in (OUT/'tables').glob('*.csv')};assert len(before)==42
 dump(OUT/'audit/base_analysis_before_recovery.json',dict(passed=True,hashes=before,reason='Base tables complete; Boolean position-accuracy subtraction prevented extension export'))
 tables={n:read(n) for n in names};extended_tables(tables);core.hashes(OUT,before)
 dump(OUT/'audit/analysis_complete.json',dict(passed=True,models=1200,primary='Transfer10000-960 and Transfer-Seen10000; independent P1/P2 separate20pairedseeds',bootstrap=10000,rng=20260925,post_acquisition_inclusive=True,never_acquired_missing=True,selection=False,recovered_from_saved_base_tables=True))
 dump(OUT/'audit/analysis_recovery_complete.json',dict(passed=True,unchanged_base_tables=len(before),only_fix='Booleanaccuracycasttofloat0/1beforesignedsubtraction;NaNspreserved',original_failure='audit/36_scientific_analysis.log',regression_red='audit/64_RED_boolean_accuracy.log',regression_green='audit/65_GREEN_boolean_accuracy.log',native_model_evaluations=0,optimizer_steps=0,threshold_changes=False,checkpoint_changes=False))
 append('## Analysis recovery GREEN\nCompleted onlythemissingextensionsfrom42alreadyverifiedbaseCSVs. Booleanaccuracyindicatorsconvertto0/1forsignedpairedsubtraction, preservingNaNs. All42basefilehashesunchanged; fourprimarycomparisons/criteria/budgetsunchanged. Sourcefrozenmodelsneitherre-evaluatednorupdated. Originalfailure/blockerandRED64/GREEN65logsretained. Resumeonlyremainingdiagnostics/statistics/figures/reports, notmeta/targettraining.')
 print('Extended analysis recovered; all42 base-table hashes unchanged; no model replay',flush=True)
if __name__=='__main__':main()
