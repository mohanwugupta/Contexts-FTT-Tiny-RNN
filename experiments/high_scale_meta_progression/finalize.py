"""Final audit, exact exposure, table-only regeneration and researcher receipts."""
import concurrent.futures,multiprocessing,tempfile
from .common import *
from .verify import target_one

def budget_markdown(budget):return pd.DataFrame([budget]).T.rename(columns={0:'Actual/planned complete'}).to_markdown()

def write_once_frame(path,rows):
 import hashlib
 if path.exists():
  assert sha(path)==hashlib.sha256(pd.DataFrame(rows).to_csv(index=False).encode()).hexdigest(),'Changed retained finalization table'
 else:write_frame(path,rows)

def expected_exposure():return dict(meta_episodes=172800,inner_updates=3456000,outer_Adam_updates=120960,Reptile_interpolations=34560,target_updates_new=162360,target_updates_reused=3960,target_clones_new=3312,target_clones_reused=144,conflict_updates=56160,optimizer_updates_scientific=3798792,support_sequence_presentations=4838400,support_token_predictions=33868800,meta_query_examples=725760,meta_query_raw_prediction_cells=5080320,query_gradient_evaluations=293760,meta_query_positive_weight_cells=4942080,overlap_novel_eligible_cells=829440,overlap_critical_component_cells=110592,B_beta0_auxiliary_critical_cells=69120,fixture_inner_target_updates=285,fixture_outer_Adam_updates=8,fixture_Reptile_interpolations=4,fixture_optimizer_updates=293)

def main():
 if (OUT/'audit/final_review.json').exists():return
 for n in ['fresh_complete','reptile_verified','reptile_targets_complete','fomaml_verified','fomaml_targets_complete','diagnostics_complete','analysis_complete','report_complete']:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed']
 regs=[pd.read_csv(OUT/f'registry/{x}.csv') for x in ['fresh','reptile_targets','fomaml_targets']];r=pd.concat(regs,ignore_index=True);assert len(r)==288 and r.status.eq('completed').all();assert not r.duplicated(['seed','scale','phase','dose']).any();assert set(r.seed)==set(range(6)) and set(r.scale)=={4,5,6};budget=expected_exposure();assert int(r.target_updates.sum())==budget['target_updates_new'] and int(r.reused_updates.sum())==3960 and int(r.diagnostic_updates.sum())==3312
 meta=pd.concat([pd.read_csv(OUT/f'registry/{x}.csv') for x in ['reptile','fomaml']],ignore_index=True);assert len(meta)==90 and int(meta.episodes.sum())==172800 and int(meta.inner_updates.sum())==3456000 and int(meta.outer_updates.sum())==120960 and int(meta.interpolations.sum())==34560 and int(meta.sequence_presentations.sum())==4838400 and int(meta.query_examples.sum())==725760 and int(meta.query_gradient_evaluations.sum())==293760
 # Effective query exposure uses the source masks, not all raw Q predictions.
 eligible=critical=0
 for seed in range(6):
  for arm in ['O1','O2']:
   eps=json.loads((OUT/f'registry/{arm}_s{seed:02d}.json').read_text())['records'];eligible+=3*sum(e['eligible_count'] for e in eps);critical+=3*sum(e['critical_cells'] for e in eps)
 assert eligible==829440 and critical==110592 and 2*604800+2*34560*42+eligible==4942080
 receipt=OUT/'audit/all_targets_verified.json'
 if not receipt.exists():
  with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:checked=list(pool.map(target_one,r.to_dict('records')))
  write_frame(OUT/'audit/all_targets_verified.csv',checked);dump(receipt,dict(passed=True,cells=288,states=sum(x['states'] for x in checked),all_four_probabilities=True,optimizer_updates=0))
 # Logical +0 labels point to existing physical targets; they add no observations.
 aliases=[]
 for s in range(6):
  for k in [4,5,6]:
   fresh=r[r.seed.eq(s)&r.scale.eq(k)&r.phase.eq('Fresh')].iloc[0];parent=r[r.seed.eq(s)&r.scale.eq(k)&r.phase.eq('R')&r.dose.eq(1920)].iloc[0]
   aliases.append(dict(seed=s,scale=k,phase='R',dose=0,physical_path=fresh.path,sha256=fresh.sha256,optimizer_updates=0))
   for arm in ['T','B','O1','O2']:aliases.append(dict(seed=s,scale=k,phase=arm,dose=0,physical_path=parent.path,sha256=parent.sha256,optimizer_updates=0))
 write_once_frame(OUT/'registry/aliases.csv',aliases);write_once_frame(OUT/'registry/exposure.csv',[budget])
 if not (OUT/'audit/exposure_verified.json').exists():dump(OUT/'audit/exposure_verified.json',dict(passed=True,**budget))
 else:assert json.loads((OUT/'audit/exposure_verified.json').read_text())==dict(passed=True,**budget)
 figreceipt=OUT/'audit/figure_regeneration.json'
 if not figreceipt.exists():
  from .figures import main as figures
  external=Path(tempfile.mkdtemp(prefix='high_scale_meta_progression_regeneration_'));figures(OUT,external);original={p.name:sha(p) for p in (OUT/'figures').glob('*') if p.suffix in ['.png','.svg']};regenerated={p.name:sha(p) for p in external.glob('*') if p.suffix in ['.png','.svg']};expected=['figure_1_'+v for v in ['full990','first100','first20','closeup']]+['figure_2_six_contexts','figure_3_reptile']+['figure_4_fomaml_'+v for v in ['balanced','first_A','first_B']]+['figure_5_joint_components','figure_6_RMS','figure_6_computation'];assert original==regenerated and set(original)=={n+ext for n in expected for ext in ['.png','.svg']};dump(figreceipt,dict(passed=True,files=len(original),hashes=original,table_only=True,optimizer_updates=0,external_destination=str(external)))
 audit=f'''# Implementation and source audit

All source dependencies are indexed by registry/parent_hashes.json and the final manifest. Parent scale-sweep manifest SHA256: {sha(SWEEP/'manifest.json')}. All18 fresh states derive from the original six archived High tensors;12 exact scale4/6 target330 branches are reused, including all331 model/policy/Adam snapshots and frozen tables. Target330 continuation was verified with original moments, not reset. Target-trained diagnostic weights never initialize Reptile. All72 FOMAML forks start bitwise from their respective common Reptile1920 state with fresh outer Adam. Native predecessor files/seals remain unchanged.

## Actual budgets

{budget_markdown(budget)}

Every figure and statistic uses existing saved numerical rows after the scientific stages finish. All288 physical target histories,90 meta trajectories and234 reserved conflict adaptations remain. The90 logical zero-dose aliases are explicit references, not added samples. Periodic meta checkpoints and per-step durable journal latest states retain source, registry, RNG and optimizer identities; registered inner trace subsets retain full model/policy/Adam states after every inner update. Full analytic query-gradient checks cover fixed trace subsets; all outer updates are independently reconstructed from saved gradients. No claim of an independent analytic check of every inner query is made. A process interruption between an uncommitted computation and its durable transaction can leave uncommitted compute exposure uncertain; no such meta interruption occurred in this run.

## Retained failures and implementation corrections

Logs01/03/06/09/14/16/24/31/35/37 retain RED failures. Log02 exposed scale5 rejection by the predecessor scale registry; its unchanged two-matrix transformation was implemented in the new registered namespace. Log07 exposed a reporting packer that assumed initialized outer Adam; only that packer was adapted to empty fresh Adam, and source gradient/parameter parity passed08. Log13 stopped at the original330 cue-helper guard. All finite partial states and stopped attempts remain under target/, with exact transfer/recovery identities in registry/implementation_recovery_v2.csv and audit/implementation_correction_v2.json. The two partially advanced scale5 runs contributed244 target updates and16 diagnostic clones, transferred into final counters without repetition. The code snapshot/stop receipt remain; target_v2 explicitly versions the correction. RED25 also caught that horizon guard within source boundary measurement and was fixed before final analysis. No threshold, hyperparameter, architecture, task mask, scientific stopping rule or optimizer contract changed.

The fixture accounting originally named285 inner/target updates without outer moves; the clarification records293 optimizer updates and4 Reptile interpolations separately. All logs and append-only scratchpad entries remain. No nonfinite rescue or favorable-case substitution is permitted. Missing matched-acquisition cells, negative signs and all acquisition failures stay in tables.

## Retained precision exception

All72 FOMAML budgets completed before read-only verification. The inherited strict Overlap forward reference (atol=rtol=2e-5) failed once on a near-cancelled logit, error4.31663141e-5 amid logits of order100. Log34 and the original verifier snapshot are retained. The strict outcome remains a failure. A separate array-only certificate uses the EXISTING predecessor measure_model bound64*float32.eps, with the same double tolerance1e-10+1e-12*max magnitude; it does not change or relabel the strict threshold. Every trace satisfies that unchanged bound and the original analytic-gradient/CE/Adam tolerances. Bound consistency is not proof that every discrepancy is solely rounding. No checkpoint, prediction or optimizer update was rerun/modified. Final review55 later completed all numerical/figure checks but stopped on a documentation-only tabulate keyword typo; logs55–57 retain its failing/passing renderer test, with alias/exposure tables compared byte-identically on resumption. See audit/fomaml_precision_checks.csv and logs49–51, including negative fixtures rejecting wrong cues and out-of-bound errors.

## Tests and scientific limitations

Numerical references establish native scaling/forward, exact source reuse and target Adam continuation, Reptile interpolation, Transfer gradients, balanced aggregation and structured early/familiar/novel source parity. Additional tests reject wrong cue horizons, stale forks/resume codes, diagnostic path leakage, incorrect critical labels, duplicate/corrupted grids and checkpoint hashes; original functional-gate tables reproduce. Final pure tests are recorded in audit/52_final_pure_tests.log;12 predecessor regressions passed in audit/26_predecessor_regressions.log. Expensive optimization fixtures were not repeatedly rerun after passing. Six trained seeds remain the statistical unit. Procedure/ecology/exposure differences and floor limits remain scientific uncertainties, not implementation defects.
'''
 (OUT/'IMPLEMENTATION_AUDIT.md').write_text(audit)
 (OUT/'REPRODUCIBILITY.md').write_text('''# Reproduction without scientific optimization

From repository root:

```sh
python -m experiments.high_scale_meta_progression.seal --verify
MPLCONFIGDIR=/tmp/high_scale_progression_mpl python -m experiments.high_scale_meta_progression.figures --output-dir /tmp/high_scale_progression_figures
python -m pytest tests/high_scale_meta_progression -k 'not native_reptile_transfer_balanced_structured_gradients and not adam_restore' --import-mode=importlib -q
```

Use an empty external figure destination. Fixed SVG salt/date metadata and deterministic CSV reads reproduce every PNG/SVG hash; audit/figure_regeneration.json records the independent table-only regeneration. Do not rerun target/meta/conflict optimization, benchmark or expensive numerical fixtures for reproduction. Budgets are spent. Frozen/negative/missing outcomes remain part of the evidence; corrections require explicit versioning and retained logs.

`registry/initializations.csv` indexes all18 fresh tensors. `registry/reused_targets.csv` indexes12 exact archived330 prefixes; `registry/implementation_recovery_v2.csv` records finite partial-state transfer without replay. `registry/{fresh,reptile_targets,fomaml_targets}.csv` indexes288 complete target bundles and source/checkpoint hashes. `registry/aliases.csv` lists zero-dose references only. Target directories contain all331/991 model/policy/Adam states, six-cue summaries and seven-position/four-class probabilities/logits, mathematical checks and decomposition. `registry/{reptile,fomaml}.csv`, task streams and meta/ journals/checkpoints index every meta procedure and computational exposure. Full inner histories are retained for fixed audit subsets; periodic complete meta states and per-step durable latest-state journals support verified resumption. Exact-source code snapshots record stage chronology.

`tables/outcomes.csv` is the complete seed-level endpoint table; `paired_seed_effects.csv`, `paired_intervals.csv`, `scale_paired_*`, `matched_*`, `conflict_*` and `joint_components.csv` preserve actual sample sizes and signed effects. Raw block/switch/position/cycle and training/evaluation data are figure coordinates. Matched bins use identical330/990 exposure windows. `tables/meta_geometry.csv` retains actual RMS, recurrence and centered contrast/source hashes. Claims map to these sources in CLAIM_LEDGER.csv. RESEARCHER_HANDOFF.md explains the algorithms and limitations.

The final manifest seals every output plus code and the complete ancestor dependency chain. Read-only --verify rehashes all three sets, without model training. Post-seal completion receipts belong outside the package under scratchpads/. Publication coverage remains PUBLICATION.md; bulk local evidence is not automatically claimed to be in a Git clone, and ignore rules were not changed.
''')
 # Tests are run by the orchestrator before this final evidence review.
 assert '17 passed' in (OUT/'audit/52_final_pure_tests.log').read_text();dump(OUT/'audit/final_review.json',dict(passed=True,physical_targets=288,meta_runs=90,conflict_runs=234,source_and_checkpoint_hashes=True,all_four_probabilities_verified=True,strict_forward_exceptions=json.loads((OUT/'audit/fomaml_verified.json').read_text())['strict_forward_failures'],strict_forward_thresholds_unchanged=True,all_fixed_budgets=budget,figure_files=len(json.loads(figreceipt.read_text())['hashes']),negative_and_missing_preserved=True,selection=False));append('## Final evidence review complete\nExact scientific budgets, source/fork identities, all physical targets/conflicts, fixed trace analytic gradients and every outer Adam reconstruction passed. All figure files regenerated byte-identically from CSVs only. Reports, correction logs and fixture accounting retained. Seal next; no further writes inside the package after sealing.');print('Final evidence review PASSED',flush=True)
if __name__=='__main__':main()
