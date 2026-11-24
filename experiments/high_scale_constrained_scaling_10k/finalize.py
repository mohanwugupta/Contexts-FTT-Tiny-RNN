"""Read-only complete scientific evidence review, exact exposure and regeneration."""
import concurrent.futures,multiprocessing,tempfile
from .common import *
from .budget import plan
from .measurements import all_registry
from experiments.heldout_context_meta_scaling_10k.adapters import bind
FIXTURE_OPTIMIZER_UPDATES=1984

def receipt(path,data):
 if path.exists():assert json.loads(path.read_text())==data,'Changed retained audit receipt'
 else:dump(path,data)

def frame_receipt(path,rows):
 """Resume derived receipts only when their exact serialization is unchanged."""
 data=pd.DataFrame(rows).to_csv(index=False).encode()
 if path.exists():assert path.read_bytes()==data,'Changed retained CSV receipt: '+str(path)
 else:path.write_bytes(data)

def finish_ledger(folder):
 """Retain the provisional ledger; resolve only explicitly pending hashes."""
 folder=Path(folder);path=folder/'CLAIM_LEDGER.csv';prior=folder/'audit/CLAIM_LEDGER_initial.csv'
 ledger=pd.read_csv(prior if prior.exists() else path,float_precision='round_trip')
 for index,row in ledger.iterrows():
  source=folder/row.source;assert source.is_file(),row.source
  if row.sha256=='pending_finalization':ledger.loc[index,'sha256']=sha(source)
  else:assert row.sha256==sha(source),'Changed claim source: '+row.source
 if not prior.exists() and ledger.sha256.ne(pd.read_csv(path).sha256).any():path.rename(prior)
 if prior.exists():frame_receipt(path,ledger)
 else:assert not ledger.sha256.eq('pending_finalization').any()

def meta_counts(meta):return dict(new_meta_episodes=int(meta.new_episodes_since_source.sum()),new_inner_updates=int(meta.new_inner_updates.sum()),new_outer_Adam_updates=int(meta.new_outer_updates.sum()),new_Reptile_interpolations=int(meta.new_interpolations.sum()))
def check_meta_counts(meta):
 got=meta_counts(meta);p=plan()
 for n,v in got.items():assert v==p[n],(n,v,p[n])
 return got

def target_one(r):
 from experiments.high_scale_meta_progression import verify as native
 directory=(ROOT/r['path']).parent;rr=dict(r,directory=str(directory.relative_to(ROOT)))
 with bind(native,OUT=ROOT,configure=configure):q=native.target_one(rr)
 q['condition']=r['condition'];return q

def exposures():
 reg=pd.read_csv(OUT/'registry/scale_histories.csv');rows=[]
 for r in reg.to_dict('records'):
  v=np.load(OUT/r['path'])['scalars'];new=np.ones(len(v),bool) if r['condition']=='P' else (v[:,0]>1920 if r['phase']!='R' else np.zeros(len(v),bool));x=v[new];a=r['phase'];s=r['seed'];start=1920//(2 if a=='B' else 1) if r['condition']=='U' else 0;eps=json.loads((OUT/f'registry/{a}_s{s:02d}.json').read_text())['records'][start:] if len(x) else [];assert len(eps)==len(x);eligible=sum(e.get('eligible_count',0) for e in eps);critical=sum(e.get('critical_cells',0) for e in eps);positive=int(x[:,11].sum())*7 if a in ['T','B'] else (42*len(x)+eligible if a.startswith('O') else 0)
  rows.append(dict(seed=s,scale=r['scale'],condition=r['condition'],phase=a,new_outer_records=len(x),new_meta_episodes=int(x[:,6].sum()),new_inner_updates=int(x[:,7].sum()),new_outer_Adam_updates=int(x[:,8].sum()),new_Reptile_interpolations=int(x[:,9].sum()),support_sequence_presentations=int(x[:,10].sum()),support_token_predictions=int(x[:,10].sum())*7,query_examples=int(x[:,11].sum()),query_raw_prediction_cells=int(x[:,11].sum())*7,query_gradient_evaluations=int(x[:,12].sum()),query_positive_weight_cells=positive,overlap_novel_eligible_cells=eligible,overlap_critical_component_cells=critical,B_beta0_auxiliary_critical_cells=4*len(x) if a=='B' else 0))
 f=pd.DataFrame(rows);frame_receipt(OUT/'registry/exposure_by_model.csv',f);tot={n:int(f[n].sum()) for n in f if n not in ['seed','scale','condition','phase']};return tot

def aliases(reg):
 rows=[]
 for s in SEEDS:
  for k in SCALES:
   fresh=reg[reg.seed.eq(s)&reg.scale.eq(k)&reg.phase.eq('Fresh')].iloc[0]
   for c in ['U','P']:
    rows.append(dict(seed=s,scale=k,condition=c,phase='R',dose=0,physical_path=fresh.path,sha256=fresh.sha256,new_optimizer_updates=0))
    parent=reg[reg.seed.eq(s)&reg.scale.eq(k)&reg.condition.eq(c)&reg.phase.eq('R')&reg.dose.eq(1920)].iloc[0]
    for a in ARMS:rows.append(dict(seed=s,scale=k,condition=c,phase=a,dose=0,physical_path=parent.path,sha256=parent.sha256,new_optimizer_updates=0))
 return rows

def main():
 if (OUT/'audit/final_review.json').exists():return
 for n in ['source_audit','phase_B_verified','phase_C_verified','phase_C_comparison','phase_D_verified','targets_complete','conflicts_complete','analysis_complete','figures_complete','report_complete']:assert json.loads((OUT/f'audit/{n}.json').read_text())['passed'],n
 meta=pd.concat([pd.read_csv(OUT/f'registry/phase_{p}_meta.csv') for p in ['B','D']],ignore_index=True);assert len(meta)==162;got=check_meta_counts(meta);reg=all_registry();assert reg.status.eq('completed').all();new=pd.concat([pd.read_csv(OUT/f'registry/{n}.csv') for n in ['B_targets','C_targets','targets']],ignore_index=True);p=plan();assert len(new)==p['new_target_runs'] and int(new.new_target_updates.sum())==p['new_target_updates'] and int(new.new_diagnostic_updates.sum())==p['new_target_clones'];con=pd.read_csv(OUT/'registry/conflicts.csv');assert len(con)==p['new_conflict_runs'] and int(con.diagnostic_steps.sum())==p['new_conflict_updates'];extra=exposures()
 for name,v in got.items():assert extra[name]==v
 actual=dict(**extra,new_target_histories=len(new),reused_target_histories=288,new_target_updates=int(new.new_target_updates.sum()),new_target_one_update_clones=int(new.new_diagnostic_updates.sum()),new_conflict_diagnostic_cells=len(con),new_conflict_updates=int(con.diagnostic_steps.sum()),reused_conflict_cells=90,fixture_optimizer_updates=FIXTURE_OPTIMIZER_UPDATES,fixture_Reptile_interpolations=16)
 actual['new_scientific_optimizer_updates']=actual['new_inner_updates']+actual['new_outer_Adam_updates']+actual['new_target_updates']+actual['new_target_one_update_clones']+actual['new_conflict_updates'];assert actual['new_scientific_optimizer_updates']==p['new_scientific_optimizer_updates'];frame_receipt(OUT/'registry/exposure.csv',[actual]);receipt(OUT/'audit/exposure_verified.json',dict(passed=True,**actual))
 target_receipt=OUT/'audit/all_targets_verified.json'
 if not target_receipt.exists():
  with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:verified=list(pool.map(target_one,reg.to_dict('records')))
  write_frame(OUT/'audit/all_targets_verified.csv',verified);receipt(target_receipt,dict(passed=True,physical_histories=len(verified),states=sum(r['states'] for r in verified),all_four_probabilities=True,all_Adam_C_columns=True,optimizer_updates=0))
 frame_receipt(OUT/'registry/aliases.csv',aliases(reg));finish_ledger(OUT)
 figure_receipt=OUT/'audit/figure_regeneration.json'
 if not figure_receipt.exists():
  from .figures import main as figures
  external=Path(tempfile.mkdtemp(prefix='hscs_table_regeneration_'));figures(OUT,external);original={p.name:sha(p) for p in (OUT/'figures').glob('*') if p.suffix in ['.png','.svg']};regenerated={p.name:sha(p) for p in external.glob('*') if p.suffix in ['.png','.svg']};assert original==regenerated;receipt(figure_receipt,dict(passed=True,files=len(original),hashes=original,table_only=True,new_optimization=0,external_destination=str(external)))
 tests=[]
 for path in sorted((OUT/'audit').glob('*.log')):
  txt=path.read_text(errors='replace');tests.append(dict(log=path.name,sha256=sha(path),actual_pass_summary='; '.join(line for line in txt.splitlines() if ' passed' in line)[-500:],retained_failure=('FAILED ' in txt or 'ERROR collecting' in txt or 'Traceback' in txt),note='34_RED_continuation_trace.log actually passed due asynchronous import race; not a RED failure' if path.name.startswith('34_') else ''))
 write_frame(OUT/'audit/test_log_index.csv',tests)
 audit=f'''# Implementation and source audit

The full predecessor chain is pinned in registry/parent_hashes.json; source manifest SHA256={sha(SOURCE/'manifest.json')}. All18 fresh tensors and90 actual per-group floors are retained; U checkpoints/moments/RNG and native1920 task prefixes were verified before execution. Historical U branches and288 target/90 conflict cases were reused exactly, never retrained. The intervention begins from fresh weights during Reptile; it is not a rescaling of trained U.

Actual fixed new exposure (historical references are not new replications):

{pd.DataFrame([actual]).T.rename(columns={0:'Actual'}).to_markdown()}

All outer updates and radial moves are reconstructed from saved gradients/states. The immutable trainer/runtime/protocol/common/registry/storage modules are pinned by audit/SCIENTIFIC_CODE_FREEZE.json. Native first-order T/B/O losses, query masks, support schedules, joint clipping, detached execution objective and native Adam remain unchanged. Benchmark09 establishes numerical equivalence CPU1/2 across the fixed native fixture;1984 fixture optimizer steps and16 interpolations are separate from scientific exposure.

Every inner update's scalar CE/execution/total loss, norm and hidden magnitude is retained. The compact top-level meta query_ce is populated for T; Reptile has no query, and B/O1/O2 retain their component query CEs in the fixed complete trace subset rather than a dense combined scalar. Those unlogged scalar fields remain NaN. They are not zero loss or failed target evaluations. All requested target-state six-cue CE/prediction grids are complete; no adaptation is replayed to fill optional meta-loss fields.

Atomic chunks commit at most60 outer records and complete model/optimizer/RNG/cursor together. Registered doses and phase boundaries commit regardless of chunk size. An unhandled process kill before commit can leave up to59 uncommitted outer computations uncertain; no stronger no-replay guarantee is claimed. No intentional interruption or uncontrolled retry occurs. Caught failures retain pending records and actual model/optimizer. The observation-only continuation adapter adds the previously declared firstP resumed trace1921/B1922 without changing any training math, RNG or update counts.

All1134 target histories were checked from saved tables/states: six cues, every target update, seven positions, allfour probabilities/logits, original targets, actual Adam counters, bitwise fixed C and the original Linear identities. No frozen physical branch is re-evaluated in this audit. The unseen conflicts use the corrected-v2 masks, original20 steps and correct/wrong-cue probes; all libraries exclude them. Twelve tasks average within each seed. Support/query exposure above is derived from actual native counters and registered masks, including Overlap's two examples per update, final partial blocks and B's beta0 audit gradient.

Retained failures are indexed in audit/test_log_index.csv and the append-only scratchpad. The derived initial504-diagnostic count omitted18 P Reptile cases; budget.py corrects the prescribed grid to522/125280 before any conflict execution. Original critical protocol.py and source_audit arithmetic remain as historical records. Audit29 caught the NaN reused-flag path bug; the registered states path now uniquely locates saved tables, and no faulty cache or evaluation was accepted. Audit34 was launched as RED but actually passed because imports finished after implementation; it is explicitly NOT counted as a failing test. Logs36/38/43/45 establish actual missing-stage failures before implementation. No threshold, tolerance, rank, objective, cue or budget was retuned.

Strict forward exceptions remain in tables/strict_precision_exceptions.csv with their original criterion unchanged. Separate existing float32 propagation bounds and original analytic-gradient/CE/Adam checks do not erase these strict failures. The sealed predecessor's original exception is also retained. Fixed full inner-state traces are a registered subset; there is no claim of an independent analytic audit of every inner update. Six seeds and procedure/exposure differences limit inference; geometry is not functional splitting or causal mediation.
'''
 (OUT/'IMPLEMENTATION_AUDIT.md').write_text(audit)
 (OUT/'REPRODUCIBILITY.md').write_text('''# Reproduce without training

From the repository root, after restoring all release evidence indexed in PUBLICATION.md:

```sh
PYTHONPATH=.:src python -m experiments.high_scale_constrained_scaling_10k.seal --verify
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/hscs_mpl python -m experiments.high_scale_constrained_scaling_10k.figures --output-dir /tmp/hscs_figures
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pytest -q --import-mode=importlib tests/high_scale_constrained_scaling_10k
```

Use a new empty external figure directory. All seven PNG/SVG families regenerate from saved numerical tables alone, with fixed SVG salt and date metadata. audit/figure_regeneration.json records byte-identical hashes. Pure tests use synthetic/frozen data and do not run the1984-step optimizer benchmark. Do not rerun meta/target/conflict/benchmark scientific stages to reproduce plots; their budgets are spent. Sealed output corrections require a new version and retained failure logs.

registry/environment.json records the actual Python, numerical and plotting versions, which match the exact copied registry/requirements.lock and pyproject.toml. registry/dependency_snapshots.json hashes those original dependency files. Byte-identical rendering is verified in that recorded environment; numerical CSVs remain the primary cross-platform reference.

Start with REPORT.md, RESEARCHER_HANDOFF.md and CLAIM_LEDGER.csv. registry/{initializations,floors,U_checkpoints}.csv index exact fresh/source identities. registry/{phase_B_meta,phase_C_meta,phase_D_meta}.csv contain progression counters and source/checkpoint hashes; only B+D count new meta exposure because D's P rows include the C prefix. registry/reused_targets.csv references288 archived histories; B_targets/C_targets/targets.csv add846. Source-registered states.pt paths locate all331/991 model/policy/Adam states and six-cue tables including seven-position/four-logit/probability distributions. registry/aliases.csv records only zero-dose references.

The physical target directories hold every training/evaluation/diagnostic record, position predictions, mathematical/decomposition checks and initial-source hash. tables/{outcomes,blocks,switch_costs,position_switch_costs,balanced_cycles,learning_effects,functional_splitting,matched_acquisition}.csv preserve signed measurements, original floors, missing bins and the unchanged170/330 criterion. tables/{paired_seed_effects,paired_intervals,matched_paired_intervals,conflict_paired_intervals}.csv use whole trained seeds and10000 fixed draws, means/medians/signs; unavailable P1920 target990 comparisons remain missing. tables/final_joint_components.csv has all components without a new weighted score.

Projection histories are complete per-outer-step arrays in tables/scale_histories/*.npz, with SCHEMA.json and per-file source/hash receipts; scale_trajectory_plot.csv supplies exact sampled plotting rows. Full gradients, raw states and projected parameters remain in immutable SQLite journals, with complete registered checkpoints and fixed inner-state traces. tables/meta_geometry.csv, effective_context_by_position.csv and parameter_directions.csv give all seven centered contextual responses and source identities. NoContext geometry analyses use saved zero-target-update performance. registry/exposure.csv and exposure_by_model.csv separate all scientific/fixture/reused work.

The final manifest pins every program output, implementation/test/PRD file and complete predecessor hash chain. Read-only hash verification does not load an optimizer, alter weights or repeat any frozen branch. Inspect retained strict precision exceptions separately from bounded float32 consistency and all retained RED logs.
''')
 append('## Final read-only review\nComplete original target grids/probabilities/Adam/C and Linear checks, full new budget/exposure, logical aliases, claims and external table-only figure hashes verified. No new optimization or frozen branch evaluation. Visual inspection receipt remains required before final sealing.');dump(OUT/'audit/final_review.json',dict(passed=True,new_meta_runs=162,final_FOMAML=144,physical_targets=1134,new_scientific_optimizer_updates=actual['new_scientific_optimizer_updates'],table_regeneration=True,no_new_optimization=True,visual_review_required=True))
if __name__=='__main__':main()
