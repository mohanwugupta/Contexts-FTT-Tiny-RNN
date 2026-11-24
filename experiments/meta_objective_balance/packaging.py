"""Researcher handoff, immutable-source map and reproducibility instructions."""
from .common import *
from .protocol import ARMS
from .report import methods,md,fmt,ci,primary,scientific_takeaways,polish

def main():
 selected=json.loads((OUT/'audit/selection_seal.json').read_text())['selected'];p=primary(read('confirm_paired_intervals'),selected);f=read('confirm_seed_outcomes');end=f[f.dose.eq(1920)];rows=[]
 for a in selected+['C']:
  g=end[end.arm.eq(a)];rows.append([a,*ARMS[a],fmt(g.ce.mean()),fmt(g.familiar_ce.mean()),fmt(g.gap.mean()),fmt(g.conflict_accuracy.mean()),fmt(g.conflict_ce.mean()),int(g.strong_familiar.eq(1).sum()),int(g.strict_all.eq(1).sum()),int(g.nonfloor_both.sum()),int(g.joint.sum()),int(g.functional_split.sum())])
 text='''# Researcher handoff: sharing versus context-specific learning

Start with Figures3 and4, then read the candidate−C paired intervals. This experiment manipulates the **relative outer-query objective**, using the same eight-unit Linear network, exact source960 model/Adam states and matched task pairs. It is a controlled paired-update fork. Historical single-episode trajectories do not replace the matched control.

'''+md(['Arm','λ','β','C CE','Familiar CE','C gap','Conflict accuracy','Conflict CE','Strong A/B /14','Strict C /14','Both initial blocks /14','Descriptive joint /14','Original gate /14'],rows)+'\n'+md(['Contrast','Outcome','Mean paired effect [95% CI]','Median effect','n'],[[r.contrast,r.metric,ci(r),fmt(r['median']),int(r.n)] for _,r in p.iterrows()])+'''
## Teach-back

Each seven-token forward starts at zero hidden state. The native two recurrent bias tensors remain separate optimizer parameters; their sum is the mathematical recurrence bias. The source manual identity forward is used despite the unused tanh container flag.

Each meta pair asks two questions of the same initialization: after20 practice updates, can it predict a shared sequence in a context not practiced, and can it make different predictions when an identical prefix has different context-dependent continuations? A shared clone is adapted and queried, its graph released; a conflict clone starts from the **unchanged** initialization and adapts separately. Its all-position and critical-position losses use the same query logits. All critical-prefix cells under both assigned contexts count.

The raw first-order gradients combine as `(1−λ)g_share+λ[(1−β)g_all+βg_critical]`, then receive one norm1 clip and one outer Adam step. Each pair costs40 inner updates and two episodes. Original inner Adam.005/CE+.2 execution/jointclip1/fresh policy and inherited outerAdam.001/moments are preserved. No derivative through inner Adam, policy meta-gradient, component normalization or projection is used. A larger conflict weight is the intended manipulation, including any change in raw gradient magnitude before clipping.

All six objective combinations were screened on seeds0–5. The rule, fixed before outcomes, chose A (λ=.25,β=0) and E (.75,0), plus balanced control C (.5,0). No critical-position-weighted arm was selected. Their β effects remain exploratory, not confirmed. The fourteen other seeds start their original960 states, receive1920 new episodes and are evaluated at+960/+1920. Labels count additional episodes: the final outer step is1920, but total FOMAML episode exposure is2880.

Target learning uses the identical seven teacher-forced tokens/next targets,330 A10/B10 updates and fresh policy/Adam. No C or Mixed target practice occurs; the learned C column is fixed. Actual practice curves show pre-update practiced-context CE, blue A/orange B, retaining switches. Frozen A/B losses at identical weights define signed own-minus-cross changes. Positive context specificity during initial blocks is distinct from the unchanged original170/330 endpoint gate. Late near-zero gains cannot establish shared learning; matched bins preserve missing eligibility and actual paired n.

C transfer is measured both absolutely and relative to familiar mean CE. Strong acquisition must hold in each familiar context. A smaller C gap from worse familiar learning is not successful transfer. The inclusive post-acquisition mean retains every state after first strong acquisition, including later deterioration. Never-acquired cases are missing. Mixed-average, Mixed-both and C are different computational probes; only the average is protected by convex CE.

Reserved conflict diagnostics adapt12 strictly unseen tasks for20 updates each, then score the identical ambiguous prefix under the appropriate contexts. Tasks average within seed before inference. Correct-versus-wrong context advantages, own/cross learning and appropriate responses supplement critical accuracy/CE; hidden separation does not substitute for functional learning. Archived+0 target/correct/wrong branches are reused exactly. Only missing wrong-context frozen branches are newly evaluated.

## What the Linear equations tell us

`K_t=W_oΣ(k=0…t−1)W_h^k`, `u=K_t(w_A+w_B)/2`, `v=K_t(w_A−w_B)/2`.

`z_A=z_0+u+v`, `z_B=z_0+u−v`, `z_C=z_0+K_tw_C`.

Mixed-average is `z_0+u`; Mixed-both is `z_0+2u`. Common sequence prediction, contextual contributions and recurrent sensitivity can all change. Their centered norms, margins and cosines describe computation, not causal mediation or separate stored skills. The shared/conflict gradient cosine measures local compatibility at the same initialization. Negative values do not prove the cause of any later behavioral tradeoff. No architecture, context-strength sweep or oracle state rescue was introduced.

## Interpreting the result responsibly

Read the report's eight explicit answers and all14 candidate−control seed differences. The objectives form a multi-outcome intervention, not a post hoc scalar score. C success alone is insufficient; strong familiar acquisition, conflict capability, initial functional specificity, Mixed-both and the original gate remain separate. A zero-spanning interval is inconclusive, not preservation or equivalence. The .1-nat tolerance was only a screening preference. Pointwise bootstrap intervals are not multiplicity-adjusted.

C was seen on other sequences, so target-C success is transfer to a held-out association, not invariance to arbitrary novel sensory cues. These deterministic next-token tasks do not establish human motor behavior, natural retrieval, response times or persistent separate skills.

## Exact accounting and preservation

All78 meta runs:97,920 new episodes,1,958,400 inner updates,48,960 paired outer updates. New120 target histories add39,600 updates and1,440 restored-Adam clone updates. Conflict diagnostics add28,800 clone updates. Total2,077,200 new scientific optimizer steps; implementation fixtures240 inner+6 outer are separate. Twenty original baseline histories/diagnostics are reused, not new replications. There are140 physical target histories; shared baseline references represent the same physical history in each plot.

Every pair retains raw component gradients, adapted prediction weights, query logits/hidden, inner scalar histories and exact outer before/after weights/moments. All saved gradients and outerAdam transitions are independently checked without training replay. Full model/optimizer/globalRNG/registry cursor checkpoints occur every60 paired updates and at all registered endpoints. Inner Adam is not serialized after every practice update. No scientific attempt failed or was silently retried; the retained selection adapter failure happened before confirmation and changed no outcomes/rules.

For safe reproduction, use REPRODUCIBILITY.md and the hash verifier; recreate figures from saved tables into a fresh external directory. Do not rerun scientific training/evaluation to make plots. IMPLEMENTATION_AUDIT.md, CLAIM_LEDGER.csv and SCRATCHPAD.md preserve source hashes, RED/GREEN tests, numerical limits, missing outcomes and every deviation. The separate I/O overhead timings were recorded after launch; the actual prelaunch CPU equivalence/complete-update benchmark is distinguished honestly.
'''
 text=text.replace('## Teach-back',scientific_takeaways()+'## Teach-back',1)
 (OUT/'RESEARCHER_HANDOFF.md').write_text(text)
 sources=['experiments/context_experience_prior/core.py','experiments/heldout_context_meta/fomaml.py','experiments/heldout_context_meta/protocol.py','experiments/heldout_context_meta/target.py','experiments/heldout_context_meta/heldout.py','experiments/heldout_context_meta/analysis.py','experiments/meta_learning_scaling/analysis.py','experiments/recurrent_learning_transfer/measures.py','experiments/heldout_context_meta_scaling_10k/heldout.py']
 audit='''# Implementation, source and reuse audit

The entire completed10k seal and its ancestor inputs/code were verified before this fork. No source file or sealed result is edited. `registry/sources.csv` identifies the20 actual P2-independent Transfer960 bundles, each with172 model parameters and six outerAdam moment/step states. These exact states are reused across every arm within seed. `audit/*_meta_verified.csv` independently validates all saved gradients/Adam updates and all periodic state hashes. `audit/*_targets_verified.csv` validates complete target histories, source matches, fresh optimization state, C-column constancy, position distributions and affine controls. `audit/final_tables_verified.json` checks grids, selection, exclusion masks, paired arithmetic and exact budgets.

## Exact scientific sources

'''+md(['Source file','SHA256'],[[s,sha(ROOT/s)] for s in sources])+'''
## New code map

| Module | Responsibility |
| --- | --- |
| common.py | Isolated namespace, exact source/load adapters |
| protocol.py | Six arms, seed split, encounter-order pair mapping, all ambiguous-prefix cells |
| audit.py | Full ancestor verification, source bundles, paired registry/exclusion seal |
| runtime.py | Unchanged native adaptation, sequential gradients, weighted single clip/Adam |
| train.py | Fixed budgets, compact pair logs, complete60-pair atomic recovery states |
| verify.py | Independent saved-data BPTT/Adam checks and native target verification |
| evaluation.py | Unchanged native target/conflict loops and missing wrong-cue adapters |
| analysis.py/statistics.py | Original functional measures, missing bins, fixed whole6/14-seed inference |
| selection.py | Prespecified screen rule and frozen decision, no favorable seed selection |
| history.py | Fixed60-pair windows, within-seed averaging |
| figures.py | Five main families and compact screening, tables only |
| report.py/packaging.py | Eight answers, exact results, methods, claims and handoff |
| final_checks.py/seal.py | Full grids/budgets/source identities and immutable final evidence |

## Reuse versus new computation

All paired tasks come from exact existing generator libraries, not new outcome-selected sequences. Both arms within every comparison receive identical registered pairs and original P2 stable mappings. The only scientific change is the registered outer-query objective and shared/conflict pairing/one-step aggregation. The matched control uses that same new procedure; historical single-step training is not mislabeled equivalent. Full blocks preserve original role/exposure balance. Target and corrected diagnostic sequence exclusions are verified before training and again after results.

+0 targets, diagnostic adaptation and correct/wrong native predictions reuse the original960 data. `registry/reused_baselines.csv` records exact paths and SHA256; target source rows are also in the screen/confirm physical registries. Every new target keeps all331 states and full six-cue/position predictions. The new diagnostics are exactly20 adaptation updates per task. Critical means are derived from the same conflict-query logits; no extra query pass or inner update is charged as evaluation.

## Restart and retention

Each run saves p0 and complete model/outerAdam/globalTorch/NumPy/PythonRNG/registrySHA/next-pair states every60 pairs. Atomic writes are fsynced and checksummed; compact chunks retain raw gradients and all outer states so every committed update is auditable. Required task input order is already sealed, not dynamically resampled. A scientific run refuses an existing started attempt: an interruption requires an explicit audit of committed and uncommitted computation before an authorized recovery, not silent extra optimization. No interruption occurred in the completed run. Inner Adam states are intentionally not serialized after every adaptation step, as requested.

## Tests and retained deviations

RED/GREEN evidence precedes accepted scientific features. Independent simultaneous-two-graph and sequential updates agree before training, including weighted raw gradients/one clipping step/inherited moments. CPU1/2 final states/gradients are bitwise equal. Actual accelerator availability is false. All critical masks/role schedules, no-C target schedule, source/optimizer/indexing, cohort/selection and signed/missing inference checks remain in audit/. Thirty-one predecessor read-only regression tests pass; optimizer fixtures are excluded from repeats.

The initially drafted saved-data verifier was backed out before its dedicated RED15/GREEN16 acceptance; this chronology is explicit, not falsely represented. Selection stage27 rejected descriptive Baseline rows as a seventh arm and stopped before writing any decision or running confirmation; RED31/GREEN32 filter the registered+480 A–F inputs without changing tables/rules. Separate I/O/clone timing was missing from the prelaunch timing decomposition; a read-only followup fills that measurement and is labeled postlaunch. No scientific settings, budgets or criteria changed to rescue any result. Font-cache warnings are retained in logs and do not alter numerical operations.
'''
 (OUT/'IMPLEMENTATION_AUDIT.md').write_text(audit)
 reused=[]
 for seed in range(20):
  td=SOURCE/f'target/s{seed:02d}_P2_I_Transfer_d960';hd=SOURCE/f'heldout_diagnostics_v2/s{seed:02d}_P2_I_Transfer_d960';wd=SCALE/f'conflict_wrong_cue/s{seed:02d}_P2_I_Transfer_d960';row=dict(seed=seed)
  for name,path in [('target_states',td/'states.pt'),('target_evaluations',td/'evaluations.csv'),('diagnostic_states',hd/'adapted_states.pt'),('diagnostic_positions',hd/'positions.csv'),('wrong_cue_cells',wd/'cells.csv')]:row[name]=str(path.relative_to(ROOT));row[name+'_sha']=sha(path)
  reused.append(row)
 write_frame(OUT/'registry/reused_baselines.csv',reused)
 ledger=[
 ('C01','Exact source and shared within-seed initialization','registry/sources.csv; audit/predecessor_verified.json; audit/*_meta_verified.csv','All source960 models/moments; controlled fork, not historical exact continuation'),
 ('C02','Paired gradient and critical-query computation','meta/*/chunk_*.npz; audit/05_GREEN_paired_gradient.log; audit/*_meta_verified.csv','Two queries, same meta initialization, single weighted clip/Adam; all ambiguous-prefix cells'),
 ('C03','Matched task ecology and exclusions','registry/pairs_seed_*.json; registry/heldout_v2_seed_*.json; registry/block_balance.csv; audit/registry_seal.json','Exact sealed original libraries reused; full-path novelty, common subsequences allowed'),
 ('C04','Exploratory selection and untouched confirmation split','SCREENING_DECISION.md; audit/selection_seal.json; tables/screen_candidate_summary.csv','Screen0–5, confirmation6–19; neither beta arm selected; no pooled validation'),
 ('C05','C loss/gap and familiar acquisition','tables/confirm_seed_outcomes.csv; tables/confirm_paired_intervals.csv; tables/confirm_evaluations.csv','Separate absolute/relative outcomes; no equivalence inferred from nonsignificance'),
 ('C06','Post-acquisition transfer','tables/confirm_acquisition.csv; tables/confirm_paired_seed_effects.csv','All states from first acquisition through330, later deterioration retained; never-acquired missing'),
 ('C07','Functional first-block and ongoing learning','tables/confirm_blocks.csv; tables/confirm_learning_effects.csv; tables/confirm_switch_costs.csv','Signed own/cross effects, actual starts/floors and restoredAdam; geometry not a skill gate'),
 ('C08','Conditional matched acquisition','tables/confirm_matched_acquisition.csv; tables/confirm_matched_intervals.csv','Missing eligible bins preserved, actual paired n; conditional not causal mediation'),
 ('C09','Unseen conflict-dependent capability','tables/confirm_heldout_seed.csv; heldout_diagnostics_v2/*/positions.csv; heldout_diagnostics_v2/*/blocks.csv','Twelve tasks averaged within seed;20 fixed updates; critical accuracy/CE/appropriate responses separate'),
 ('C10','Correct and wrong context effects','tables/confirm_wrong_cue_seed.csv; conflict_wrong_cue/*/cells.csv; registry/reused_baselines.csv','Correct branches reused; only missing wrong native predictions evaluated; no added optimization'),
 ('C11','Joint pattern and original gate','tables/confirm_seed_outcomes.csv; tables/confirm_functional_splitting.csv','Descriptive first20 joint differs from unchanged endpoint gate; no posthoc scalar success score'),
 ('C12','Gradient compatibility','tables/confirm_meta_history.csv; tables/confirm_meta_windows.csv; tables/confirm_gradient_seed_summary.csv','Local raw gradient cosine, tasks/windows not independent networks; no interference-causality claim'),
 ('C13','Linear contextual mechanism','tables/confirm_geometry.csv; tables/confirm_mechanism_intervals.csv; target/*/decomposition.csv.gz','Centered magnitudes/margins describe computation; no causal mediation or oracle rescue'),
 ('C14','Exploratory beta and lambda factorial','tables/screen_factorial_seed_effects.csv; tables/screen_factorial_intervals.csv','Six-seed screening only; no significance-based confirmation claim'),
 ('C15','Complete accounting and immutable evidence','audit/final_tables_verified.json; registry/reused_baselines.csv; audit/benchmark.json; SCRATCHPAD.md','2,077,200 scientific steps plus246 fixtures; failures/deviations retained; no hidden rerun')]
 write_frame(OUT/'CLAIM_LEDGER.csv',[dict(claim_id=i,claim=c,exact_sources=s,limits=l) for i,c,s,l in ledger])
 (OUT/'REPRODUCIBILITY.md').write_text('''# Reproduce evidence without new scientific training

From repository root in the recorded Python3.11/Torch2.8/NumPy2.2.6/Matplotlib/font environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m experiments.meta_objective_balance.seal --verify
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/meta_balance_mpl python -m experiments.meta_objective_balance.figures --stage screen --output-dir /tmp/meta_balance_recreated_figures
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/meta_balance_mpl python -m experiments.meta_objective_balance.figures --stage confirm --output-dir /tmp/meta_balance_recreated_figures
```

Choose a fresh destination outside the sealed package. Verification reads hashes only; plotting reads saved tables only, no checkpoint load, gradients, optimization or new statistical inference. Fixed SVG hash salt/metadata and the original plotting environment permit byte equality; `audit/figure_regeneration.json` records the completed comparison. The CLI refuses writing inside the sealed namespace after manifest creation.

Do not run train/evaluation/benchmark or the scientific pipeline to reproduce figures. Every scientific optimizer update is already preserved and counted. Post-seal verification receipts belong outside this package, in `scratchpads/meta_objective_balance_seal.log`.

## Exact archive layout

- `registry/sources.csv`:20 original P2-independent Transfer960 source bundle paths/SHA256, actual source weights and inheritedAdam step/moments. `parent_hashes.json` binds the full existing10k evidence and all ancestors. Previous programs are read-only.
- `registry/pairs_seed_*.json`:all matched shared/conflict pairs and original support/query schedules. Six screen seeds use240pairs, fourteen confirmation seeds960. Every objective arm uses the same seed-specific library. Role balances/sourceblock hashes and reserved corrected diagnostic paths are explicit.
- `meta/sXX_ARM/p0.pt,p60.pt,…`:complete atomic prediction-model/outerAdam/globalRNG/registry-hash/cursor restart bundles, with fixed60-pair cadence. Registered final/intermediate states are included. +1920 episodes uses p960, not p1920. JSON sidecars bind snapshot/chunk hashes.
- `meta/*/chunk_*.npz`:all raw shared/all-conflict/critical-conflict query gradients, adapted prediction weights, query logits/hidden, critical masks, each inner CE/execution/norm scalar, and outer before/after parameters/moments for every pair. No per-inner-update Adam-state serialization. Manual BPTT/Adam validation covers every pair without replaying optimization.
- `meta/*/episodes.csv`:one compact numerical record per paired outer update, losses/norms/cosine/clipping/reset/forward/budget checks. Training histories are averaged within seed in fixed60-pair windows before intervals.
- `target/*/states.pt`:all331 model/policy/Adam states for120 new targets, source snapshot SHA and fixed C column. Six-cue evaluations/training/diagnostic scalars and full seven-position logits/probabilities/hidden/decompositions/numerical bounds remain beside each bundle.
- `registry/reused_baselines.csv`:all20 exact original+0 target, corrected conflict and wrong-context paths/hashes. These are reused, never independent new replications. Physical screen42 and confirmation98 histories total140 distinct histories.
- `heldout_diagnostics_v2/*/`:120 new diagnostic initializations,12 reserved tasks each,20 native adaptation updates/task, packed inner histories/adapted states, correct predictions and signed own/cross blocks. `conflict_wrong_cue/*/` contains only missing frozen wrong-context before/after branches; archived matches are not rerun.
- `tables/screen_*`:six-seed exploratory outcomes/factorial estimates and compact histories. `SCREENING_DECISION.md` and its seal freeze A/E plus control C before confirmation starts.
- `tables/confirm_*`:fourteen-seed raw outcomes, all331-state curves, signed first/ongoing block effects, same-weight switches, original gate, matched-bin eligibility, position errors, geometry, joint flags, paired mean/median effects/intervals and exact plotting values. Missing bins remain missing; neither tasks nor repeated checkpoints replace independent seeds.
- `audit/`:all RED/GREEN logs, exact source/registry/decision hashes, native/analytic/full-grid/budget verification, retained selection-adapter failure, benchmark and postlaunch I/O timing limitation, figure byte comparison and final review.

## Controlled recovery, not silent extra computation

The source, periodic model/Adam/globalRNG state, sealed library SHA and next-pair index fully specify a continuation. Recovery requires verifying the saved code/source/library identities and accounting for committed and any uncommitted attempted work. The runner refuses an existing scientific attempt by default; it never silently resets optimizer moments or repeats a partial run. No training interruption occurred in this execution. Default reproduction uses hashes and tables only. Do not overwrite a predecessor seal or tune a weight/threshold/budget after an unfavorable outcome.
''')
 for name in ['RESEARCHER_HANDOFF.md','IMPLEMENTATION_AUDIT.md','REPRODUCIBILITY.md']:
  path=OUT/name;path.write_text(polish(path.read_text()))
 dump(OUT/'audit/products_complete.json',dict(passed=True,report=True,handoff=True,implementation_audit=True,reproducibility=True,claim_rows=15,reused_baseline_hash_rows=20));append('## Researcher products complete\nSource-to-claimledger,actualalgorithmteachback,explicitnon-equivalence/causalitylimits,sourcehash/reusemapandtable-onlyreproductiondirectionscomplete. Allscreen/confirmoutcomesandfailurespreserved.')
if __name__=='__main__':main()
