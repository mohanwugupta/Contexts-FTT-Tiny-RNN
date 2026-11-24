"""Researcher handoff, source-to-claim ledger and safe reproduction directions."""
import re
from .common import *
from .report import table,fmt,ci,interpret_interval,QUESTIONS,scientific_takeaways

def main():
 primary=read('primary_intervals');end=read('endpoints');final=end[end['update'].eq(330)&end.cue.eq('C')&end.dose.eq(10000)];joint=read('joint_early20');joint=joint[joint['update'].eq(330)&joint.dose.eq(10000)];rows=[]
 for key,g in final.groupby(['condition','arm']):
  j=joint[(joint.condition==key[0])&(joint.arm==key[1])];rows.append([*key,fmt(g.ce.mean()),fmt(g.familiar_ce.mean()),fmt(g.gap.mean()),fmt(g.gap.median()),int(g.strong_familiar.eq(1).sum()),int(g.strict_all.eq(1).sum()),int(j.early20_coexistence.eq(1).sum()),int(j.functional_split.eq(1).sum())])
 pr=table(['Independent family','Primary contrast','Paired gap change [95% CI]','Median change','Improved seeds','n'],[[r.condition,r.contrast,ci(r),fmt(r['median']),int(r.n_negative),int(r.n)] for _,r in primary.iterrows()])
 text='''# Researcher handoff: held-out-context meta-learning to10,000

Start with Figure3 and the four registered primary contrasts below. Negative gap effects favor Transfer or additional training. Means, medians and negative-change counts must be read together. A negative-change count includes any negative sign, without a minimum practical magnitude. The full report answers all nine PRD questions; no favorable dose was selected.

'''+pr+'\n'+table(['Family','Arm','Final C CE','Familiar CE','Mean C gap','Median C gap','Strong familiar /20','Strict C /20','Early20 descriptive joint /20','Original endpoint gate /20'],rows)+'''
## Teach-back: what actually learned

These are eight-unit identity RNNs predicting the next token, not free-running sequence generators. They start from the same episode960 FOMAML checkpoints that followed1920 Reptile episodes. All160 trajectories receive9040 additional episodes, reaching10000 FOMAML episodes. The existing outer Adam moments and step counters continue; they are not reset. Seen and Transfer already differ at960 because their earlier objectives differ; they share the original Reptile parent atFOMAML0. The final arm contrast includes this cumulative history. The secondary within-seed scale interaction subtracts the960 arm difference.

Each episode copies the meta-model, starts fresh inner Adam and execution policy, and practices for20 updates with the original CE+0.2 execution loss, Adam0.005 and joint norm1 clipping. Then query CE is differentiated on the adapted prediction model. Those gradients update the initialization through persistent outer Adam0.001 with norm1 clipping. This is first-order MAML, not an extra query practice step or Reptile interpolation. There is no optimizer Jacobian or meta-learned policy.

On shared tasks, Transfer queries a context absent from support; Seen includes that query context during support. Both arms query the same sequence/cue. Support membership necessarily differs, while full-block marginal exposure is balanced. Conflict episodes are identical across arms: P2 has stable context assignments, P1 balanced ambiguity. The task distribution, original code operations, parameter precision and update order are unchanged apart from reserving the already corrected unseen diagnostic sequences as required.

At the five prescribed new doses,800 fresh target runs learn the original sequence through330 A10/B10 updates. Target policy/Adam reset, C never receives target practice, and its input column remains bitwise fixed. All331 states and six-cue position predictions are saved. Historical0/240/960 data contribute400 reused physical trajectories, never extra replications.

## How to interpret learning and transfer

`G_C(t)=CE_C(t)−(CE_A(t)+CE_B(t))/2`. Inspect absolute C and familiar loss separately. A smaller gap caused by worse familiar learning is not successful transfer. Post-acquisition gap includes every state from first strong A/B acquisition through330, retaining later deterioration; never-acquired cases are missing.

A block's learning specificity is practiced-context CE decrease minus unpracticed-context decrease, both measured between identical frozen start/end weights. Positive signs mean preferential learning. The first20 table retains every actual start/end A/B loss. First-block descriptive coexistence requires positive own gain and specificity in each of the A0→10 and B10→20 blocks, outside the inherited floor flag, along with familiar/C/Mixed-both success. This is not the original endpoint gate. A failed descriptive flag can reflect a floor-limited B block after fast A-to-B learning, rather than loss of contextual capability. Near-zero effects cannot establish shared learning or preserved specialization.

Fixed CE bins2–4,1–2,0.5–1,0.1–0.5 retain the first eligible block, separate A/B starting losses and missing eligibility. Conditional matched comparisons are not randomized mediation. Restored-Adam probes at20/40/80/120/170/330 copy full model/policy/moments, then discard the A-only and B-only updates. The original170/330 splitting criterion and thresholds remain unchanged.

Reserved conflict tasks directly test whether identical token prefixes yield appropriately different predictions under different contexts after20 adaptation updates. Twelve tasks are averaged within each seed before inference. Correct branches are reused, while only missing wrong-context inputs are evaluated on frozen saved task models. These diagnostics do not tune the training curriculum or prove persistent original A/B skills.

## The exact Linear computation

With `K_t=W_oΣ(k=0…t−1)W_h^k`, the logits satisfy

`z_A=z_0+u+v`, `z_B=z_0+u−v`, `z_C=z_0+K_t w_C`,

`u=K_t(w_A+w_B)/2`, `v=K_t(w_A−w_B)/2`.

Hence `z_C−z_Mixed-average=K_t[w_C−(w_A+w_B)/2]`, while `z_Mixed-both=z_0+2u`.

Baseline sequence computation, C's contextual response and the familiar common contribution can all change. Their norms and correlations do not identify causal mediation. C need not equal the A/B logit midpoint to predict correctly. Only Mixed-average has the familiar convex CE bound; C and Mixed-both do not. No context-strength sweep or oracle state optimization was run.

## Statistical and scientific limits

Primary independent P1/P2 comparisons each have20 paired seeds. All repeated doses/arms use10000 fixed whole-seed bootstrap resamples; medians and individual effects accompany means. Pointwise intervals are not equivalence or multiplicity-adjusted tests. Missing bins are not zero effects. Initially centered means the original organization, not a permanent parameter constraint.

C was an experienced prior-task context paired here with a held-out target sequence. Successful transfer is not proof of general robustness to arbitrary unseen sensory cues. The experiment does not fit human motor behavior or response times. Interpret the observed joint component rates, functional floors and conflict capability before making a human-like learning claim.

## Integrity, storage and restart

The20 seed-specific task registries are each shared across eight trajectories (P1/P2 × I/C × Seen/Transfer). Thus180800 executed unique seed-task slots correspond to1446400 actual meta episodes;1600 unused generated slots remain after the final prefix. This is shared input design, not a reduced training budget.

Full predecessor hashes and all160 exact model/Adam sources were verified before training. Task RNG is keyed by seed/block; exact original960 and final10000 consumed RNG states are recovered in `audit/task_rng_cursor_states.json`. Global RNG snapshots and sealed registry cursors are saved with new restart states. Updates themselves contain no random draw, as verified by the clone benchmark.

Complete blocks are exactly balanced. The final160-slot prefix is not rebalanced, and its remaining80 registered tasks are preserved but untrained. Their realized counts are in block_summary.csv.

The predecessor packed every inner/outer tensor trajectory. This scaling run retains compact per-episode query/inner records and full traces for a fixed44-episode subset per trajectory, with an atomic durable restart state after every completed episode and full registered checkpoints. Independent algebra verifies the saved query gradients and Adam updates on that subset; it does not claim every unretained intermediate tensor was independently audited. A transaction committed before checkpoint export can be recovered without optimizer replay. An interrupted, uncommitted computation must be disclosed and audited rather than silently counted as completed or rerun as if no extra attempt occurred.

A table-only Boolean-accuracy export error was corrected without changing any of the42 base tables, scientific inputs, thresholds or models. The failed pass, regression test and recovery receipts remain in audit/.

Scientific budget:28,928,000 inner +1,446,400 outer +264,000 target +9,600 target-clone +115,200 conflict-clone updates =30,763,200 new optimizer steps. Separate implementation fixtures used180 inner +9 outer updates. No invalid predecessor diagnostic run was repeated or counted as new evidence.

Read `REPRODUCIBILITY.md` for hash verification and table-only figure regeneration. Do not rerun scientific training to recreate plots. `CLAIM_LEDGER.csv` and `SOURCE_REUSE_AUDIT.md` map claims and sources; all failures, negative effects and uncertainties remain in the tables and append-only scratchpad.
'''
 from .report import prose_spacing
 text=text.replace('## Teach-back: what actually learned',scientific_takeaways()+'## Teach-back: what actually learned',1).replace('A failed descriptive flag can reflect a floor-limited B block after fast A-to-B learning, rather than loss of contextual capability.','The first20 floor flag is zero for every final-dose seed, so measured early floors do not explain missing descriptive flags. Late probes remain floor-limited.')
 (OUT/'RESEARCHER_HANDOFF.md').write_text(prose_spacing(text))
 ledger=[
 ('S01','Exact960 continuation and all160 conditions','registry/sources.csv; audit/predecessor_verified.json; audit/protocol_seal.json; meta/*/e*.pt; audit/meta_verified.json','OuterAdam moments/steps inherited; no global RNG is needed by deterministic update; exact keyed task stream audited'),
 ('S02','Same task distribution and heldout reservation','registry/seed_*_b*.json; registry/heldout_v2_seed_*.json; registry/block_summary.csv; audit/task_rng_cursor_states.json','Full blocks balanced; final160 prefix retains realized imbalance; common subsequences allowed'),
 ('S03','FOMAML and storage fidelity','audit/04_GREEN_continuation_storage.log; audit/benchmark.json; audit/early_saved_gradient_audit.json; audit/meta_verified.json; meta/*/journal.sqlite; meta/*/audit_e*.pt','Same scientific operations; compact storage differs from predecessor; independent algebra checks fixed subset, not all missing intermediates'),
 ('S04','Primary scaling benefit or failure','tables/primary_seed_effects.csv; tables/primary_intervals.csv; tables/endpoints.csv','Four prespecified independent comparisons; all20 paired seeds; distinguish mean/median/heterogeneity and familiar loss'),
 ('S05','Empirical dose response and terminal changes','tables/scaling_seed_effects.csv; tables/scaling_intervals.csv; tables/endpoint_summary.csv; tables/gap_median_intervals.csv','No monotonicity or power law assumed; zero-spanning terminal interval is not evidence of plateau/equivalence'),
 ('S06','Functional learning during acquisition','tables/early_updates.csv; tables/block_transfer.csv; tables/early20_seed.csv; tables/early20_intervals.csv; tables/switch_costs.csv','Signs retained; first20 baseline/end losses explicit; both-context CE≤.05 is floor flag, not shared-learning evidence'),
 ('S07','Matched-acquisition specificity','tables/matched_acquisition.csv; tables/matched_scaling_seed_effects.csv; tables/matched_scaling_intervals.csv','Fixed bins, first eligible either direction and supplemental A/B; missing eligibility and paired n retained; conditional not causal'),
 ('S08','Original functional and descriptive coexistence','tables/learning_effects.csv; tables/functional_splitting.csv; tables/joint_early20.csv; tables/joint_early20_summary.csv','Original170/330 criterion unchanged; first20 and legacy broader coexistence are separate descriptions'),
 ('S09','Reserved unseen conflict capability','registry/heldout_diagnostics_v2.csv; tables/heldout_seed.csv; tables/heldout_paired_intervals.csv; heldout_diagnostics_v2/*/tasks.csv','Only corrected-v2 library; all12 tasks averaged within seed; identifiable mappings for both priors; not model selection'),
 ('S10','Correct-versus-wrong context effects','registry/conflict_wrong_cue.csv; conflict_wrong_cue/*/cells.csv; conflict_wrong_cue/*/positions.csv.gz; tables/heldout_wrong_seed.csv','Correct branches reused; initial correct CE reconstructed from all block gains; only missing wrong branches evaluated; no optimizer update'),
 ('S11','Exact affine mechanism and positional errors','target/*/decomposition.csv.gz; tables/geometry.csv; tables/C_positions.csv; tables/position_scaling_intervals.csv; tables/mechanism_scaling_intervals.csv','All7 positions and raw/centered terms; inherited numerical bounds; norms/correlations are not causal mediation; no strength sweep'),
 ('S12','Post-acquisition residual disadvantage','tables/acquisition.csv; tables/post_acquisition_curve.csv; tables/post_scaling_seed_effects.csv; tables/post_scaling_intervals.csv','First pass inclusive through330; later worsening retained; never-acquired missing/censored'),
 ('S13','Complete sources, grids and optimizer budgets','audit/final_tables_verified.json; tables/physical_run_registry.csv; audit/meta_complete.json; audit/targets_complete.json; audit/heldout_complete.json','1200 physical target histories,400 reused;30,763,200 new scientific updates;189 fixtures separate'),
 ('S14','Meta-query and conflict training histories','tables/meta_query_seed.csv; tables/meta_query_summary.csv; meta/*/episodes.csv; meta/*/journal.sqlite','Fixed cumulative/last240 windows; episodes averaged within seed; historical Reptile query undefined; no selection')]
 write_frame(OUT/'CLAIM_LEDGER.csv',[dict(claim_id=i,claim=c,exact_sources=s,limitations=l) for i,c,s,l in ledger])
 (OUT/'REPRODUCIBILITY.md').write_text('''# Reproduce the completed experiment without retraining

From repository root in the recorded Python/Torch/NumPy/Matplotlib/font environment:

```sh
OPENBLAS_NUM_THREADS=1 python -m experiments.heldout_context_meta_scaling_10k.seal --verify
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/heldout_10k_mpl python -m experiments.heldout_context_meta_scaling_10k.figures --output-dir /tmp/heldout_10k_recreated_figures
```

Choose a fresh destination outside the sealed package. Verification reads hashes only. Figure regeneration reads CSVs only and performs no checkpoint evaluation, gradients or optimizer steps. Native prediction tables, bootstrap indices and SVG hash salt are fixed; the original plotting environment permits byte comparison. `audit/figure_regeneration.json` records the final comparison. Do not run train, run_targets, heldout, benchmark or pipeline to recreate figures.

## Exact evidence map

- `registry/sources.csv`:160 actual predecessor960 bundles with SHA256, original model weights, six outerAdam states/counters and settings.
- `registry/seed_*_b*.json`:760 complete blocks5–42 across20 seeds, matched both arms/priors/organizations. All rejected duplicate and reserved-path candidates remain. Only9040 episodes per trajectory are executed; final80 slots are unused.
- `registry/heldout_v2_seed_*.json`:exact corrected predecessor diagnostics,24 individually unseen full8-symbol paths/seed, reserved before new generation. Prefixes/subsequences remain allowed.
- `audit/task_rng_cursor_states.json`:exact recovered predecessor960 and final10000 PCG64 consumption state. The final block is generated ahead of consumption, so its stored full generation-end state differs from cursor160. Both are explicit; continue remaining registered slots before generating another block.
- `meta/*/journal.sqlite`:checksummed compact records for all9040 new episodes and the latest complete durable model/outerAdam/global-RNG/task-cursor state. Each episode atomically commits its summary and restart state together. Registered checkpoint blobs are retained in the same transaction.
- `meta/*/e{1920,3840,5760,7680,10000}.pt` and JSON:complete restartable prediction model, outerAdam, global torch/NumPy/Python RNG and task-registry cursor/hash, with SHA256 and inherited step counts.
- `meta/*/audit_e*.pt`:full21-state inner prediction/policy/Adam traces on44 prespecified episodes per run. Also before/adapted/after prediction parameters, raw query gradients/logits/hidden, and outerAdam before/after. This is deliberately a subset, unlike the predecessor's fully packed all-episode histories.
- `meta/*/episodes.csv`:portable scalar index for every new episode, kind, inner losses/norms, query CE/norm, update norm, Wh spectral norm and memory high-water mark. Full20 inner scalar records/raw query arrays remain in SQLite. Existing first960 histories are source-linked, not repeated.
- `target/*/states.pt`:all331 model/policy/Adam states for800 new target runs. Training/evaluation/position/decomposition/check shards retain every repetition, six cues, all7 positions, logits/probabilities/hidden, numerical bounds and restored-Adam probes.
- `tables/physical_run_registry.csv`:1200 physical target histories, including400 exact0/240/960 reused histories with original paths/hashes. Historical0 is one baseline, even when referenced in both arm plots.
- `tables/primary_seed_effects.csv` and `primary_intervals.csv`:four fixed independent primary comparisons, each20 paired seeds. `scaling_*`, `matched_scaling_*`, `post_scaling_*`, `position_scaling_*` and `mechanism_scaling_*` retain other prespecified comparisons, signed individual effects and conditional n.
- `tables/early_updates.csv`:every first20 frozen start/end A/B CE and signed own/cross effect. `block_transfer`, `switch_costs`, `learning_effects`, `functional_splitting` retain original definitions and thresholds. `joint_early20` explicitly separates first-block coexistence, strict-C coexistence, legacy broader coexistence and the original endpoint gate.
- `heldout_diagnostics_v2/`:480 new20-step task-clone evaluations at3840/7680/10000; historical/960 records reused. Each run has12 tasks, complete adapted states, packed inner traces, own/cross block gains and position logits/probabilities.
- `conflict_wrong_cue/`:only previously missing frozen wrong-context branches before/after task adaptation; correct branches reused and their initial CE reconstructed from the saved four-block gain sums. No optimizer update.
- `tables/meta_query_*`:cumulative and last240 query summaries at fixed doses, actual shared/conflict task counts and20-seed intervals. Historical Reptile has no comparable query objective.

## Restart and exact replay

The update code is deterministic and source operations/precision are unchanged. A registered checkpoint plus sealed remaining task library and cursor is sufficient to replay model/Adam updates exactly; unused final slots are not resampled. New global RNG snapshots are stored, although the episode update itself has no stochastic operation. Constructor RNG is forked/restored. The inherited required task stream is recovered exactly; no unavailable required state is approximated.

A restart first verifies state checksums, run/source identity, exact next episode and code hashes. SQLite commits prevent duplicate or skipped *committed* episodes. A crash after commit but before export recovers the registered blob without optimizer replay. If an interruption occurs during an uncommitted computation, record that incomplete attempt and audit it; do not silently claim that no extra computation occurred. The completed run's actual interruption/failure record and budgets are in the scratchpad/audit. Default reproduction does not execute scientific updates.

The full-hash verifier is safe after sealing. Post-seal verification/regeneration receipts belong outside this package, in `scratchpads/heldout_context_meta_scaling_10k_seal.log`. Do not overwrite predecessor manifests or outputs, add data-dependent thresholds, select checkpoints using C/Mixed, or rerun the prior context-strength sweep.
''')
 with (OUT/'SOURCE_REUSE_AUDIT.md').open('a') as f:f.write('''\n## Completed execution and validation\n\nAll registered source/task hashes remain bound by the source and final manifests. `audit/meta_verified.json` checks all episode summary hashes, budgets/objective/reset flags and every restart/checkpoint state; independent BPTT/Adam algebra is checked on the fixed full-trace subset, with inherited tolerances. `audit/targets_verified.json` checks all800 new histories, every position distribution/logit/margin, source match, C constancy, fresh/restored optimizer states and exact mathematical decompositions. `audit/final_tables_verified.json` verifies all1200 physical histories and exact old-data reuse.\n\nThe exact task-RNG recovery separately reproduces predecessor block4 and new block42 accepted/rejected libraries. It records both the full ahead-of-time generation state and the consumed prefix160 state. No model optimizer is called in that audit.\n\nNew scientific compute is30,763,200 optimizer steps, with180 inner+9 outer implementation-fixture steps separate. A table-only analysis extension initially failed on Boolean accuracy subtraction. RED64/GREEN65 established signed numeric0/1 subtraction, with NaNs retained. Recovery resumed from the42 saved base tables and verified every original file hash unchanged. The failed log/blocker and recovery receipts remain. No scientific training or native evaluation was repeated. The old invalid diagnostic v1 is neither repeated nor used as valid evidence. All new target/diagnostic outcomes remain unselected. No context-strength sweep, new architecture or hyperparameter tuning occurs.\n''')
 for name in ['REPRODUCIBILITY.md','SOURCE_REUSE_AUDIT.md']:
  doc=OUT/name;doc.write_text(prose_spacing(doc.read_text()))
 for name in ['FINAL_HELDOUT_CONTEXT_META_SCALING_10K_REPORT.md','RESEARCHER_HANDOFF.md','REPRODUCIBILITY.md','SOURCE_REUSE_AUDIT.md']:
  txt=(OUT/name).read_text();assert '<bound method' not in txt
  for link in re.findall(r'\]\(([^)]+)\)',txt):
   if '://' not in link:assert (OUT/link.split('#')[0]).exists(),(name,link)
 dump(OUT/'audit/products_complete.json',dict(passed=True,nine_questions=True,claim_ledger=True,all_links_valid=True,source_audit=True,no_scientific_reruns=True))
 append('## Researcher products\nReportanswersninePRDquestions;handoffteachesactualFOMAML,pairedscaling,first20/matchedfloorinterpretation,unchangedgateandLinearidentities. Claimledgerandprecision/source/restart/RNG/storageguidecomplete. Everyresultlinksactualsavedtables/checkpoints;allnegative/inconclusiveandmissingoutcomesremain.')
if __name__=='__main__':main()
