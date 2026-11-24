"""Source-to-claim map, observed-result handoff and final product receipts."""
import re
from .common import *
from .report import ci,table,fmt,polish

def main():
 ps=read('paired_intervals');joint=read('joint_outcomes');ep=read('endpoints');counts=[]
 for c in CONDITIONS:
  z=ep[(ep.condition==c)&(ep.dose==1920)&(ep['update']==330)&(ep.cue=='C')];jj=joint[(joint.condition==c)&(joint.dose==1920)&(joint['update']==330)];counts.append([c,int(z.strong_familiar.sum()),fmt(z.ce.mean()),int(z.success.sum()),int(jj.functional_split.sum()),int(jj.joint_original.sum()),int(jj.nonfloor_coexistence.sum())])
 effects=ps[(ps.organization=='I')&(ps.dose==1920)&(ps['update']==330)&(ps.cue=='C')&(ps.metric=='ce')&(ps.contrast.isin(['P1:1920-240','P2:1920-240','P2-P1','change_P2-P1:1920-240']))]
 text='''# Researcher handoff — scaling context-experience priors

Read `FINAL_META_LEARNING_SCALING_REPORT.md` and the five figure families first. The prior experiment remains immutable. This is an extension of its80 P1/P2 runs, not a new architecture or random-restart cohort. I/C suffixes mean initial A/B organization; cueC is a separate pretrained input channel.

## Actual final outcomes

'''+table(['Condition','Strong familiar/20','C CE','C success/20','Original functional/20','Original joint/20','Descriptive earlier coexistence/20'],counts)+'\n'+table(['Primary independent contrast','C CE effect [95% CI]','Paired n'],[[r.contrast,ci(r),int(r.n)] for _,r in effects.iterrows()])+'''
Negative C CE effects favor the first/higher-dose condition. Intervals spanningzero are inconclusive for benefit, not proof of equivalence. Binary original splitting and descriptive nonfloor coexistence must not be interchanged.

## Teach-back: what actually ran

The eight-unit identity RNN predicts the next token at each of seven teacher-forced positions. We reuse original model/objective/Adam/clip code. Source240 metaweights initialize episode241; Reptile epsilon stays0.01 through1920. Every task gets fresh Adam and execution policy,20 updates, then prediction-only interpolation. No task restarts from the previous adapted weights. The first240 episodes are not rerun, and no recurrent renormalization occurs.

At eachfixed dose, target A10/B10 starts with fresh Adam/policy and runs330 updates. No C/Mixed target loss. The learned C column receives zero gradient and zero fresh momentum and remains bitwise unchanged. All331 fullstates, allsix cues and rawpositions are saved. Dose0 and240 histories are exact reuse, not extra replications. Fullnewbudget:2,688,000 innerupdates,134,400 Reptile moves,184,800 targetupdates,6,720 isolated diagnosticupdates; no gradient-based implementation fixtures.

## Equations and interpretation

With zero h0, S_t=ΣW_h^k and K_t=W_oS_t. zA=z0+u+v; zB=z0+u−v; zC=z0+K_twC; Mixed-average=z0+u; Mixed-both=z0+2u. Only averageMixed CE is convex-bounded by familiar CE; accuracy, C andBoth are not. Class-centered K/u/v norms are descriptive, not mediation or skill-count tests. A/B hidden separation does not establish independent skills.

Original functional splitting is unchanged at170/330. Source floor=both block-start CEs≤0.05. Matched-acquisition bins use first available10-update-block start with meanfamiliarCE in the fixed ranges; actualA/BCE and practicedcue are retained. Mainfirstblock/eitherdirection and supplemental firstA/firstB are conditional, not randomized causal effects. Missingranges remain missing; never treat vanishing floor gains as proof of sharedlearning. The historical secondary trajectory gate is unassessed because diagnostic times differ.

The random-input control uses20 unitdirections perpairedseed, fixedacrossdoses/conditions, normmatched to eachnetwork's learned C column. Averagevectors withinseed before inference. Normmatching changes absolute magnitude acrossdoses; it cannot by itself isolate recurrent robustness from C-norm changes. Hypothetical8-dimensional random drives need not be realizable by the three existing cue channels; these probe input-drive robustness rather than natural access. NaturalC is prior-experienced but never trained with the target sequence; it is not a cue never encountered before.

## Evidence navigation and limits

`physical_run_registry.csv` identifies680 unique targethistories;720 plotted/model references include shared dose0 baselines. `matched_acquisition.csv` includes missing eligibility; `paired_seed_effects` and `matched_seed_effects` preserve withinseed contrasts. `joint_outcomes` distinguishes originaljoint from earlier-coexistence and unavailabledata. `geometry` and rawdecomposition shards retain the exact Linear contributions. StageA through240 was analyzed and verified before continuation; do not select a favorable dose retrospectively.

No human data were fitted. One task ecology and architecture, finite20seeds, performance floors and P1/P2 temporal-order differences limit inference. The first240 outer learningrate decays and later epsilon is fixed asregistered; the fullcurve is not assumed to be a scalinglaw. Reproduce from saved tables/checkpoints without scientifictraining; see REPRODUCIBILITY.md and the manifest verifier. Preserve allnull/negative/nonfinite results and original thresholds.
'''
 (OUT/'RESEARCHER_HANDOFF.md').write_text(polish(text))
 ledger=[('S01','Allnine doses/all20pairedseeds; exactreuse','registry/reuse_registry.csv; tables/physical_run_registry.csv; audit/protocol_seal.json','Dose0 shared across P1/P2; physical680,logical720; no extra replications'),('S02','Deterministic matched tasks/no leakage/duplicate rejection','registry/seed_*_block_*.json; registry/continuation_summary.csv; registry/duplicate_rejections.csv','Temporal sequenceorder differences remain; P1 ambiguity intentional'),('S03','Exact240 continuation at241,epsilon.01,freshper-task policy/Adam','meta/*block*.pt; meta/*episodes.csv; audit/outer_verification.json','Original240 episodes notrerun; no renormalization; prediction-only outer moves'),('S04','Primaryadditionaldose/P2condition effects','tables/paired_seed_effects.csv; tables/paired_intervals.csv','Whole20pairedseeds; prespecified170/330; no selection/powerlaw'),('S05','Actual learningtrajectory andacquisitionspeed','tables/training.csv; tables/training_intervals.csv; tables/acquisition_timing.csv','ActualpreupdateCEverified; firstpassmaytransient; neveracquiredmissing'),('S06','Context-specificlearning matchedforfamiliarperformance','tables/block_transfer.csv; tables/matched_acquisition.csv; tables/matched_seed_effects.csv; tables/matched_intervals.csv','Fixedbins/firsteligibleblocks; conditional,notrandomized; actualstartingA/BCEretained'),('S07','Originalfunctionaljoint versus descriptivenonfloorcoexistence','tables/functional_splitting.csv; tables/learning_effects.csv; tables/switch_costs.csv; tables/joint_outcomes.csv','170/330only; no relaxedthreshold; floorslimit sensitivity; secondarytrajectorygateunassessed'),('S08','Exactaffinecomputation andgeometricalmechanisms','tables/geometry.csv; target/*/decomposition.csv.gz; target/*/mathematical_checks.csv.gz','Centerednormsnotmediation; JensenonlyaverageMixedCE; geometrynotskills'),('S09','RandomdirectionrobustnessseparatefromnaturalC','random_inputs/*_metrics.csv; random_inputs/*_positions.csv; random_inputs/*_controls.csv; random_inputs/*_vectors.csv; tables/random_robustness_seed.csv; tables/random_robustness_summary.csv','20vectorswithinseed,notindependentnetworks; magnitudeschangewithCnorm; fixedunitdirections'),('S10','Allpositions/logits/probabilities/fullstatesretained','target/*/states.pt; target/*/position_predictions.csv.gz; verification.json','Failed/unavailablecellsretained; no newsourcebranchreplay'),('S11','Human-like claim andremaininguncertainties','FINAL_META_LEARNING_SCALING_REPORT.md; RESEARCHER_HANDOFF.md','No newhumanfit; fixedH8Linear/ecology; earlyfunctionalcoexistencedoesnotreplaceoriginaljointgate')]
 write_frame(OUT/'CLAIM_LEDGER.csv',[dict(claim_id=i,claim=c,exact_sources=s,limits=l) for i,c,s,l in ledger])
 audit='''# Source/reuse and algorithm audit

| Component | Actual code/evidence | Reuse or new work |
| --- | --- | --- |
| Original architecture/RNG | context_experience_prior/core.py ExperienceLinear; high_scale_initialization/core.py LinearElman.forward | Imported unchanged; nn.RNN is a parametercontainer, manual identity recurrence executes |
| Objective/policy/Adam/clip | context_experience_prior/core.py step/objective/diagnostic; original confirmatory_srn and execution_objective_search | Original CE+.2execution/detachedconfidence/Adam.005/jointclip1; no new loss |
| Parent integrity | audit/protocol_seal.json; SOURCE manifest | All359parentoutputs/83code/115inputs checked beforetraining; parentunchangedafter |
| Original240 metaweights | registry/meta_resume_registry.csv; parent meta/seed_*.pt records240 | Exactsourceweights, next241 epsilon.01; no randomrestart |
| Dose0 target | registry/reuse_registry.csv P0-I/C |40unique full331-state histories, allsix archivedbranches/probes/mathdata reused |
| Dose240 target | registry/reuse_registry.csv P1/P2-I/C |80unique complete target histories/probes reused exactly |
| Newtasks | protocol.py; registry/seed_*block*.json |7balanced240blocks/seed, same sourceallowed/draw andschedule; predeclaredduplicate rejection |
| NewReptile | continuation.py invoking parent meta.run_episode |2,688,000innerupdates/134,400outermovesplanned, completebefore/adapted/after saved |
| Newtarget | target.py invoking sourcecore.step/trace/measures/diagnostic |560new330-update targetruns; all331model/policy/Adam states; Cconstancy andactualCEassertions |
| Learningcriterion | recurrent_learning_transfer/measures.py |Unchangedendpoint conjunction; no representationalthreshold substituted |
| Blocks/intervals | centered_context_linear/analysis.py; new analysis.py |Same signedgains/floors; fixedbins; commonwhole-seed10000draws |
| Linearcontrols | parent measures.py and nonzero_context_generalization/core.py |IndependentS/K, float64identities, boundedfloat32native andhypotheticalcolumn validation |
| Scientificoutputs | physical_run_registry/checkpoint hashes; rawshards; finalmanifest |Newnamespaceonly; oldsealeddata/codeuntouched |

Allphysical paths are root-relative and hashes are exact. Per-run shards are whole schema-aware DataFrames, never positional appends. Eachactual recorded pre-update loss is checked against its same-weight frozen evaluation and originalobjective identity before plotting. StageA initiallaunchguard failure occurred beforeworkers; initialreportrendering accessor error was display-only and retained. No scientificgradientupdatewas repeated for either correction. Allfail/passlogs remain in audit and sourceexecution snapshots in audit/executed_source. Fulltasklibraries, discardedduplicatecandidates, nonfinite/failed/unavailable statuses, conditionalsubgroupdenominators andnegativegains are retained. No rawhuman data, credentials or external publication.
'''
 (OUT/'SOURCE_REUSE_AUDIT.md').write_text(polish(audit))
 # Strict final link and seven-answer checks.
 report=(OUT/'FINAL_META_LEARNING_SCALING_REPORT.md').read_text();assert all(f'### {i}.' in report for i in range(1,8));assert '<bound method' not in report
 for name in ['FINAL_META_LEARNING_SCALING_REPORT.md','RESEARCHER_HANDOFF.md','REPRODUCIBILITY.md','SOURCE_REUSE_AUDIT.md']:
  for link in re.findall(r'\]\(([^)]+)\)',(OUT/name).read_text()):
   if '://' not in link:assert (OUT/link.split('#')[0]).exists(),(name,link)
 dump(OUT/'audit/final_products.json',dict(passed=True,seven_questions=True,all_report_links_valid=True,claim_ledger=True,source_audit=True,handoff=True,physical_runs=len(read('physical_run_registry')),fixture_gradient_updates=0));append('## Final scientificproductscomplete\nSevenregisteredquestions answeredfromactualtables,primarypairedcontrasts/jointvsdescriptivecoexistence/eligibledenominators/negativeandfloorlimitsretained. Fivefigurefamilies+fullall-dose/early/matched/diagnostic/mechanismcompanions,seed/positiondata,sourceclaimledger,handoff,sourceauditandreproductionguidewritten. No humanfit,mediation,universalscalingorindependentskillsclaimfromgeometry.')
if __name__=='__main__':main()
