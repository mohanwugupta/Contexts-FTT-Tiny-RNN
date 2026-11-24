"""Observed-result teach-back and exact source/claim crosswalk, tables only."""
import re
from .common import *
from .analysis import summarize
from .report import primary_filter,QUESTIONS
from experiments.meta_learning_scaling.report import ci,table,fmt,polish

def main():
 ep=read('endpoints');pa=read('paired_intervals');j=read('joint_outcomes');m=read('matched_intervals');held=read('heldout_summary');pr=primary_filter(pa);rows=[]
 for c in CONDS:
  for a,d in [('Historical',0),('Seen',960),('Transfer',960)]:
   x=ep[(ep.condition==c)&(ep.arm==a)&(ep.dose==d)&(ep['update']==330)&(ep.cue=='C')];jj=j[(j.condition==c)&(j.arm==a)&(j.dose==d)&(j['update']==330)];rows.append([c,a,fmt(x.ce.mean()),fmt(x.familiar_ce.mean()),fmt(x.gap.mean()),fmt(x.gap.median()),int(x.strict_all.sum()),int(jj.functional_split.sum()),int(jj.descriptive_coexistence.sum())])
 lead='''## Observed result in plain language

The registered mean C-gap comparison favors Transfer-meta in both independent priors. At+960/target330, Transfer C gaps average0.1446nats(P1) and0.2028(P2), versus0.4599/0.4293 forSeen, while familiar loss stays about0.0003 and all20 networks percondition acquireA/B strongly. The primary paired effects are−0.3153CI[−0.6106,−0.0624] and−0.2265CI[−0.4398,−0.0545]. However, medianpaired improvements are only0.00168/0.00535nats, so the muchlarger mean effects reflect heterogeneous loss tails, not uniform gains across seeds. Strict-all C targets reach15/20(P1-I) and16/20(P2-I), versus15/20 and13/20 Seen. The residual gap is reduced, not universally eliminated.

Preservation of specialization is qualified. IndependentP1 matched-specificity contrasts are inconclusive. IndependentP2 Transfer−Seen specificity is lower in CE0.5–1 (−0.334CI[−0.592,−0.061], pairedn13) andCE2–4 (−1.190CI[−1.752,−0.611], pairedn8); othermainstrata span zero. These conditionalpointwise findings are not a global equivalence or impairment test. No newmodel passes the original170/330 endpoint gate; latefloors limit its sensitivity. Unseen-conflict diagnostics show some appropriatecontextuse afteradaptation, but do not establish uniformly preservedskills. The experiment supports directed transfer improvement with remaining residual errors and functional-learning tradeoffs, not an unqualified human-like solution.

'''
 path=OUT/'FINAL_HELDOUT_CONTEXT_META_REPORT.md';text=path.read_text();parts=text.split('\n',2);text=parts[0]+'\n\n'+polish(lead)+parts[2]
 text+='\n## Final per-seed-distribution summary\n\n'+table(['Condition','Arm','C CE mean','Familiar CE mean','C gap mean','C gap median','Strict-all/20','Original functional/20','Earlier coexistence/20'],rows)
 diag=read('C_frozen_position_summary');dd=diag[(diag['update']==330)&(diag.metric.isin(['C_only_errors','shared_errors','C_helped_positions','C_minus_baseline_CE']))];text+='\nFrozen-position diagnosis (same weights, C versus zero input): C-only errors have a correct NoContext prediction; shared errors are wrong underboth; C-helped positions are correctonlyunderC. This identifies an input contribution at a frozen state, not mediation of a training-arm effect.\n\n'+table(['Condition','Arm','Dose','Position effect','Mean [95% CI]','n'],[[r.condition,r.arm,r.dose,r.metric,ci(r),int(r.n)] for _,r in dd.iterrows()])
 qq=read('meta_query_summary');qq=qq[(qq.dose==960)&(qq.window=='last240')&(qq.metric=='query_ce')];text+='\nMeta-query CE over the final240 tasks is averaged withinseed. Shared seen/withheld queries have different supportmembership bydesign; lowertrainingqueryloss alone is not the targetgap test. P1 conflicting queryambiguity is intentional.\n\n'+table(['Condition','Arm','Task type','Query CE [95% CI]','Seeds'],[[r.condition,r.arm,r.kind,ci(r),int(r.n)] for _,r in qq.iterrows()]);path.write_text(polish(text))
 hand='''# Researcher handoff — held-out query meta-learning

Start with the final report and Figure3. The question is continuous C disadvantage relative to strong familiar A/B performance, not only Csuccess≤0.5. All20 seeds are retained perP1/P2 andI/C. Historical+0 means the sourceepisode1920 model, not an untrained initialization.

'''+polish(lead)+table(['Condition','Arm','C CE','Familiar CE','Gap mean','Gap median','Strict-all/20','Original functional/20','Earlier coexistence/20'],rows)+'''
## Actual algorithm teach-back

We changed Reptile’s endpoint interpolation to first-order MAML. Each episode starts from the current initialization in a copied taskmodel, fresh innerAdam and fresh executionpolicy. Twenty sourceCE+.2execution updates adapt it. The adapted model then predicts a query sequence. Its **query CE gradient** is copied onto the initialization’s corresponding predictionparameters; clipped outerAdam.001 updates that initialization. The taskmodel is discarded. No query innerstep, no outerpolicytraining, no derivative throughAdam and no Reptile interpolation. OuterAdam moments persist across960episodes; inner moments reset eachtask. The independent reference and fullanalytic saved-gradient/Adam audit verify this.

Sharedtasks holdQout ofTransfer support; Seen includesQ plus a counterbalanced companion. Bothquery the exactsameS,Q. Supportmembership therefore differs inherently, with matchedmarginal exposures. Conflict episodes are identicalacrossarms; P2mappingstable, P1balancedambiguous. Queryloss is a mean, not a batchdependent sum. Maintraining registries were sealed beforemeta. Correcteddiagnosticv2 was sealed before its diagnosticupdates; the initialsequence-overlapfailure andallv1 results are preserved.

At+240/+960 fresh330-step A10/B10 target runs use originalobjective/Adam/clip. Cneverlearns the target and has zero freshCmoments and bitwisefixedcolumn. All331 fullstates and everyposition/logit/probability/hidden are saved. Historical80 targethistories/probes are exactreuse, not independent newreplications. Originalfunctionalcriterion and fourmatchedranges unchanged; firsteligibleblock/eitherdirection and supplementalfirstA/B retain actualA/Bstartinglosses andmissingeligibility. Postacquisitiongap includes everylaterstate fromfirstfamiliarpass; never-acquired is missing/censored, notzerogap.

## Computation and limits

ForLinearrecurrence, K_t=WoΣWh^k, zA=z0+u+v, zB=z0+u−v, zC=z0+K_twC. C−Mixed-average=K_t[wC−(wA+wB)/2]. CorrectCprediction is the goal, not equalityto theA/Bmidpoint. Centerlogits before normcomparison. Geometry/normcorrelations cannot establish causalmediation or separate skills. Originalendpointgatefailureatfloor is notproof of absentcontextualorganization.

FrozenCremoved equalsNoContext exactly, not an independentrescue. Half/1.5C and fixed20normmatchedrandomdrives are diagnostic. Historicalrandomcache is reused. Random8D drives need not be realizablethrough actualthreecuechannels; they test sensitivity, not a new sensorycontext or naturalretrieval. Normmatching changes amplitude withlearnedCnorm, so it cannot cleanly isolateWh/K changes frominputmagnitude.

Unseenconflictdiagnostics use12 sealedtasks/seed and20 diagnosticupdates/task fromeachinitialization; stableidentifiable mappingsforallpriors testcapacityforcontextuse, not P1trainingambiguity. Prefixposition3 hasidenticaltokens butdifferenttargets. Tasks/positions are averagedwithinseed before CI, not independentnetworks.

No human data were fitted, noarchitecture altered, and no checkpoint/rate/threshold chosen onC/Mixed outcomes. Bothadditionalqueryarmsrunall960episodes andalltargetcheckpoints; poorseeds andnegativefindingsremain. ReadREPRODUCIBILITY.md andtheclaimledger/sourceaudit. Reproducefromsavedtables/seals withouttraining. PrimaryP1/P2 independentcomparisons areseparaten20; withinseedprior-average secondaryn20, nevern40.
''';(OUT/'RESEARCHER_HANDOFF.md').write_text(polish(hand))
 ledger=[('H01','Exact1920 starts andmatchedqueryintervention','registry/historical.csv; registry/seed_*_b*.json; audit/protocol_seal.json','Sharedsupportmembershipdiffersinherently; balancedmarginals, samequerydata'),('H02','Query-gradientFOMAMLnotReptile','meta/*.pt; meta/*.npz; audit/meta_verified.json; audit/05_GREEN_fomaml_reference.log','Firstorder, nooptimizerJacobian; predictionparamsonly; newhyperparametersfixednotvalidated'),('H03','PrimaryCgapTransfer−Seen/+960/I','tables/paired_seed_effects.csv; tables/paired_intervals.csv; tables/endpoints.csv','P1/P2separaten20; mean/median/heterogeneity; noC-success-onlyclaim'),('H04','Residualpost-acquisitiondisadvantage','tables/acquisition.csv; tables/post_acquisition_curve.csv; tables/post_acquisition_intervals.csv','Firstpassinclusive; neveracquiredmissing; laterworseningretained'),('H05','Positionerrorsandcontextdrive','tables/C_positions.csv; tables/C_error_concentration.csv; tables/C_frozen_position_diagnosis.csv; tables/C_frozen_position_summary.csv','Frozeninputeffectsnottrainingmediation; exactlyzeroCEsharessetmissing'),('H06','Functionalretentionandmatchedacquisition','tables/block_transfer.csv; tables/matched_acquisition.csv; tables/matched_intervals.csv; tables/learning_effects.csv; tables/switch_costs.csv','Conditionaleligiblepairs, fixedbins, allsigns; geometrynotseparateskills'),('H07','Originaljointversusearliercoexistence','tables/functional_splitting.csv; tables/joint_outcomes.csv','Original170/330thresholdsunchanged; floorlimits; secondarytrajectorygateunassessed'),('H08','Contextuseonunseenconflicts','registry/heldout_v2_seed_*.json; heldout_diagnostics_v2/*/tasks.csv; tables/heldout_summary.csv','12tasksaveragedwithinseed; identifiablediagnosticsforallpriors; notselection'),('H09','ExactC-minus-sharedcomputation','target/*/decomposition.csv.gz; reuse_enrichment/*/decomposition.csv.gz; tables/geometry.csv','Kidentitywithfixedbounds; CEcorrectnessnotmidpointconstraint; nocausalnormclaims'),('H10','Frozeninputcontrolsandrandomcache','tables/control_metrics.csv; tables/control_positions.csv; tables/control_numerical.csv; tables/control_summary.csv','Natural/zero reused, removed=NoContext; randomsensitivitynotnewchannel; n20notvectors'),('H11','Fullstate/source/serializationfidelity','audit/targets_verified.json; registry/new_targets.csv; target/*/states.pt; audit/meta_verified.json','All331targetstates/fullinner21/outerhistories; sourcemanifestsimmutable'),('H12','Queriesversustargetgeneralization','tables/meta_query_seed.csv; tables/meta_query_summary.csv; tables/paired_intervals.csv','Queryconditionsinherentlydifferent; ambiguityP1; trainingCEnotcriterion/selection')]
 write_frame(OUT/'CLAIM_LEDGER.csv',[dict(claim_id=i,claim=c,exact_sources=s,limits=l) for i,c,s,l in ledger])
 audit='''# Source and reuse audit

| Component | Actual source | Disposition |
| --- | --- | --- |
| Architecture/manualidentity | context_experience_prior/core.py ExperienceLinear; high_scale_initialization/core.py LinearElman.forward | OriginalH8, noactivation/encoder/gatingchange; temporaryfactoryweightsfullyoverwrittenbysource |
| Innerobjective/policy/Adam/clip | context_experience_prior/core.py step/objective/policy/optimizer/diagnostic | Importedunchanged; CE+.2execution, detachedconfidence, Adam.005, jointclip1 |
| Source1920meta | meta_learning_scaling/v1/meta/seed_*_block_7.pt | Exactlastmeta, bothqueryarmsidenticalstart; noold1920episodesreplayed |
| Historicaltarget | registry/historical.csv → scalingtarget/s*_d1920 |80completehistories/probes/sixcuesreused, notnewreplications |
| Newtasks | heldout_context_meta/protocol.py andsealedregistry |960/seed, exactroles/exposures; samepathsandquerylabelsacrossarms, duplicates/exclusionsfixed |
| FOMAML | fomaml.py/train.py |153600querygradient/outerAdamupdates, noReptile; allbefore/adapted/after andlosslessinner/outerhistories |
| Referenceandfullaudit | test_fomaml.py; verify.py | Independentinner/referenceupdate; analyticBPTTfromsaveddata+adjacentAdamaudit,noquerygradientreplay |
| Newtargettrajectories | target.py, sourcedstep/trace/probes/measures |320×330actualupdates, full331states/fixedC/sixcues; schemawholefiles, preupdateCEvsfrozenevals |
| Originalfunctionalcriterion | recurrent_learning_transfer/measures.py | Same170/330conjunction; nogeometrygate orrelaxedthreshold |
| Matching/bootstrap | meta_learning_scaling/analysis.py; centered_context_linear/analysis.py | Samebins/floor/signedgains; paired20whole-seed10kresamples |
| Missinghistoricalmechanisms | data.py/mechanisms.py | Savednativepositions+weights/operators only, nooldnativeforwardreplay |
| Frozencontrols | controls.py/sourcefunctionalinject/operators | Natural/zero/cachebranchesreused; missingfixeddrivesonly; nooptimization |
| Diagnosticconflicts | heldout.py + heldout_correction.py |12strictlyunseen-v2tasks/checkpoint,400initializations,96000validdiagnosticsteps; initial96000invalidstepsretainedseparately; noouterupdateorselection |

Sourcehashes/protocol: audit/protocol_seal.json. Newcheckpoint/packedhashes: registry/meta_blocks.csv, registry/new_targets.csv. Allpredecessors are verified unchanged after the newprogram. The reference testinitially found1.16e−10biasdifference because its tokenprojectionwaspositionwise whilecanonicalsourcebatches; matchingprojectiongranularity correctedthefixture, nottheactualalgorithm orscientificthresholds. Failed04andpassing05 logsremain. Fixtures120inner+6outerupdates are separatefromregisteredscience. No othernewtraining-basedfixtureis rerunforfinalreadonlychecks. Initialheldouttaskpairswerefreshbut46individualpathswerepreviouslyseen. V1isinvalidforthatclaimandretained;v2enforcesstrictindividualfreshnesswithunchanged12tasks/seed,roles,streamand20-updatebudget. Onlydiagnosticcopiesrerun;no meta/targetweightsorhyperparameterschanged. Theadditional96000diagnosticupdatesareatechnicalcorrectioncost,actualnewoptimizersteps3527040versusplannedvalid3431040.
''';(OUT/'SOURCE_REUSE_AUDIT.md').write_text(polish(audit))
 for name in ['FINAL_HELDOUT_CONTEXT_META_REPORT.md','RESEARCHER_HANDOFF.md','REPRODUCIBILITY.md','SOURCE_REUSE_AUDIT.md']:
  txt=(OUT/name).read_text();assert '<bound method' not in txt
  for link in re.findall(r'\]\(([^)]+)\)',txt):
   if '://' not in link:assert (OUT/link.split('#')[0]).exists(),(name,link)
 assert all(f'## {i}.' in (OUT/'FINAL_HELDOUT_CONTEXT_META_REPORT.md').read_text() for i in range(1,8));dump(OUT/'audit/products_complete.json',dict(passed=True,seven_questions=True,source_claim_ledger=True,all_links_valid=True,fixture_inner_updates=120,fixture_outer_updates=6,selection=False));append('## Finalscientificproducts\nObservedmeangapbenefitsvsSeenwithsmallmedianpairedgains/residualgap,qualifiedfunctionaltradeoffs,unseencontextchecks,fullpositionsandfixeddriveeffectsreported. Sevenquestionsanswered,exactsourceclaimledger/teachback/audit/reproguidewritten. No binaryC-onlyhumanlikeclaim, thresholdchangeorposthocmodelselection.')
if __name__=='__main__':main()
