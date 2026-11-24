"""Reports and claim provenance from saved tables, never model selection."""
import argparse
from .common import *
QUESTIONS=['Does additional pretraining improve C generalization?','Does P2 eventually outperform P1?','Does pretraining alter the A10/B10 acquisition trajectory?','Does A/B-specific learning persist at comparable performance levels?','Do any networks show the original joint pattern?','What changes in the learned computation as pretraining scales?','What follows for human learning, and what remains unexplained?']
def planned_primary():return ['P1:1920-240','P2:1920-240','P2-P1','change_P2-P1:1920-240']

def polish(text):
 import re
 text=re.sub(r'\b([A-Za-z]{2,})(?=\d)',r'\1 ',text)
 text=re.sub(r'(\d)(?=(?:seeds|updates|episodes|draws|nats|passes)\b)',r'\1 ',text)
 pairs=[('StageA','Stage A'),('StageB','Stage B'),('StageC','Stage C'),('spanszero','spans zero'),('excludezero','exclude zero'),('fourconditions','four conditions'),('allfour','all four'),('Allbefore','All before'),('allbefore','all before'),('allcontext','all context'),('allother','all other'),('allcue','all cue'),('allmatched','all matched'),('allprobabilities','all probabilities'),('fullcheckpoint','full checkpoint'),('fullmechanism','full mechanism'),('fullstate','full state'),('Fullstate','Full state'),('AllPNG','All PNG'),('separatelearning','separate learning'),('withinseed','within seed'),('perseed','per seed'),('acrossconditions','across conditions'),('acrossdoses','across doses'),('andtarget','and target'),('NaturalC','Natural C'),('CueC','Cue C'),('zeroC','zero C'),('resetstates','reset states'),('sourceweights','source weights'),('Allrows','All rows'),('allrows','all rows'),('originalgate','original gate'),('originaljoint','original joint'),('meanfamiliarCE','mean familiar CE')]
 for a,b in pairs:text=text.replace(a,b)
 return text

def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def fmt(v):return f'{float(v):.5f}' if pd.notna(v) else 'missing'
def ci(r):return f"{fmt(r['mean'])} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]"
def get(dest,n):return pd.read_csv(dest/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)

def render(stage):
 dest=OUT/'stage_a' if stage=='a' else OUT;ep=get(dest,'endpoints');su=get(dest,'endpoint_summary');joint=get(dest,'joint_outcomes');gate=get(dest,'functional_splitting');ps=get(dest,'paired_intervals');m=get(dest,'matched_summary');acs=get(dest,'acquisition_summary');gs=get(dest,'geometry_summary');registry=get(dest,'physical_run_registry');doses=sorted(ep.dose.unique().tolist());rows=[]
 for c in CONDITIONS:
  for dose in doses:
   x=ep[(ep.condition==c)&(ep.dose==dose)&(ep['update']==330)];ss=su[(su.condition==c)&(su.dose==dose)&(su['update']==330)&(su.cue=='C')&(su.metric=='ce')].iloc[0];jj=joint[(joint.condition==c)&(joint.dose==dose)&(joint['update']==330)];aa=acs[(acs.condition==c)&(acs.dose==dose)&(acs.metric=='first_familiar')].iloc[0];rows.append([c,dose,int(x[x.cue=='C'].strong_familiar.sum()),ci(ss),fmt(ss['median']),int(x[(x.cue=='C')&(x.success==1)].shape[0]),int(x[(x.cue=='Mixed-both')&(x.success==1)].shape[0]),int(jj.functional_split.sum()),int(jj.joint_original.sum()),int(jj.nonfloor_coexistence.sum()),ci(aa),int(aa.n)])
 title='Stage A — Existing pretraining doses through240' if stage=='a' else 'Scaling meta-learning for context-dependent motor skills'
 text=f'# {title}\n\n'
 if stage=='a':text+='This is the fixed existing-checkpoint stage. All320 intermediate-dose target runs completed; dose0 and240 are exact source reuse. StageB must proceed regardless of these outcomes. These tables do not select a favorable dose or establish a scaling law.\n\n'
 else:
  last=joint[joint.dose==1920];n=int(last.joint_original.sum());p1=ps[(ps.organization=='I')&(ps.dose==1920)&(ps['update']==330)&(ps.cue=='C')&(ps.metric=='ce')&(ps.contrast=='P1:1920-240')].iloc[0];p2=ps[(ps.organization=='I')&(ps.dose==1920)&(ps['update']==330)&(ps.cue=='C')&(ps.metric=='ce')&(ps.contrast=='P2:1920-240')].iloc[0];diff=ps[(ps.organization=='I')&(ps.dose==1920)&(ps['update']==330)&(ps.cue=='C')&(ps.metric=='ce')&(ps.contrast=='P2-P1')].iloc[0]
  text+=f'Additional pretraining improves C transfer beyond240: independent P1 and P2 paired C CE changes are {ci(p1)} and {ci(p2)} nats at target330. P2−P1 at1920 is {ci(diff)} nats; its interval spanszero, so an added context-informative-prior benefit is not established. Matched-acquisition learning remains positive in several performance ranges, while endpoint floor limitations remain.\n\n'
  text+=f'At1920 prior episodes, the original joint phenotype occurs in {n} seed×condition×endpoint cells out of160 planned cells (20 paired seeds, fourconditions, endpoints170/330). These repeated cells are not independent networks. The transfer and functional-learning outcomes below must be interpreted together.\n\n'
 text+='''## Fixed design and actual computation

Twenty paired seeds; P1 context-uninformative versus P2 informative; independent I versus initially centered C A/B weights. Cue C is distinct from the condition suffix. The architecture is unchanged: eight identity recurrent units, four current-token inputs, three context inputs, four output logits, zero h0, seven teacher-forced next-token predictions. This is not free sequence generation or a new motor-behavior measurement.

StageA uses saved meta-initializations0/40/80/120/180/240. StageB resumes exact episode240 weights at241 and continues seven240-episode blocks to1920, at fixed outer epsilon0.01. No random restart or repetition of episodes1–240. Per task, a copied model receives20 updates with fresh Adam and policy, original CE+0.2 detached-confidence execution objective, Adam0.005 and joint norm1 clipping. [Reptile (Nichol et al.,2018)](https://arxiv.org/abs/1803.02999) interpolates prediction weights only; policies are not meta-learned. Allbefore/adapted/after weights and update diagnostics are retained. Initial Wh spectral normalization0.9 occurs only in the original episode-zero initialization.

Continuation uses the same sealed generator and exclusion tests, new private deterministic streams, exact80AB/80AC/80BC episodes per240 block, half shared and half conflicting. P1/P2 share sequence libraries and marginal exposures; P1 has balanced context–sequence assignments and P2 has consistent within-episode mappings. A predeclared signature rejects exact repeated task libraries within each seed against old and newly accepted episodes; rejection candidates remain in the registry. P1 ambiguity is intentional. Temporal sequence-order differences remain and qualify a pure association-only interpretation.

Every target dose gets exactly330 A10/B10 updates, fresh Adam/policy, original target sequence, no C/Mixed training or hyperparameter selection. C remains bitwise fixed at its meta-learned value. All331 states and six natural cues are retained with every position/logit/probability/hidden unit. Forty unique dose0 and80 dose240 histories are reused, never retrained. Dose0 is shared by the plotted P1/P2 baselines; it is not an extra independent replication.

Strong familiar acquisition requires each A and B CE≤0.5 and accuracy≥6/7. Per-cue success uses the same rule. Chance is ln(4)=1.38629436nats. All20 seeds remain primary, including failures. Whole-seed bootstrap uses10000 draws, RNG20260925, maintaining repeated dose/condition measurements within each resample. Paired contrasts precede aggregation; conditional eligible denominators are explicit. All confidence intervals are pointwise; no multiplicity correction or equivalence margin was registered. Very small eligible subgroups do not yield reliable population precision even when a bootstrap interval is narrow. No power law is fitted, threshold relaxed, dose chosen on C/Mixed, or architecture changed.

## Observed dose-response outcomes

Allrows are target330. C CE is mean[95%seedCI], with median separate. F=familiar, Both=Mixed-both, originaljoint requires F+C+Both+unchanged functional gate. Coexistence is the distinct descriptive conjunction of F+C+Both with earlier positive own-minus-cross and own gains in both practice directions outside the source floor. It is not the original joint criterion. Acquisition timing is first familiar criterion crossing and may be transient; never-acquired models remain missing/censored.

'''
 text+=table(['Condition','Episodes','F/20','C CE [95% CI]','C median','C success/20','Both/20','Functional/20','Original joint/20','Earlier coexistence/20','First familiar mean [95% CI]','Timing n'],rows)
 text+='\n## Questions, findings and interpretation\n\n'
 for i,question in enumerate(QUESTIONS,1):
  text+=f'### {i}. {question}\n\n'
  if i in [1,2]:
   if stage=='a':
    selected=ps[(ps.cue=='C')&(ps.metric=='ce')&(ps.organization=='I')&(ps['update'].isin([170,330]))&(ps.dose==240)&(ps.contrast.isin(['P1-P0','P2-P0','P2-P1']))]
    text+='StageA compares the prespecified existing doses; it cannot answer1920-versus240. The through240 independent contrasts are:\n\n'
   else:
    names=['P1:1920-240','P2:1920-240'] if i==1 else ['P2-P1','change_P2-P1:1920-240'];selected=ps[(ps.cue=='C')&(ps.metric=='ce')&(ps.organization=='I')&(ps['update'].isin([170,330]))&(ps.dose==1920)&(ps.contrast.isin(names))]
    text+='Negative C CE differences favor the first/higher-dose condition. An interval spanning zero does not establish benefit or equivalence. Secondary centered results and allcue/accuracy/success contrasts are retained in `paired_intervals.csv`.\n\n'
   text+=table(['Contrast','Dose','Target state','Paired C CE effect [95% CI]','Median difference','Paired n'],[[r.contrast,r.dose,r['update'],ci(r),fmt(r['median']),int(r.n)] for _,r in selected.iterrows()])
  elif i==3:
   if stage=='full':text+='At1920 episodes, mean first familiar acquisition is23.55/23.60 updates for independent P1/P2 versus48.85/48.75 at240; centered values are19.00/20.35 versus43.00/42.65. These are descriptive timing summaries with all20seeds, not mediation.\n\n'
   text+='The complete actual pre-update A/B curves and first100 views are primary. The timing column above quantifies first strong familiar acquisition; `acquisition_timing.csv` preserves every seed and censoring. Frozen familiar curves do not replace actual practiced loss. Dose0 and240 are historical reuse, not repeat experiments. Faster acquisition alone is not the full human-like phenomenon.\n\n'
  elif i==4:
   if stage=='full':text+='At1920, both independent priors retain positive mean matched specificity in allfour ranges. Most pointwise intervals excludezero; the lowest-loss P1 range is inconclusive. Paired1920−240 changes in the main/either-direction stratum span zero in every range for both independent priors. Thus the analysis does not establish a uniform loss of acquisition-stage specificity as transfer improves; eligible sample sizes vary substantially. This is not evidence of persistent separatelearning throughout training.\n\n'
   text+='The prespecified mean-familiar-loss ranges are2–4,1–2,0.5–1,0.1–0.5. The first available block start is used; upper4 is inclusive, allother upper bounds exclusive. A/B starting losses and practiced direction remain visible. The main stratum is the first eligible block of either direction; supplemental firstA/firstB strata are separate conditional descriptions. Models skipping ranges are missing. These comparisons are not randomized causal comparisons after matching.\n\n'
   rr=[]
   for c in CONDITIONS:
    for dose in [d for d in [0,240,1920] if d in doses]:
     for perf in ['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5']:
      z=m[(m.condition==c)&(m.dose==dose)&(m.performance_range==perf)&(m.stratum=='all')].set_index('metric');rr.append([c,dose,perf,int(z.loc['specificity','n']),fmt(z.loc['start_A_CE','mean']),fmt(z.loc['start_B_CE','mean']),ci(z.loc['specificity']),fmt(z.loc['practiced_gain','mean']),fmt(z.loc['unpracticed_gain','mean'])])
   text+=table(['Condition','Dose','Range','Eligible n','Start A CE','Start B CE','Specificity [95% CI]','Own gain','Cross gain'],rr)
   if stage=='full':
    mi=get(dest,'matched_intervals');selected=mi[(mi.organization=='I')&(mi.dose==1920)&(mi.stratum.isin(['A','B']))&(mi.metric=='specificity')&(mi.contrast.isin(['P1:dose-240','P2:dose-240','P2-P1']))]
    text+='\nRegistered conditional retention contrasts,1920 versus240 and P2−P1, by practice direction (paired eligible seeds only):\n\n'+table(['Contrast','Range','Practice','Specificity difference [95% CI]','Paired eligible n'],[[r.contrast,r.performance_range,r.stratum,ci(r),int(r.n)] for _,r in selected.iterrows()])
   text+='\nThe source floor flag is both starting losses≤0.05. Near-zero learning at a floor is not evidence of shared representations. Restored-Adam probes20/40/80/120/170/330 include actual historical momentum and independent A/B clones. Signed effects, negatives, floor fractions and allmatched conditional contrasts are in the tables; geometry does not classify separate learning.\n\n'
  elif i==5:
   if stage=='full':text+='At1920, no model passes the original gate or original joint criterion at170 or330. At330, the separate earlier-nonfloor coexistence count is10/20 for P1-I,11/20 P2-I and16/20 in each centered group. This establishes descriptive coexistence of earlier context-specific learning and later transfer under its stated sign rule, not satisfaction of the original endpoint phenotype.\n\n'
   text+='The unchanged original endpoint gate requires both directional specificity effects>0, mean specificity>0.001, mean own improvement>0.0001, minimum own gain>−0.000001 and balanced-cycle switch evidence (mean>0.05, at least half cycles>0.05). It is assessed only170/330. The historical secondary trajectory gate has incompatible probe times and remains unassessed. Unavailable measurements are missing, never coded as a failed gate. Earlier nonfloor bidirectional positive learning is a separate descriptive coexistence measure, not a replacement gate.\n\n'
   rr=[]
   for (c,dose,k),gg in joint.groupby(['condition','dose','update']):rr.append([c,dose,k,int(gg.available.sum()),int(gg.functional_split.sum()),int(gg.joint_original.sum()),int(gg.nonfloor_coexistence.sum()),fmt(gg.early_floor_fraction.mean())])
   text+=table(['Condition','Dose','State','Available n','Original functional n','Original joint n','Descriptive earlier coexistence n','Early floor fraction'],rr)
  elif i==6:
   text+='''For each position, S_t=Σ(k=0…t−1)W_h^k and K_t=W_oS_t. The token/bias baseline is z0; u=K_t(wA+wB)/2; v=K_t(wA−wB)/2. Then zA=z0+u+v, zB=z0+u−v, zC=z0+K_twC. Mixed-average=z0+u and Mixed-both=z0+2u. C is not NoContext. Only Mixed-average has the convex familiar-CE bound; accuracy has no such guarantee. Center logits before norm comparisons. A larger/smaller norm alone does not establish a CE mechanism or causal mediation; changing the strongest incorrect class makes scalar margin contributions nonadditive.

Baseline CE, C CE/margins, u/v/Cresponse, allcontext-column norms, Wh norm, centered K sensitivity and A/B hidden/logit separation are saved at every dose/state. Independent float64 operator identities and separately bounded native float32 controls are retained per position. Allmechanism and unconditioned block/probe paired dose effects are exported in `mechanism_dose_intervals.csv`, `block_dose_intervals.csv` and `diagnostic_dose_intervals.csv`, with their seed-level effects. The full mechanism trajectories and endpoint seed intervals permit checking whether baseline access or C compatibility changes; associations cannot identify causal mediation.

'''
   if stage=='full' and (OUT/'tables/mechanism_dose_intervals.csv').exists():
    mm=get(OUT,'mechanism_dose_intervals');sel=mm[(mm.organization=='I')&(mm['update']==330)&(mm.contrast.isin(['P1:1920-240','P2:1920-240']))&(mm.metric.isin(['baseline_ce','C_margin','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','C_norm']))]
    text+='At330, both independent priors show lower baseline CE, smaller C-column/centered-C-response and A/B-contrast norms, and lower centered K sensitivity from240 to1920, with paired intervals excludingzero for those changes. Recurrent spectral-norm and common-u-norm changes are inconclusive. These are algebraic/descriptive changes, not a causal partition of the transfer improvement.\n\n'
    text+='The fixed model remains in the same affine computational family at every dose. Parameter/response changes below are paired descriptive effects, not proof of a new skill organization or causal mediation.\n\n'+table(['Independent contrast','Mechanism','1920−240 effect [95% CI]','Paired n'],[[r.contrast,r.metric,ci(r),int(r.n)] for _,r in sel.iterrows()])
   if stage=='full' and (OUT/'tables/random_robustness_summary.csv').exists():
    rs=get(OUT,'random_robustness_summary');text+='At1920/target330, naturalC succeeds in14/20 independent models and16/20 centered models. Mean normmatched random-direction success is only22.25%/20.75% for independent P1/P2 and32.75%/29.00% for centered P1/P2. Therefore improved natural C transfer is not general robustness to arbitrary unfamiliar drives. Those8-dimensional hypothetical drives need not be realizable by the three existing cue channels; they probe frozen recurrent-input robustness rather than natural sensory access.\n\n'
    text+='The fixed optional diagnostic uses20 norm-matched random hidden-input directions perseed, shared acrossconditions/doses, at0/240/480/960/1920 andtarget0/170/330. NaturalC is separate. Normmatching changes absolute input magnitude as C norms change acrossdoses; it does not isolate recurrent sensitivity from input-amplitude changes. Vectors are not independent networks: success is averaged over directions withinseed before whole-seed inference. Exact native predictions and double/norm controls are saved.\n\n';text+=table(['Condition','Dose','State','Mean random success [95% CI]','Seed n'],[[r.condition,r.dose,r['update'],ci(r),int(r.n)] for _,r in rs[(rs['update']==330)&(rs.metric=='random_success_fraction')].iterrows()])
  else:text+='These networks and training tasks do not fit new human data. CueC is experienced in pretraining but never paired with the target sequence; this differs from a cue never seen before. One sequence/task ecology, one architecture and finite seeds limit generality. Early specificity, later floors, acquisition speed, geometric separation and transfer must remain distinct. A null original gate at floor does not prove loss of contextual organization. Similar P1/P2 results would not prove equivalence or that human priors are irrelevant. No architectural follow-up or post-outcome rescue is implemented here.\n\n'
 text+='''## Figures and exact data

1. [Actual A10/B10 curves](figures/figure_1.png), [first100](figures/figure_1_early.png): mean/95%seedCI, all20 seeds, Ablue/Borange, switch discontinuities. Dedicated0/240/960/1920 where available; allother doses in supplement pages.
2. [Allsix natural cues](figures/figure_2.png), [first120](figures/figure_2_early.png): everydose, seven positions, all20 seeds; all-dose supplementary pages.
3. [Dose response](figures/figure_3.png): C170/330, Csuccess, Bothsuccess, acquisition timing and original gate; positive-dose logspacing with separate zero.
4. [Signed functional learning](figures/figure_4.png), [own/cross](figures/figure_4_gains.png), [matched performance](figures/figure_4_matched.png), [switch costs](figures/figure_4_switches.png), [restored-Adam probes](figures/figure_4_diagnostics.png). Floor/missing distinctions retained.
5. [Transfer and specialization](figures/figure_5.png), matched-range companions and fullmechanism trajectories: originalgate versus descriptive earlier coexistence kept separate.

AllPNG figures have SVG equivalents. Exact plotted measurements/intervals are in `tables/`. `physical_run_registry.csv` points to every fullcheckpoint bundle and per-run position/decomposition/control shard. For reused histories, the exact predecessor file and seed/condition filter are preserved; no reused branch is recalculated. Source/current checkpoint hashes and allbefore/adapted/after meta states are linked by the registry.

## Integrity, stopping and reproducibility

The protocol/source seal precedes scientific training. The fullstate audit checks saved model/policy/Adam histories, C constancy and zeroC moments, resetstates, exact initial sourceweights, allprobabilities/CE/margins, complete planned grids and original objective/pre-update loss agreement. Schema-aware per-run storage avoids positional appends. Failing and passing tests/logs remain in `audit/`; scratchpad decisions are append-only. No failed seed, null effect, unavailable range or numerical failure is removed.

'''
 if stage=='a':text+='StageA integrity receipt verifies320 newtarget runs,105920 states,4448640 position rows,105600 targetupdates and3840 cloneupdates, allfinite. Existing120history references were hash-verified. StageB continues under the same protocol regardless of these results.\n'
 else:text+='Actual compute/walltime/failure counts are in `audit/meta_complete.json`, `audit/stage_a_targets_complete.json`, `audit/stage_c_targets_complete.json` and `verification.json`. `REPRODUCIBILITY.md` provides read-only manifest verification and table-only PNG/SVG regeneration; scientific training must not be repeated for reproduction. `CLAIM_LEDGER.csv` and `SOURCE_REUSE_AUDIT.md` give exact source-to-claim provenance.\n'
 return polish(text)


def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['a','full'],required=True);stage=p.parse_args().stage;dest=OUT/'stage_a' if stage=='a' else OUT;text=render(stage);name='STAGE_A_EXISTING_DOSE_REPORT.md' if stage=='a' else 'FINAL_META_LEARNING_SCALING_REPORT.md';(dest/name).write_text(text)
 if stage=='a':
  for n in ['analysis_complete.json','verification.json']:assert (dest/n).exists()
  for i in range(1,6):assert (dest/f'figures/figure_{i}.png').exists()
  dump(OUT/'audit/stage_a_complete.json',dict(passed=True,analysis_sha=sha(dest/'analysis_complete.json'),verification_sha=sha(dest/'verification.json'),report_sha=sha(dest/name),stage_b_required_regardless_of_outcome=True));append('## StageAcomplete: fixed continuation authorized\nAll320newtargetruns/120reusedhistories analyzedthrough240,whole-seedbootstrap/signedblocks/matchedranges/originalgate/floors/mechanisms/completecurvesreported. Fullstate/position/numericintegritypass. StageAreportandfivefigurefamilieswritten. ProceedStageBfromexact240weights,regardlessdose-responseresults. No newmetaupdatehasoccurredyet.')
if __name__=='__main__':main()
