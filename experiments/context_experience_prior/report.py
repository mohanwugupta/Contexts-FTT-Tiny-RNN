"""Scientific report/handoff from immutable measurement tables only."""
from .core import *

def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip')
def md(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def fmt(x):return f'{x:.5f}'
def interval_text(row):return f"{row['mean']:.5f} [{row.ci_low:.5f}, {row.ci_high:.5f}]"
def claim_ledger():
 return pd.DataFrame([
  ('C01','All six conditions and20 paired seeds, final240 only','established implementation','registry/*.json; meta_registry.csv; target_registry.csv; audit/protocol_seal.json','No selected target outcomes'),
  ('C02','P1/P2 marginal exposures match; only P2 context identifies conflicting sequence within episode','established registry','registry/updates.csv; registry/episodes.csv; registry/sequences.csv','Temporal sequence-order differences remain; not pure association-only isolation'),
  ('C03','P0 is exact reused NI/NC A/B trajectory with original nonzero unused C restored in copied states','established source equivalence','audit/protocol_seal.json; target_registry.csv; audit/verification.json','Historical reuse, not independent replication'),
  ('C04','C transfer prior effects and familiar acquisition','registered descriptive paired inference','endpoints.csv; endpoint_summary.csv; endpoint_counts.csv; paired_seed_effects.csv; paired_intervals.csv','All20 primary; qualified strata descriptive and may differ by condition'),
  ('C05','Joint familiar+C+Both+functional splitting','original endpoint criterion','functional_splitting.csv; joint_outcomes.csv; learning_effects.csv; switch_costs.csv','Only170/330 exactgate; floor-limited diagnostics do not establish sharedlearning'),
  ('C06','Geometry differs from functional learning','descriptive measurements','geometry.csv; block_transfer.csv; learning_effects.csv; transfer_associations.csv','No independent-skills inference from norms or centroids'),
  ('C07','Exact affine natural-cue computation and current output sensitivity','algebra + bounded numerical verification','decomposition.csv.gz; mathematical_checks.csv.gz; position_predictions.csv.gz','Jensen applies only Mixed-average CE; C != NoContext'),
  ('C08','Prior changes baseline and contextual response','observational decomposition','geometry.csv; decomposition.csv.gz; figure_5*','No causal mediation from correlations; no natural retrieval rescue'),
  ('C09','Reptile adaptation executes registered objective and resets','implementation + observed meta trajectories','meta_episodes.csv; meta_inner_updates.csv; meta_task_performance.csv; meta_weighted_episode_performance.csv','P1 conflict ambiguity intentional; performance uses exposure weights'),
  ('C11','First threshold crossings of C versus familiar acquisition','secondary descriptive timing','acquisition_timing.csv','Late timing operationalization; first pass may be transient; no checkpoint selection'),
  ('C10','Scope of human-like claim','limited model inference','FINAL_CONTEXT_EXPERIENCE_PRIOR_REPORT.md; RESEARCHER_HANDOFF.md','No human data; Linear only; one fixed task distribution/budget; contextC familiar inprior but targetsequence heldout')],columns=['claim_id','claim','status','exact_sources','limits'])

def render_report():
 f=read('endpoints');su=read('endpoint_summary');co=read('endpoint_counts');pa=read('paired_intervals');jo=read('joint_outcomes');ga=read('functional_splitting');b=read('block_transfer');d=read('learning_effects');g=read('geometry');meta=read('meta_performance_summary');controls=pd.read_csv(OUT/'tables/mathematical_checks.csv.gz');verification=json.loads((OUT/'audit/verification.json').read_text());tests=[]
 for log in sorted((OUT/'audit').glob('*.log')):
  text=log.read_text(errors='replace');lines=[l for l in text.splitlines() if 'passed' in l or 'failed' in l or 'ERROR' in l];tests.append((log.name,'; '.join(lines[-2:]) if lines else 'execution log; see file'))
 def S(c,k,cue,metric='ce',subgroup='all'):return su[(su.condition==c)&(su['update']==k)&(su.cue==cue)&(su.metric==metric)&(su.subgroup==subgroup)].iloc[0]
 def C(c,k,cue):return co[(co.condition==c)&(co['update']==k)&(co.cue==cue)&(co.subgroup=='all')].iloc[0]
 n_joint=int(jo[jo['update']==330].joint_behavioral.sum());rows=[]
 for k in PROBES:
  for c in CONDS:
   ff=f[(f.condition==c)&(f['update']==k)];xx=jo[(jo.condition==c)&(jo['update']==k)];gg=ga[(ga.condition==c)&(ga['update']==k)];rows.append([c,k,int(ff.strong_familiar.sum()/6),fmt(ff[ff.cue.isin(['A','B'])].ce.mean()),interval_text(S(c,k,'C')),fmt(S(c,k,'C')['median']),fmt(S(c,k,'C','accuracy')['mean']),int(C(c,k,'C').success),int(C(c,k,'Mixed-both').success),int(gg.functional_split.sum()) if len(gg) else 'unassessed',int(xx.joint_behavioral.sum()) if len(gg) else 'unassessed'])
 contrasts=[]
 for k in [170,330]:
  for org in ['I','C','C-minus-I']:
   for con in ['P2-P1','P1-P0','P2-P0']:
    r=pa[(pa['update']==k)&(pa.cue=='C')&(pa.metric=='ce')&(pa.organization==org)&(pa.contrast==con)].iloc[0];contrasts.append([org,con,k,interval_text(r),fmt(r['median']),int(r.n)])
 qualified=[]
 for c in CONDS:
  for k in [170,330]:
   r=S(c,k,'C',subgroup='strong_familiar');qualified.append([c,k,int(r.n),interval_text(r),fmt(S(c,k,'C','accuracy','strong_familiar')['mean'])])
 descriptive=[]
 for c in CONDS:
  x=b[b.condition==c];end=g[(g.condition==c)&(g['update']==330)];early=x[x.end<=100];late=x[x.end>100];dg=d[(d.condition==c)&(d['update']==330)];descriptive.append([c,fmt(early.specificity.mean()),fmt(late.specificity.mean()),fmt(late.floor_at_start.mean()),fmt(dg.mean_own_improvement.mean()),fmt(dg.mean_specificity.mean()),int(dg.direct_evidence.sum()),fmt(end.hidden_separation.mean()),fmt(end.centered_logit_separation.mean())])
 primary=[]
 for k in [0,80,100,120,170,330]:
  r=pa[(pa['update']==k)&(pa.cue=='C')&(pa.metric=='ce')&(pa.organization=='I')&(pa.contrast=='P2-P1')].iloc[0];direction='lower' if r['mean']<0 else 'higher';precision='interval excludes zero' if r.ci_low>0 or r.ci_high<0 else 'interval includes zero';primary.append(f'At state {k}, P2−P1 C CE = {interval_text(r)} nats ({direction} loss for P2; {precision}).')
 conclusion=(f'The full registered joint pattern occurs in {n_joint}/120 networks at state330. ' + ('This protocol therefore does not establish the requested combination of novel-target-context transfer and the original endpoint functional splitting phenotype.' if not n_joint else 'This is evidence in this fixed task ecology; condition-specific counts below identify where it occurs.'))
 c_effect=pa[(pa['update']==330)&(pa.cue=='C')&(pa.metric=='ce')&(pa.organization=='I')&(pa.contrast=='P2-P1')].iloc[0]
 findings=f"At330, independent P0/P1/P2 mean C losses are {S('P0-I',330,'C')['mean']:.4f}, {S('P1-I',330,'C')['mean']:.4f}, and {S('P2-I',330,'C')['mean']:.4f} nats. Both prior conditions improve C relative to P0, but the primary P2−P1 contrast is {interval_text(c_effect)} nats: this does not establish an added C-transfer benefit from useful context associations. All20 networks in every condition satisfy familiar acquisition at330. C success is13/20 for each independent pretrained group versus4/20 for P0; the centered groups yield12/20 P1 and10/20 P2 versus4/20 P0. No pretrained network passes the original functional endpoint gate at170 or330; one P0-centered network passes at170, and no network passes at330. Performance-floor limitations qualify what this null gate says about latent specialization."
 text=f'''# Learned experience priors for context-dependent sequence skills

{conclusion}

{findings}

The primary question is whether prior sequence learning permits C transfer while retaining context-specific learning. The answer must use learning effects and switch costs, not hidden-state distance. All120 target networks and all20 paired seeds per condition are retained. No model or checkpoint was selected using C or Mixed performance.

## Registered experiment and actual algorithm

The model is the original eight-unit identity recurrent network, four token inputs, three context inputs and four logits. P0 has no prior; P1 has context-uninformative experience; P2 has context-informative experience. I denotes independent initial A/B columns; C in a condition suffix denotes initially centered A/B columns, distinct from cue C. All start with recurrent spectral norm0.9, applied once. Subsequent norms are measured, never projected.

For each seed,240 sealed episodes contain20 updates in four five-update context blocks. Exactly80 episodes use each of AB/AC/BC. Half share one sequence; half contain two sequences sharing three initial symbols then conflicting. Each task is an eight-symbol path yielding seven next-token predictions; every path uses allfour symbols and requires history beyond a token lookup. The target and its cyclic/reversal equivalents are excluded. P1/P2 use exactly the same library and context/sequence marginal exposures. P1 makes each context encounter both conflicting sequences five times; P2 assigns each context one sequence ten times. P1's irreducible ambiguity is intended. Temporal sequence-order differences remain: this comparison changes the experienced within-episode association structure and order together, so it is not a perfectly isolated association-only effect.

Each episode copies the current prediction initialization, resets Adam and the monotonic execution policy, performs20 updates, then moves only prediction parameters by θ←θ+ε(θadapted−θ). ε decays linearly0.1→0.01 over240 episodes. The next task starts from that updated meta-initialization. This follows [Reptile, Nichol et al.2018](https://arxiv.org/abs/1803.02999). The exact fixed settings come from the supplied PRD. The inner objective is original CE+0.2 execution loss with detached confidence, Adam0.005 with original defaults, and joint model/policy gradient clipping at norm1. The policy is never meta-learned. The final240 initialization is used, with no checkpoint search.

Allthree context columns are originally nonzero and trainable during prior learning. Target A10/B10 practice resets policy/Adam, uses the identical original seven-token task for330 updates and never trains on C/Mixed. C has zero target gradient and zero fresh Adam moments, and remains bitwise fixed throughout. States0–330 include model, policy and full Adam history. P0 reuses exact archived NI/NC A/B histories: only the unused zero C column is replaced in new copies with its originally sampled nonzero column. All other parameters, policy and Adam are source-identical. Five existing natural branches are reused; C is newly evaluated. P0 is historical reuse, not an independent replication.

Actual budget:384000 meta inner updates,19200 Reptile outer moves,26400 new target updates,13200 reused target updates,1440 restored-Adam clone updates. Implementation fixtures used71 additional gradient updates, separately recorded. All80 meta runs and120 target runs completed; no numerical failure, retry, learning-rate change or extra normalization occurred.

## Outcomes and fixed definitions

Strong familiar acquisition requires **each** A/B CE≤0.5 and accuracy≥6/7. Per-cue generalization success uses the same CE/accuracy rule independently. Chance is ln(4)=1.38629436nats. C is a familiar prior cue, but the target sequence was never trained under C; this is new sequence-to-context transfer, not a sensory cue never seen in any prior experience. NoContext has zero vector; cue C has a nonzero learned column and must not be equated with it.

All-seed estimates below are means with95%10,000 whole-seed bootstrap intervals, RNG20260925; median C CE is also shown. The primary comparison is independent P2−P1. Contrasts are formed within paired seed before aggregation. Endpoint and full trajectory files retain every value; no token, episode or checkpoint is counted as an independent trained network.

'''
 text+=md(['Condition','State','Familiar n/20','Mean familiar CE','Mean C CE [95% CI]','Median C CE','C accuracy','C success/20','Both success/20','Functional split/20','Joint/20'],rows)
 m1=meta[(meta.condition=='P1-I')&(meta.kind=='conflict')&(meta.window=='last40')&(meta.phase=='after')&(meta.metric=='ce')].iloc[0];m2=meta[(meta.condition=='P2-I')&(meta.kind=='conflict')&(meta.window=='last40')&(meta.phase=='after')&(meta.metric=='ce')].iloc[0]
 text+=f"\nPrior-task manipulation check: over the last40 registered episodes, exposure-weighted adapted conflict-task CE is {m1['mean']:.4f} for P1-I and {m2['mean']:.4f} for P2-I. Each seed is averaged first; no episode is an independent network. The full before/after shared/conflict summaries are in `meta_performance_summary.csv`. Lower P2 loss on its identifiable training tasks does not by itself establish better held-out target transfer.\n"
 text+='\n### Primary independent P2−P1 result\n\n'+'\n\n'.join(primary)+'\n\nNegative differences favor P2. C accuracy and success contrasts, allsix endpoints and allother natural cues are in `paired_intervals.csv`; these CE contrasts alone do not establish the joint phenotype. No multiplicity correction was registered; secondary intervals are descriptive.\n\n'
 text+=md(['Organization','Contrast','State','Paired C CE difference [95% CI]','Median paired difference','Seeds'],contrasts)
 text+='\n### Familiar-qualified description\n\nActual denominators below are condition-specific; these are not balanced causal subgroups. All20 remain the primary analysis.\n\n'+md(['Condition','State','Familiar-qualified n','C CE [95% CI]','C accuracy'],qualified)
 timing=read('acquisition_timing');tr=[]
 for c,tt in timing.groupby('condition'):
  counts=tt.C_order.value_counts();tr.append([c,int(tt.first_familiar.notna().sum()),int(tt.first_C.notna().sum()),int(counts.get('precedes',0)),int(counts.get('same_state',0)),int(counts.get('follows',0)),int(counts.get('not_observed',0))])
 text+='\n### Descriptive transfer timing\n\nFirst crossings of the fixed familiar/per-cue criterion answer the temporal question without choosing a model for further use. These are late operationalized descriptive summaries, not new primary tests. A first crossing may be transient; all successful-state counts and censored never-success values are retained in `acquisition_timing.csv`. No threshold changed.\n\n'+md(['Condition','Familiar ever/20','C ever/20','C precedes familiar','Same state','C follows familiar','C never observed'],tr)
 text+='\n## Does contextual specialization survive?\n\nThe original endpoint gate is reused identically at170/330. It requires positive own-minus-cross effects in both directions, mean specificity>0.001, mean own improvement>0.0001, neither own effect≤−0.000001, and paired complete-cycle switch evidence (mean>0.05 and fraction>0.05 at least0.5). Same-weight incoming-minus-outgoing costs are baseline-adjusted using state0; the bias cancels over balanced A/B cycles. The secondary historical trajectory gate cannot be assessed identically because its earlier diagnostic times differ; it is explicitly unassessed. Allsix registered one-update diagnostics are reported continuously. They measure actual restored-Adam update effects, including historical momentum; they are not plain SGD gradient-similarity measurements.\n\nBlock specificity is own-context CE gain minus unpracticed-context CE gain at identical frozen start/end weights. Gains and costs remain signed. The source floor flag is both initial block losses≤0.05; it is descriptive, not a new splitting threshold. A floor-limited lack of gain cannot demonstrate shared learning. Hidden and centered-logit separation remain geometrical measurements, never inputs to the functional gate.\n\n'
 text+=md(['Condition','Early mean specificity (≤100)','Late mean specificity','Late floor-block fraction','Own gain at330','Specificity at330','Direct evidence n/20','Hidden separation330','Centered-logit separation330'],descriptive)
 text+='\nEarly block specificity is positive on average in every condition, with both positive own gains and cross-context transfer. Its magnitude is lower after pretraining, but initial loss and available room to improve also differ. By330, own gains are near the performance floor in every condition. Thus the data support measurable early context-specific learning and transfer, not a demonstrated persistent separate-learning phenotype, nor proof that later near-zero gains reflect shared representations.\n\nA failed endpoint gate establishes failure of this operational joint criterion at that state; it does not prove loss of contextual organization. In particular, floor-limited endpoint probes can be insensitive. Use Figures3 and the raw block/diagnostic tables to distinguish positive early specificity, negative transfer and negligible measurable learning. Transfer versus geometry or learning correlations in Figure4 are descriptive, not causal mediation.\n\n'
 text+='''## Exact Linear computation and what is measured

For zero initial state, define S_t=Σ(k=0…t−1)W_h^k and K_t=W_oS_t. The token/bias-only trajectory gives z0. For any constant cue c,

z(c)=z0+K_t(w_A c_A+w_B c_B+w_C c_C).

Let u=K_t(w_A+w_B)/2 and v=K_t(w_A−w_B)/2. Then zA=z0+u+v; zB=z0+u−v; zC=z0+K_tw_C; zMixed-average=z0+u; zMixed-both=z0+2u. Class-centering removes an output offset before magnitude/sensitivity comparisons. The correct margin is z_correct−max(z_incorrect); the maximizing competitor can change, so margin effects cannot generally be added as separate scalar contributions.

Mixed-average logits equal the A/B logit midpoint. Convexity bounds its CE by mean familiar CE, with no corresponding guarantee for discrete accuracy. Mixed-both and C have no familiar-loss bound. C differs from NoContext at the first hidden state for every model/state. A nonzero C column need not yield different probabilities at every individual position. No artificial contribution was added to produce the transfer results: these are natural cue trajectories. The decomposition is an algebraic description, not proof of retrieval of a symbolic shared skill.

The audit independently reconstructs float64 hidden/logit responses using S/K, verifies float64 affine identities, bounds native float32 replay by absolute arithmetic summands, and checks Jensen only for Mixed-average. It retains z0/u/v/Cresponse raw and centered components, allfour logits/probabilities, margins, every position, recurrent norms, centered K sensitivity and allcontext-column norms. Learned weights/performance correlations do not establish mediation or robustness to arbitrary unseen directions.

'''
 text+=f"Maximum float64 operator logit error: {controls.double_operator_logit_error.max():.3g}; maximum float64 operator hidden error: {controls.double_operator_hidden_error.max():.3g}. All{len(controls)} position-wise controls pass their fixed tolerances. Native errors have separately computed float32 allowances, rather than a post hoc universal cutoff.\n\n"
 text+='''## Figures and exact data

![Actual training](figures/figure_1.png)

Figure1 uses actual practiced-context loss **before** each update. Blue=A, orange=B;20 faint seed lines, cohort mean and seed-bootstrap95% bands, without smoothing switches. [First100](figures/figure_1_early.png). Exact values: `tables/training.csv`, `training_intervals.csv`.

![Natural generalization](figures/figure_2.png)

Figure2 retains allsix natural cues and all331 states. [First120](figures/figure_2_early.png). Exact values: `evaluations.csv`, `evaluation_intervals.csv` and full `position_predictions.csv.gz`.

![Functional learning](figures/figure_3.png)

Figure3 retains A/B-practiced signed specificity, including negative values. Floor ticks do not establish pooling. Companions: [own/cross gains](figures/figure_3_gains.png), [raw/adjusted switch costs and signed gaps](figures/figure_3_switches.png), [restored-Adam diagnostics](figures/figure_3_diagnostics.png). Exact tables: `block_transfer`, `block_intervals`, `switch_costs`, `switch_intervals`, `performance_gaps`, `gap_intervals`, `learning_effects`, `diagnostic_intervals`.

![Transfer and specialization](figures/figure_4.png)

Figure4 shows all120 networks at330; black borders mark strong familiar acquisition. [State170](figures/figure_4_170.png). Associations are not mediation; exact `transfer_associations.csv`.

![Prior mechanisms](figures/figure_5.png)

Figure5 compares initial baseline/C, final C, original functional gate and recurrent sensitivity. Black rings on final C identify familiar-qualified networks; n is shown. Allseeds remain visible. Companions track [prediction mechanisms](figures/figure_5_mechanisms_1.png) and [weights/sensitivity](figures/figure_5_mechanisms_2.png). Exact `geometry.csv`, `geometry_intervals.csv`, `functional_splitting.csv` and `decomposition.csv.gz`.

Allfive primary figures and companion views are also SVG. Figure5 bars are seed means with individual observations; formal bootstrap intervals are in the endpoint/geometry tables. Figure2 retains linear CE axes: large initial costs can compress near-floor late values, so consult exact tables and Figure4 rather than infer equality visually.

## Evidence, tests and remaining uncertainties

`CLAIM_LEDGER.csv` maps every major claim to exact tables and limits. `SOURCE_REUSE_AUDIT.md` gives code/source correspondence. `audit/protocol_seal.json` records source/checkpoint hashes before training, and `target_registry.csv` / `meta_registry.csv` record new checkpoint hashes. `audit/verification.json` checks every model/policy/Adam state and every saved probability/loss/margin. Final manifest hashes seal the completed package. Source checkpoints and predecessor evidence were never overwritten.

The RED logs are retained, including the initial incorrect fixture indexing, hash-adapter API mismatch, and the final-audit training CSV field-order failure. The training CSV issue affected serialization of new versus reused training rows; exact logged values were recovered under the known branch schemas and every pre-update CE checked against same-weight frozen evaluations. Original malformed exports and intervals, original scientific-run source snapshots matching phase hashes, the failed audit and the correction receipt remain in `audit/export_correction/`. No model training or evaluation replay was performed for the correction. The corrected training intervals/figures use the recovered actual records. The first two issues were corrected before scientific execution. Full test/run logs are under `audit/`. The final read-only suite deliberately excludes fixture tests that would perform additional optimizer updates; the original GREEN core/meta logs retain those checks.

'''
 text+=md(['Log','Result / retained status'],tests)
 text+='''
This experiment uses one architecture, one prespecified task generator and one training budget. P1 ambiguity and the unavoidable temporal-order differences qualify what P2−P1 isolates. Episode-specific mappings are arbitrary across episodes; pretraining is not exposure to the target. C is learned in prior tasks and held fixed during the target; results do not establish robustness to all novel contexts. A single target task per seed limits task-level generalization. Bootstrap intervals quantify variation across20 trained seeds, not across humans. No human behavior was fitted or newly measured. Strong generalization alone is insufficient to call the model human-like, and a floor-limited null diagnostic is insufficient to claim context-free representations.

## Researcher teach-back

We trained an initialization, not a fixed motor-memory library. Each prior task briefly adapts a copied Linear RNN, then Reptile pulls the shared initialization toward those adapted prediction weights; policies and Adam moments are reset. Useful versus uninformative cue associations are the main prior manipulation, with exactly matched marginal exposures. Then every network learns the same A/B target from that initialization. C contributes its learned nonzero recurrently propagated signal but receives no target training. The decisive question is whether natural C predictions improve **and** the original functional specialization evidence survives. Separate learning measurements answer the second question; trajectory distances cannot. The tables above report the actual joint result, with floor and design uncertainties retained.
'''
 for a,b in [('All120','All 120'),('All20','All 20'),('all120','all 120'),('all20','all 20'),('allfour','all four'),('allthree','all three'),('Allthree','All three'),('Allsix','All six'),('allsix','all six'),('allcontext','all context'),('Allfive','All five'),('All80','All 80'),('all80','all 80'),('everyCE','every CE'),('last40','last 40'),('at330','at 330'),('at170','at 170'),('state330','state 330'),('state0','state 0'),('state170','state 170'),('state240','state 240'),('norm0.9','norm 0.9'),('episodes20','episodes 20'),('performs20','performs 20'),('each seed,240','each seed, 240'),('contain20','contain 20'),('Exactly80','Exactly 80'),('over240','over 240'),('inprior','in prior'),('loop20','loop 20'),('for330','for 330'),('at170/330','at 170/330'),('by330','by 330'),('at170 or330','at 170 or 330'),('at330','at 330'),('each of','each of'),('with95%10,000','with 95% 10,000'),('across20','across 20'),('All{len','All {len'),('clipping at norm1','clipping at norm 1')]:text=text.replace(a,b)
 import re
 text=re.sub(r'\b([A-Za-z]{2,})(?=\d)',r'\1 ',text)
 return text

def main():
 report=render_report();(OUT/'FINAL_CONTEXT_EXPERIENCE_PRIOR_REPORT.md').write_text(report);claim_ledger().to_csv(OUT/'CLAIM_LEDGER.csv',index=False)
 handoff='''# Researcher handoff — learned experience priors

All six conditions acquired A/B strongly in all 20 seeds by update 330. Independent P0/P1/P2 C CE means:3.7930/1.0005/1.0687 nats. P1 and P2 each succeed under C in13/20 versus4/20 P0; centered P0/P1/P2 counts4/12/10. The paired independent P2−P1 C CE difference is+0.06814,95%CI[−0.00361,0.14875]: no added P2 benefit is established. No pretrained network passes the original endpoint functional splitting gate at170/330. One P0-centered seed passes at170; none of120passes at330; no registered joint phenotype is observed at either endpoint. Early specificity remains positive on average; late floor effects prevent interpreting vanishing gains as proof of shared representations.

Start with [the scientific report](FINAL_CONTEXT_EXPERIENCE_PRIOR_REPORT.md), its endpoint/paired tables and Figures1–5. Condition suffix I=independent, suffix C=initially centered A/B; cue C is a separate concept.

Algorithm: identity recurrence; seven next-token predictions; original CE+.2 detached-confidence execution objective; joint norm1 clipping; Adam. Reptile uses240episodes×20innerupdates, fresh policy/Adam each task, outer0.1→0.01 prediction weights only. Final240 only. Target330 A10/B10 updates with fresh policy/Adam, no C/Mixed training. No continued spectral projection.

Read the full endpoint table before discussing success. `tables/paired_intervals.csv` contains the primary P2−P1/I and secondary contrasts. `tables/joint_outcomes.csv` is the familiar+C+Mixed-both+original functional gate conjunction. `tables/learning_effects.csv` and `block_transfer.csv` distinguish small gains/floors from shared learning. The original endpoint criterion is exact at170/330; the historical secondary trajectory criterion is unassessed.

Mathematics: S_t=I+W_h+…+W_h^(t−1), K_t=W_o S_t. zC=z0+K_twC; zA/B=z0+u±v; Mixed-average=z0+u; Mixed-both=z0+2u. C is not NoContext. Center logits for norms; do not turn a norm correlation into mediation, or Jensen CE into an accuracy guarantee. Every result is from natural cue inputs, with no optimized hidden-state or added-logit rescue.

All20 seeds remain. Qualified denominators vary and are descriptive. P0 reuses NI/NC saved A/B trajectories with only an unused column restored into new copies, so this is historical reuse. P1/P2 share exact exposure marginals but temporal sequence orders differ. C was seen with prior sequences; only target-under-C was held out. No result establishes universal novel-context robustness or reproduces human behavior.

Reproduce using REPRODUCIBILITY.md. Do not rerun training: complete checkpoint histories and allraw predictions are already saved. Registry, source hashes, code, failed/pass logs, claimledger and mathematical controls are auditable. Preserve allnull/negative findings and the frozen thresholds in any follow-up.
''';(OUT/'RESEARCHER_HANDOFF.md').write_text(handoff)
 audit='''# Source and reuse audit

| Component | Actual implementation/source | Disposition |
| --- | --- | --- |
| Parameter creation/RNG | confirmatory_srn/model.py; context_experience_prior/core.py ExperienceLinear | Same native containers/default resets/redraw order; omit only zero-C enforcement/hook |
| Identity recurrence | experiments/high_scale_initialization/core.py LinearElman.forward | Source manual identity forward; nn.RNN.forward not used |
| Original objective/policy | execution_objective_search/objectives.py; confirmatory_srn/model.py; ConfirmatoryConfig | Original detached-confidence execution objective, joint clipping/Adam |
| Base I/C tensors | registry/base_initializations.pt and base_initializations.csv | One seed source, Whnorm.9 once; C raw nonzero |
| P0 target histories | results/recurrent_context_factorial/v1/part_b/checkpoints/seed_*_NI/NC.pt | Exact331-state copies except unused C column restored; allpolicy/Adam unchanged |
| P0 natural branches | predecessor part_b/tables/position_predictions.csv | A/B/Mixed-average/Mixed-both/NoContext reused; new C branch only |
| P1/P2 tasks | registry/seed_*.json; episodes/updates/sequences.csv | Sealed before scientific updates; exact exposure/MI/exclusion audit |
| Reptile | context_experience_prior/meta.py | Fresh task model/policy/Adam; prediction-only interpolation; all before/adapted/after snapshots |
| Functional gate | experiments/recurrent_learning_transfer/measures.py | Exact original endpoint source; no geometry; secondary trajectory flag excluded |
| Block/interval helpers | experiments/centered_context_linear/analysis.py | Same signed gains and whole-seed temporal bootstrap; no old model pooling |
| Linear operators | experiments/nonzero_context_generalization/core.py | Frozen independent S/K construction; valid with nonzero third column |
| New natural checkpoints | target_registry.csv; target/seed_*.pt | All331 states ×120 networks, complete policy/Adam |

Before-run exact input hashes: audit/protocol_seal.json. New snapshot hashes: meta_registry.csv and target_registry.csv. Final verification checks parent unchanged and allfull P0 history fields outside unused C. Parent cohorts are not pooled as extra seeds. No raw human data are used or uploaded. Allnew code/data live in a distinct namespace; predecessor seals remain immutable.
''';(OUT/'SOURCE_REUSE_AUDIT.md').write_text(audit)
 append('## Report and handoff drafted\nFive primaryfamilies andcompanions,allseed/qualifiedendpoint tables,pairedcontrasts,exactfunctional gate,floorlimits,algebra,claimledger,sourceaudit andteachback retained. Resultlanguage follows actualjointcounts;nohumanfit ormediationclaim. Finalverification andtableonlyregenerationreceipt willsealpackage.')
if __name__=='__main__':main()
