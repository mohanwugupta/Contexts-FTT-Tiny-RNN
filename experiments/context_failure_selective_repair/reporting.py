"""R5: independent table arithmetic, conservative claims and deterministic artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from context_sequence.provenance.hashing import sha256
from .core import ROOT,CONTRACT,read,save,report,dump,table,bootstrap
from .factorial import summarize
from .repair import seed_contrasts,selective_gate
from .decision import validate_claim
from .figures import generate
FAILURE='We localized effects of state history and context on future performance, but did not identify a unique disruption whose correction explains the natural Mixed deficit.'


def audit_tables(src):
 f=read(src,'r1_branch_rollouts');cells=read(src,'r1_cells');nat=read(src,'r0_native_replay');con=read(src,'r3_seed_contrasts');patch=read(src,'r3_interventions');rev=read(src,'r3_reverse');r1c=read(src,'r1_seed_contrasts');stats=read(src,'r3_bootstrap')
 if len(cells)!=360 or len(f)!=1440 or sorted(cells.seed.unique())!=list(range(20)):raise ValueError('Incomplete factorial')
 for q in cells.itertuples():
  t=f[(f.seed==q.seed)&(f.k==q.k)&(f.source_context==q.source_context)&(f.branch==q.branch)].sort_values('transition');ref=tuple({'raw_ce':nat[(nat.seed==q.seed)&(nat.cue==g)].sort_values('transition').raw_ce.to_numpy()} for g in ['A','B'])
  v=summarize(t,ref)
  for metric in ['current_ce','future_ce','full_remaining_ce','worst_future_ce']:
   if abs(v[metric]-getattr(q,metric))>1e-12:raise ValueError('R1 table summary mismatch')
  if v['viable_continuation']!=q.viable_continuation:raise ValueError('Criterion mismatch')
 for family,context,sign,frame in [('forward','Mixed',1,patch),('reverse','Mixed',-1,rev)]+[('familiar',g,s,read(src,'r3_familiar_controls').query('context==@g and sign==@s')) for g in ['A','B'] for s in [-1,1]]:
  new=seed_contrasts(frame);old=con[(con.group==family)&(con.context==context)&(con.sign==sign)].sort_values(['k','seed','contrast']).reset_index(drop=True);new=new.sort_values(['k','seed','contrast']).reset_index(drop=True)
  if not np.allclose(new.effect,old.effect,equal_nan=True,atol=1e-12,rtol=0) or not np.array_equal(new.n_references,old.n_references):raise ValueError('Paired per-seed arithmetic mismatch')
 for (family,context,sign,k,contrast),df in con.groupby(['group','context','sign','k','contrast']):
  if sorted(df.seed)!=list(range(20)):raise ValueError('Seed coverage dropped')
  new=bootstrap(df);old=stats[(stats.group==family)&(stats.context==context)&(stats.sign==sign)&(stats.k==k)&(stats.contrast==contrast)].iloc[0]
  for metric in ['mean','ci_low','ci_high']:
   if not np.isclose(new[metric],old[metric],equal_nan=True,atol=1e-12,rtol=0):raise ValueError('Bootstrap mismatch')
 # Independent R1 history contrasts equal exact full-reference interventions.
 for k in [3,5]:
  a=con[(con.group=='forward')&(con.k==k)&(con.contrast=='reference_minus_natural')].sort_values('seed');b=r1c[(r1c.k==k)&(r1c.endpoint=='future_ce')&(r1c.contrast=='history_mean_AB')].sort_values('seed')
  if not np.allclose(a.effect,b.effect,atol=1e-12,rtol=0):raise ValueError('Full-reference reuse contradiction')
 for (seed,k,side),df in patch.groupby(['seed','k','reference']):
  q=df[df.arm.isin(['U','W','random'])&(df.status=='valid')]
  if len(q):
   if len(q)!=3 or q.amplitude.nunique()!=1 or np.ptp(q.realized_norm)>1e-10:raise ValueError('Matched control violation')
   if not q.current_softmax_valid.eq(True).all() or q.softmax_max_difference.max()>1e-10:raise ValueError('Current softmax violation')
 allrows=pd.concat([patch,rev,read(src,'r3_familiar_controls')]);applied=allrows[allrows.state_applied.eq(True)]
 if len(applied)!=33:raise ValueError('Patch budget ledger mismatch')
 return dict(r1_cells=len(cells),r1_transition_rows=len(f),all20_retained=True,seed_unit='seed',matched_k3_seed_count=patch[(patch.k==3)&(patch.arm=='U')&(patch.status=='valid')].seed.nunique(),reverse_valid_cells=int(((rev.arm=='U')&(rev.status=='valid')).sum()),actual_new_patch_rollouts=len(applied),paired_differences_recomputed=True,bootstrap_recomputed=True,source_calls=0)


def products(src,dest,figures=True):
 src=Path(src);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 for name in ['tables','figures','audit']:(dest/name).mkdir(exist_ok=True)
 verification=audit_tables(src);dump(dest/'audit/r5_table_verification.json',verification)
 nat=read(src,'r0_native_replay');r1=read(src,'r1_bootstrap');r3=read(src,'r3_bootstrap');patch=read(src,'r3_interventions');rev=read(src,'r3_reverse');fam=read(src,'r3_familiar_controls');gaps=read(src,'r2_natural_gap_components');sensitivity=read(src,'r2_future_sensitivity');cases=read(src,'r1_reference_cases');con=read(src,'r3_seed_contrasts');decision=json.loads((src/'audit/r3_decision.json').read_text())
 primary=r3[(r3.group=='forward')&(r3.k==3)];gate=selective_gate(primary,decision['softmax_violations'])
 if gate!=decision['selective_mechanism_gate']:raise ValueError('Gate disagreement')
 summary=r1[r1.contrast.isin(['history_mean_AB','history_A','history_B','cue_A','cue_B','interaction_A','interaction_B'])].copy();summary['program']='R1';summary2=r3[r3.contrast.isin(['U_minus_natural','U_minus_W','U_minus_random','reference_minus_natural','silent_minus_natural','visible_minus_natural'])].copy();summary2['program']='R3';summary2['endpoint']='future_ce';save(dest,'r5_summary',pd.concat([summary,summary2],ignore_index=True))
 seedtable=con[(con.group=='forward')&con.contrast.isin(['U_minus_natural','U_minus_W','U_minus_random','reference_minus_natural'])].copy();save(dest,'r5_primary_seed',seedtable)
 # Quantization audit derived only from the saved proposed/applied state vectors.
 quant=[]
 for q in patch[patch.state_applied.eq(True)].itertuples():
  h=gaps[(gaps.seed==q.seed)&(gaps.k==q.k)&(gaps.reference==q.reference)&(gaps.reference_kind=='factorial')].iloc[0];anchor=np.array([h[f'anchor_{j}'] for j in range(8)]);state=np.array([getattr(q,f'state_{j}') for j in range(8)]);cast=state.astype(np.float32).astype(float)
  quant.append(dict(seed=q.seed,k=q.k,reference=q.reference,arm=q.arm,realized_exact_norm=np.linalg.norm(state-anchor),source_cast_norm=np.linalg.norm(cast-anchor.astype(np.float32).astype(float)),cast_state_error=np.max(abs(cast-state)),amplitude=q.amplitude))
 save(dest,'r5_source_cast_audit',pd.DataFrame(quant))
 # Full denominators; failures/infeasibility never become invented outcomes.
 coverage=[]
 for k in [3,5]:
  for arm in ['natural','reference','visible','silent','U','W','random']:
   f=patch[(patch.k==k)&(patch.arm==arm)];v=f[f.status=='valid'];coverage.append(dict(k=k,arm=arm,n_seed_denominator=20,n_reference_denominator=40,n_valid_seed=v.seed.nunique(),n_valid_references=len(v),n_viable_seed=v[v.viable_continuation.eq(True)].seed.nunique(),n_viable_reference=int(v.viable_continuation.eq(True).sum())))
 cov=pd.DataFrame(coverage);save(dest,'r5_coverage',cov)
 claims=[
 ('C1','Natural physical Mixed has higher cohort mean next-token CE than familiar contexts.','tables/r0_native_replay.csv','Descriptive all20, seven teacher-forced transitions.','Familiar B itself is imperfect; unequal familiar competence.','verified descriptive'),
 ('C2','Familiar history with current/future Mixed provides no clear average k3 future advantage.','tables/r1_bootstrap.csv','History mean future difference+.2623, interval[-.2221,.8161].','Five seeds have ≥1 viable reference; no evidence that all histories fail.','inconclusive mean advantage'),
 ('C3','Within familiar history, the current Mixed cue changes subsequent performance under identical Mixed future inputs.','tables/r1_seed_contrasts.csv','Cue_A+.7493; cue_B+.6454 nats; current/future separate.','Unadjusted descriptive intervals; not a unique hidden mediation pathway.','finite within-model causal contrast'),
 ('C4','Every current output map has rank3/kernel5; invisible changes can have familiar future-logit sensitivity.','tables/r2_output_kernel_audit.csv','Full SVD rank3/kernel5; future spectra/chain/source-FD audits stored separately.','Local Jacobian is not task information; saturated float32 finite differences can be inaccurate.','verified algebra; local sensitivity only'),
 ('C5','The prespecified selective-repair gate did not pass.','tables/r3_bootstrap.csv','Primary coverage1/20 seeds, U−W and U−random adverse in that seed.','Most interventions infeasible; cannot infer absence of a useful silent computation.','selective mechanism not identified'),
 ('C6','Reverse necessity comparison is unestimable under fixed matched feasibility.','tables/r3_reverse.csv','Zero feasible matched reverse cells at k3 and k5.','Viable observed references exist; no reverse behavioral effect measured.','blocked by registered geometry'),
 ('C7','No new training experiment is justified by v1.','tables/r3_interventions.csv','No selective mechanism, zero model/optimizer updates; original hashes verified.','A separate preregistered intervention family could be informative.','no new training'),
 ('C8','Current output-silent applied interventions preserve allfour current probabilities within double/float32 contracts.','tables/r3_interventions.csv','Every forward silent applied state audited before source future scoring; allfuture tokens/cues matched.','Float32 identity is bounded roundoff, not bitwise equality.','verified numerical scope')]
 ledger=[]
 for cid,claim,source,evidence,alternative,verdict in claims:
  validate_claim({'asserts':[term for term in ['natural retrieval','lost sequence information','global unreachability','human mechanism','attractor'] if term in claim.lower()]});ledger.append(dict(claim_id=cid,claim=claim,source_table=source,source_sha256=sha256(src/source),evidence=evidence,alternative_or_counterexample=alternative,verdict=verdict))
 claimdf=pd.DataFrame(ledger);save(dest,'r5_claim_sources',claimdf)
 report(dest,'CLAIM_LEDGER.md','# Claim ledger\n\n'+table(claimdf)+'\n\nThe source-to-claim table stores exact SHA256 identities. The full source/checkpoint map is audit/frozen_inputs.json and tables/r0_checkpoints.csv. No claim of natural retrieval, lost sequence information, global unreachability, a human mechanism or an attractor is licensed. These terms occur here only to delimit unsupported interpretations.\n')
 means=nat.groupby('cue').raw_ce.mean();r1primary=r1[(r1.k==3)&r1.contrast.isin(['history_mean_AB','cue_A','cue_B','interaction_A','interaction_B'])];r3primary=primary[primary.contrast.isin(['U_minus_natural','U_minus_W','U_minus_random','reference_minus_natural','silent_minus_natural','visible_minus_natural'])]
 allsilent=pd.concat([patch,rev,fam]);allsilent=allsilent[(allsilent.status=='valid')&allsilent.arm.isin(['U','W','random','silent'])];maxdouble=allsilent.softmax_max_difference.max();max32=allsilent.source_float32_softmax_difference.max()
 counts=cases.groupby('k')[['n_viable','any_viable','any_current_correct','any_current_correct_then_error']].sum().reset_index()
 phase_status=json.loads((src/'audit/phase1k_final_status.json').read_text()) if (src/'audit/phase1k_final_status.json').exists() else {'status':'pending'}
 regression_text='34 new tests and57 independent read-only predecessor tests passed. Phase1K: '+('18 read-only tests passed;2 acquisition-training tests excluded under the no-training instruction.' if phase_status.get('returncode')==0 else 'collection/import remained blocked in bounded retries; exact status '+json.dumps(phase_status)+'. No Phase1K pass is claimed. Source and predecessor data were not modified.')
 lineage='The initial v1 R2 run was superseded by v1_1 after a precision error in a synthetic test and a shell failure to stop on RED. The original failing logs and test snapshot remain in v1; only the test’s weight averaging was corrected from float32 to float64, with tolerance unchanged. The scientific algorithms and thresholds did not change. R0/R1 outputs were copied byte-for-byte and hash-verified; R2 was rerun only after GREEN. The accepted scratchpad uses PRD sections because the named template was unavailable; the user explicitly accepted this.'
 tests=[]
 for p in sorted((src/'audit').glob('*.log')):
  lines=p.read_text(errors='replace').splitlines();tail=' | '.join(lines[-2:])
  tests.append(dict(log=str(p.relative_to(src)),sha256=sha256(p),last_lines=tail))
 save(dest,'r5_test_evidence',pd.DataFrame(tests))
 final=f'''# Context failure and selective repair — final report

**What failed?** Natural physical Mixed produces mean next-token CE {means['Mixed']:.6f} nats versus A {means['A']:.6f} and B {means['B']:.6f}. Holding the current and future cue at Mixed, familiar inherited histories do not give a clear mean k3 future advantage: the paired A/B-reference mean difference is +0.262313 nats,95% trained-seed interval[-0.222091,0.816107]. Changing only the current familiar cue to Mixed increases subsequent mean CE within A history by0.749261[0.329812,1.183521] and within B history by0.645418[0.005686,1.404572]. The latter cue contrasts have descriptive unadjusted intervals. The consequences of the current input depend on inherited state; these experiments do not identify a unique hidden pathway.

**What was disrupted?** No selective recurrent computation was identified. The output map has a five-dimensional current-softmax kernel. Familiar-only derivatives define a fixed two-dimensional future-sensitive subspace, but projected natural-gap norms are measurements of local sensitivity, not amounts of sequence information lost or CE caused.

**What caused the deficit?** The selective causal explanation remains unestablished. At k3 only1/40 A/B reference cells, representing1/20 seeds, permits the registered matched U/W/random intervention. This seed12/A case has U−natural=+3.828413e-6 nats, U−W=+5.567129e-6 and U−random=+1.077161e-5: tiny adverse effects. The full-cohort selective gate fails. No reverse matched perturbation is feasible at either position. This is a feasibility-limited outcome, not evidence of a zero reverse effect or of absent hidden capacity.

**What can be learned from this?** Recommend **no new training experiment**. First inspect the fixed rays against the saved tanh-interior geometry under a separate diagnostic protocol. Do not relax the margin, clip states, retune dimensions, resample controls or train until a favorable result appears.

**{FAILURE}**

## Frozen protocol and exact evidence

Exactly20 original canonical High_monotonic checkpoints at original170; no C0/C1–C3, Low, D3 endpoint or sequence-only pooling. Seven teacher-forced token inputs(0,1,2,0,3,1,2) predict(1,2,0,3,1,2,3). h0=0; k is the output after consuming x_k. A=(1,0,0), B=(0,1,0), physical Mixed=(.5,.5,0). Third cue weights are zero. Only original next-token logits are scored; the separate monotonic execution policy is not a prediction head.

R0 verifies20 checkpoint hashes and all predecessor scientific seals/source files. Sixty natural A/B/M rollouts and16 representative branch tails were used only for fidelity at1e-12 CSV tolerance; every reused natural baseline was checked. R1 reuses all360 exact P4 pulse cells and1440 transition rows, with **zero new factorial scientific evaluations**. The optional Phase1K CSVs are cloud-only; no observed digest is invented. The available P4 evidence supplies every required exact branch. Phase1K native-restoring pulses, persistent takeovers and geometric scores are not silently substituted.

Allnine inherited-state × current-cue cells appear at both k3(primary,4 future predictions) and k5(secondary,2 future predictions). All subsequent cues are physical Mixed. A/A and B/B cells have familiar current outputs but counterfactual Mixed futures; they are not full-familiar trajectories. Both A-history/Mixed-update and B-history/Mixed-update references are retained, never selected by CE. Natural full-A/full-B gap references are separate diagnostics in R2.

## R1 — current cue, history and future

{table(r1primary)}

Negative CE differences favor the first branch. Inference resamples20 whole trained seeds,10,000 draws,RNG20260925. Histories, branch cells, transitions and references are repeated measures. Future CE excludes current k; full-remaining CE includes it as a secondary endpoint. Familiar viability is the unchanged predecessor `future_summary`: future mean CE≤position-specific mean familiar A/B future CE+.10, equality passes. Zero-slack and worst-step outcomes are exported.

{table(counts)}

Counts are seed-level and nonexclusive except n_viable, which counts both reference cells. At k3,7/40 references across5/20 seeds are viable. Sixteen seeds have at least one correct current reference prediction;15 have at least one such reference followed by a later error. This does not by itself demonstrate loss of memory. At k5,21/40 references across16/20 seeds are viable; the shorter horizon and position differ, so this is secondary rather than a replication of the same estimand.

## R2 — equations and numerical meaning

The source recurrence is h_t=tanh(W_h h_(t−1)+W_x x_t+W_c c_t+b_h), z_t=W_o h_t+b_o. Let P=I−11ᵀ/4 and O=PW_o. Full right SVD yields V spanning row(O), N spanning ker(O). For d=r−h_M, d_visible=VVᵀd and d_silent=NNᵀd; their sum equals d. O d_silent=0 implies z′−z is constant across classes, hence **allfour** softmax probabilities and every possible current-target CE are identical in exact arithmetic. Current-output identity does not imply future identity, because O diag(1−h_next²)W_h N need not vanish.

For familiar g∈{{A,B}}, G_g(k)=∂vec(Pz_(k+1),…,Pz_7)/∂h_k. It uses natural familiar poststates, the same known future input tokens and that familiar cue, with frozen weights and **no target labels or Mixed CE**. Stack G_A/G_B; SVD of G_fam N gives fixed U=NQ[:,:2] and W=NQ[:,2:]. Both references are projected only after this geometry is frozen. All40 seed/k top-two subspaces pass the fixed singular-value/gap identification test; dimension2 is never selected by outcome.

Independent double autodiff/chain maximum disagreement={sensitivity.chain_error.max():.3e}. Nonsaturated source finite-difference fixtures pass at1e-3 and1e-4. Across real saturated source states, maximum absolute derivative discrepancies are{sensitivity['fd_0.001_max_error'].max():.6f} and{sensitivity['fd_0.0001_max_error'].max():.6f}; maximum relative discrepancies are{sensitivity['fd_0.001_relative_error'].max():.6f} and{sensitivity['fd_0.0001_relative_error'].max():.6f}. Boundary one-sided flags and raw Jacobians are retained. These are numerical limitations of local double sensitivities compared with finite float32 changes, and weaken fine orientation claims; no scales or thresholds were tuned. Behavior is always scored with the original float32 source, not the double derivative model.

## R3 — selective intervention, feasibility and negative results

At natural Mixed poststate h and each fixed reference r, d=r−h. U direction is UUᵀd. The W gap projection is rescaled to the same norm; only if numerically zero is its hash fallback used. Random is a hash-seeded direction in N with that same norm. Hash identity=(20261113,seed,k,reference,control_kind). For the three matched rays, compute the largest joint feasible scale≤1 from the strict interior |h_j|≤1−1e-6, then multiply by.95. The actual interval can have a positive lower bound because a natural float32 tanh state may already lie at±1. If the safety factor violates that bound, the comparison is infeasible. No clipping or direction replacement is permitted. Visible/full-silent arms use the same scale where feasible. Exact observed full references are reused bounds, with boundary/interior status explicit; boundary states are not portrayed as strict-interior applied patches.

{table(cov)}

n_valid_seed means at least one valid reference, not complete both-side support. At k3 onlyseed12/A is matched feasible; k5 has both sides for seeds2,6,18. Only3/40 and5/40 full observed references respectively are in the registered strict interior. The rest remain exact observed source data. Most proposals are infeasible; failed cells are never assigned zero CE or dropped from status denominators. The short-horizon k5 effects remain conditional and cannot repair the primary coverage gate.

{table(r3primary)}

A one-seed bootstrap has degenerate endpoints; it supplies no population uncertainty estimate and is **not** evidence for a cohort effect. All available current-softmax checks pass: largest applied silent double discrepancy={maxdouble:.3e}; source float32 discrepancy={max32:.3e}, below its predeclared P2 arithmetic bound. It is not bitwise identity. `r5_source_cast_audit.csv` separately records float32 state quantization; some realized displacements are very small, so arithmetic size matters.

Only33 new patch rollouts were executed across forward and familiar controls; reverse0. The320 planned reverse rows retain baseline/eligibility/infeasibility, including all7 and21 originally viable reference cells. No matched reverse comparison is estimable. Familiar both-sign controls keep A/B future cues. Only one seed/reference/context/sign cell (seed12/A-reference, B future,+ sign) has a valid matched familiar test at k3; none at k5. This is insufficient to establish preserved familiar competence or a trade-off for the cohort. Full per-step CE/probabilities, worst-step losses and actual prefix+current+future full-sequence costs are saved separately.

## Primary figures

![Exact state by cue factorial](figures/fig_1_state_by_cue.png)

Figure1: allnine k3 cells, current/future separately; dots are20 trained seeds, and viable counts use the registered future criterion.

![Error trajectory](figures/fig_2_error_by_lag.png)

Figure2: cue-specific CE and correct-target margins, lag0 separate from the common Mixed future. Thin lines retain individual seeds.

![Natural gap geometry](figures/fig_3_gap_components.png)

Figure3: both fixed reference sides averaged within seed, raw gap/component norms and all singular spectra. Silent norm contains U/W components; it is not an additive CE decomposition.

![Selective repair and coverage](figures/fig_4_selective_repair.png)

Figure4: conditional U contrasts, full-reference bound and explicit feasibility denominators. Sparse effects are not made to look like full20 support.

![Reverse and familiar controls](figures/fig_5_reverse_controls.png)

Figure5: zero estimable reverse comparisons shown as feasibility counts, not zero behavioral effects. Both familiar signs remain explicit.

## Interpretation, uncertainties and stopping decision

The factorial causally localizes effects of one current cue under identical subsequent inputs and frozen recurrence, with substantial seed dependence. A useful immediate output does not ensure maintained prediction. Familiar inherited states are not a general solution to a common Mixed future. The present protocol does not isolate a unique disruption: it cannot distinguish distributed effects, an inadequate familiar-derived U, nonlinear sensitivity changes, or a feasible-intervention limitation. It does not establish absent sequence information, a global accessibility boundary, autonomous generation, long-run stability, a human neural mechanism or an attractor.

No training, optimizer update, cue optimization or hidden-state optimization occurred. Original model bytes remain unchanged. R4 is design only and recommends no new training. No threshold, rank, margin, sign, reference or budget was changed after unfavorable outcomes. All negative/blocked/inconclusive cases remain evidence.

## Tests, provenance and reproduction

RED behavioral tests preceded every stage; GREEN and REFACTOR logs are indexed with SHA256 in `tables/r5_test_evidence.csv`. Independent table checks recompute every R1 cell, all paired R3 contrasts and every registered bootstrap from seed rows. {regression_text} Read-only predecessor P2/P4/D1/S3 results are in `audit/predecessor_readonly_green.log`; the interrupted initial combined invocation is preserved separately in `audit/predecessor_regression.log`. Phase1K bounded diagnostics/final status are in `audit/phase1k_*`. Tests18/19 replay170 training updates and are deliberately excluded. Import/cache workarounds change no scientific code; any remaining blocker is explicit.

{lineage}

See `audit/frozen_inputs.json` for exact source/data hashes, `tables/r0_checkpoints.csv` for all20 checkpoints, `tables/r0_reuse_registry.csv` for360 provenance rows, `tables/r5_claim_sources.csv` and `CLAIM_LEDGER.md` for source-to-claim hashes. Stage manifests are immutable. Final seal rejects missing, changed or extra scientific bytes; only Finder .DS_Store metadata is explicitly excluded.

Read-only final verification: `PYTHONPATH=.:src python -m experiments.context_failure_selective_repair.verify --output results/context_failure_selective_repair/v1_1`.

Table-only regeneration to a new empty destination: `PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.context_failure_selective_repair.regenerate --source results/context_failure_selective_repair/v1_1 --destination /tmp/context_failure_regenerated`. It verifies the seal, makes no model calls, recreates five PNG/SVG figures, reports and derived tables, and compares every product hash.
'''
 report(dest,'FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md',final)
 handoff=f'''# Researcher handoff

{FAILURE}

## Actual algorithm and fixed source

Source `src/context_sequence/evaluation/rollout.py:Model.step`: float32 seven-input affine map (four token channels+three raw cue channels), recurrent8×8 map, tanh, four-logit linear next-token head. h_t=tanh(W_h h_(t−1)+W_x x_t+W_c c_t+b), z_t=W_o h_t+b_o. Original source biases are folded float32 as in all predecessors. The separate execution policy is neither substituted for this head nor updated. Teacher forcing uses exactly seven transitions, starts at zero, and never generates its own input tokens.

`experiments/context_failure_selective_repair/audit.py`: verifies source identities/cue definitions and replays76 fidelity cases. Original20 canonical High_monotonic seeds,170 updates, scale2; actual checkpoint paths and SHA256 in r0_checkpoints. `factorial.py`: consumes1440 saved P4 rows/360cells; no matched branch rerun. Both A/B-history references get one Mixed update and then the same Mixed future; zero best-reference selection. Current k and future k+1…7 scored separately via unchanged imported future_summary.

`geometry.py`: full-right SVD O=(I−11ᵀ/4)W_o; V rank3, N nullity5. If δ=Nα, Oδ=0, so softmax(z+W_oδ)=softmax(z), allfour classes. This does not force O diag(1−h_next²)W_hδ=0. Familiar-only G_A/G_B derivatives describe allfuture centered logits, not CE or labels. Stack, SVD G_fam N; fixed U2,W3. Identifiability uses absolute1e-8,relative1e-4σ1 and σ2−σ3>same bound. Exact double chain/autodiff, source finite differences eps1e-3/1e-4, one-sided at boundary. No fitting.

`repair.py`: d=r−h_M; natural h_M, full observed r, h+aVVᵀd, h+aNNᵀd, h+aUUᵀd, and h+a q_W/q_random. Controls have identical exact displacement norm, hash RNG identity and common a=.95×largest jointly feasible scale≤1. Interior1e-6, tolerance1e-10, zero norm1e-12. Infeasibility stays missing; no clipping. Current nullspace identity checks precede behavior; source float32 head has the imported P2 roundoff bound. Reverse r−aUUᵀd and matched controls use the fixed forward a and originally viable references, never a chosen harmful sign. Familiar controls use both signs and own A/B future cues.

`reporting.py`: independently recomputes summaries, paired within-reference differences, seed means and bootstrap10,000 draws/20260925. All20 seed rows retained; absent references are NA, never0. Complete20 support and favorable95% upper limits for U−W/U−random plus zero current violations are required for a selective claim. Reverse necessity is separate. Conditional one-seed degenerate bootstrap limits do not estimate population uncertainty.

## Results and limitations to teach back

Natural mean CE A={means['A']:.6f}, B={means['B']:.6f}, Mixed={means['Mixed']:.6f}. k3 inherited familiar histories give mean future difference+.262313[-.222091,.816107], not a clear improvement. Current Mixed-vs-own cue effects on future CE are+.749261 in A history and+.645418 in B history under the same subsequent Mixed cues. Seven of40 fixed references across5/20 seeds meet the inherited criterion; current correctness often deteriorates later.

U geometry identifiable40/40 does not imply a usable intervention. Strict joint geometry admits only1/40 primary cells; its U future CE change is+3.83e-6. Secondary k5 admits6/40 cells. No reverse matched cell is feasible. All complete current-softmax audits pass, but float32 roundoff and very small amplitudes limit interpretation. Local derivative FD discrepancies at saturated source states are recorded, not hidden. There is no selective-mechanism or necessary-computation result, and no new training is justified.

## Ninety-second teach-back

We asked whether Mixed breaks a specific recurring computation, rather than whether another full hidden state can work. First we reused the complete historical3×3 experiment: A, B or Mixed history followed by one A, B or Mixed cue, then identical Mixed inputs. Familiar history alone did not reliably repair the future; changing one current cue could affect later predictions. Next we divided the natural reference gap into directions visible to the current output and directions that leave allfour current probabilities unchanged. Familiar continuations, without correct-label losses, fixed two silent directions with large future-logit sensitivity. We tried that natural-gap component against equally large silent controls. Most natural states were too close to tanh boundaries for all the fixed control rays to stay in the required interior. Only one primary reference cell was admissible, with a tiny adverse effect; none of the reverse comparisons was admissible. So we localized state-and-cue effects but did not identify the selective mechanism. The right follow-up is to understand this intervention geometry under a separately fixed protocol, not train or search for a favorable rescue.

## Two adverse cases

1. Seed11/B-history reference at k3: future CE10.118783 versus natural Mixed2.777713. A familiar history can severely worsen the common Mixed future; familiar inheritance is not guaranteed competence. This adverse example is illustrative, never used for primary selection.
2. Seed12/A-history reference at k3: the only feasible matched U cell; a=.0001138595, exact displacement norm≈8.04e-5. U raises future CE≈3.83e-6 and is worse than both controls. The full reference itself has CE2.257857 versus natural Mixed.710387 and is not viable, while the tiny U state inherits native viability. A viability label alone therefore does not establish a successful repair. No population inference from one estimable seed.

## Validation

{regression_text}

## Reproduce and check

Final authoritative output is `results/context_failure_selective_repair/v1_1`. The initial v1 R2 is explicitly superseded; no predecessor bytes were overwritten. {lineage}

Read the five stage reports, append-only scratchpad, claim ledger and source tables. Run `python -m experiments.context_failure_selective_repair.verify --output results/context_failure_selective_repair/v1_1` with PYTHONPATH=.:src; regenerate with the command in the final report. New stage execution (to a distinct initialized directory) is `python -m experiments.context_failure_selective_repair.core --stage r0 ...`, then r1,r2,r3,r4,r5 in order. Never rerun already matched factorial branches. No R4 weights were updated. Preserve the existing failed/infeasible evidence if a separately preregistered protocol is proposed.
'''
 report(dest,'RESEARCHER_HANDOFF.md',handoff)
 files=['audit/r5_table_verification.json','tables/r5_summary.csv','tables/r5_primary_seed.csv','tables/r5_source_cast_audit.csv','tables/r5_coverage.csv','tables/r5_claim_sources.csv','tables/r5_test_evidence.csv','CLAIM_LEDGER.md','FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md','RESEARCHER_HANDOFF.md']
 if figures:files += ['figures/'+p for p in generate(src,dest/'figures')]
 return files


def run(out):
 files=products(out,out);dump(out/'audit/r5_products.json',{p:sha256(out/p) for p in files})
 return dict(scientific_model_calls=0,training_updates=0,primary_figures=5,products=len(files),selective_mechanism_identified=False,all_nulls_retained=True)
