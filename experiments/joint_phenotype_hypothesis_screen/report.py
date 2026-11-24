"""Read-only numerical interpretation; experiments can fail while tests pass."""
from .common import *
from . import analysis,engine

def confirmation_summary(s,selected):
 rows=[]
 for candidate in selected:
  a=s[s.candidate.eq(candidate)&s.seed.isin([2,3,4,5])];n=len(a);complete=n==4 and sorted(a.seed.tolist())==[2,3,4,5];familiar=int(a.familiar.sum());joint=int(a.joint.sum());strict=int(a.strict_joint.sum());rows.append(dict(candidate=candidate,n_observed=n,n_planned=4,familiar_passes=familiar,joint_passes=joint,strict_joint_passes=strict,reproducible_feasibility=bool(complete and familiar==4 and joint>=3),status='complete_confirmation' if complete else 'incomplete_confirmation'))
 return rows
def exposure():
 e=engine.Exposure();r=e.counts();reserved=e.physical_calls();return dict(**r,physical_optimizer_call_reservations=reserved,known_failed_before_optimizer_calls=1,physical_optimizer_calls=reserved-1,auxiliary_fixture_attempts=sum(x['run'].startswith('fixture_') for x in e.events),attempts_include_failed_fixtures=True,outer_interpolations=sum(len(pd.read_csv(p)) for p in (OUT/'runs').glob('*/meta_episodes.csv')),reused_B00_meta_inner=sum(4800 for p in (OUT/'runs').glob('B00_s*/complete.json')),reused_target_steps=sum(330 for p in (OUT/'runs').glob('*/complete.json') if json.loads(p.read_text()).get('reused_target')==330),reconstruction=0)
def main():
 assert json.loads((OUT/'audit/confirmation_complete.json').read_text())['passed'];before=sha(OUT/'registry/optimizer_events.jsonl');f=analysis.collect();s=f['summary'];selection=json.loads((OUT/'selection.json').read_text());ref=json.loads((OUT/'reference_thresholds.json').read_text());confirmation=confirmation_summary(s,selection['selected']);x=exposure()
 for name,data in [('metrics',f['metrics']),('summary',s),('candidate_registry',f['registry']),('background_summary',f['background'])]:csv(OUT/f'{name}.csv',data)
 csv(OUT/'practice.csv.gz',f['practice']);put(OUT/'confirmation_summary.json',confirmation);put(OUT/'registry/exposure.json',x)
 cohort=[]
 for (candidate,role),g in s[s.status.eq('completed')].groupby(['candidate','role']):
  r=dict(candidate=candidate,role=role,n=len(g),seeds=g.seed.tolist(),familiar_passes=int(g.familiar.sum()),early_passes=int(g.early.sum()),joint_passes=int(g.joint.sum()),strict_joint_passes=int(g.strict_joint.sum()))
  for k in ['Q_A','Q_B','E_normalized','D_A','D_B','C_ce','Mixed_ce','NoContext_ce']:r.update({k+'_mean':g[k].mean(),k+'_median':g[k].median(),k+'_n':int(g[k].notna().sum())})
  cohort.append(r)
 csv(OUT/'cohort_summary.csv',cohort)
 pilot=s[s.seed.isin([0,1])&s.candidate.isin(['A00','B00']+analysis.rules.ORDER)];valid=pilot[pilot.status.eq('completed')];overview=[]
 for candidate,g in valid.groupby('candidate',sort=False):overview.append(dict(candidate=candidate,n=len(g),familiar=int(g.familiar.sum()),early=int(g.early.sum()),joint=int(g.joint.sum()),strict=int(g.strict_joint.sum()),Q_A=g.Q_A.mean(),Q_B=g.Q_B.mean(),E_normalized=g.E_normalized.mean(),C_CE=g.C_ce.mean(),Mixed_CE=g.Mixed_ce.mean(),NoContext_CE=g.NoContext_ce.mean(),classification=', '.join(g.classification.astype(str))))
 table=pd.DataFrame(overview);csv(OUT/'report_pilot_overview.csv',table)
 bg=f['background'];bgpilot=bg[bg.seed.isin([0,1])];acquisition=valid[valid.candidate.isin(analysis.rules.ORDER)];joint_candidates=acquisition.groupby('candidate').joint.sum();lead=('Promoted '+', '.join(selection['selected'])+' under the fixed rules.' if selection['selected'] else 'No candidate qualified for confirmation under the fixed rules. No reserved-seed candidate runs were launched.')
 body=f'''# Joint A/B splitting and C-transfer hypothesis screen

{lead}

This is a small scale5/U procedure screen, not a population claim. Seeds0/1 supplied pilots; seeds2–5 were reserved from this new-arm selection. A00 is ordinary Reptile1920 before transfer objectives. B00 is ordinary Reptile240. Their exact archived models and target histories were reused. They are different controls. All final target practice remained A/B-only for330 updates. C, physical Mixed-average(.5,.5,0), and NoContext were frozen probes.

**Joint feasibility:** A01, A04 and A06 each pass in one pilot seed, while no candidate passes in both. The native A00 parent already passes in seed0, so a single favorable candidate result is not a demonstrated rescue. **Partial signal:** A06 reduces mean C CE from {valid[valid.candidate.eq('A00')].C_ce.mean():.6g} to {valid[valid.candidate.eq('A06')].C_ce.mean():.6g} nats and retains A/B acquisition in both seeds. Its seed1 still has C CE {valid[valid.candidate.eq('A06')&valid.seed.eq(1)].C_ce.iloc[0]:.6g}, {int(valid[valid.candidate.eq('A06')&valid.seed.eq(1)].C_correct.iloc[0])}/7 correct, and normalized switching {valid[valid.candidate.eq('A06')&valid.seed.eq(1)].E_normalized.iloc[0]:.6g}, below the fixed {ref['E_normalized']*.5:.6g} preservation threshold. The threshold is unchanged. **Tradeoffs:** SGD and moment resets can strengthen some switching/specificity measures while damaging C or familiar acquisition. **Inconclusive:** B00 is adaptation-viable but background competence is0/6 in each pilot seed. B05 improves C in both seeds and acquires A/B in2/2 versus1/2 for B00, but fails early preservation and useful C in its second seed. This is an acquisition-plus-transfer partial signal under an inadequate background control, not an isolated transfer effect or a joint solution.

## Fixed pilot findings

{table.to_markdown(index=False,floatfmt='.6g')}

Q means exclude undefined values only for display; exact eligible n and every seed are in summary.csv/cohort_summary.csv. Undefined Q remains nonlearning/floor-limited and cannot pass. An improved component is not joint feasibility. Same-seed gates are never pooled across seeds or conditions. The two pilot seeds do not support a population inference. NoContext is diagnostic, not a promotion requirement.

## What each hypothesis did

'''
 descriptions={'A01':'Cue-column freeze after the full native clipped Adam proposal','A02':'Readout-only prediction commits, native policy updates','A03':'Actual internal Adam LR0.0005, readout/policy0.005','A04':'Background-calibrated prediction SGD with native policy Adam','A05':'First-moment-only resets at real target switches','A06':'Initial output weights×.25; bias unchanged','B01':'sqrt7 onset cue, energy-matched to constant delivery','B02':'New Haar-orthogonal cue codes each background episode; original target vectors','B03':'Four missing-cue presentations on unrelated shared episodes only','B04':'80×60 ordinary Reptile, native20 schedule repeated3times, equal4800 inner exposure','B05':'H32 with independent parameter RNG and unrelated-bank output-gain match','B06':'Learned bounded token-drive gate, zero-initialized, ordinary loss only'}
 for candidate in analysis.rules.ORDER:
  g=pilot[pilot.candidate.eq(candidate)]
  if g.empty or not g.status.eq('completed').all():body+=f'- **{candidate}:** '+(', '.join(g.status.astype(str)) if len(g) else 'unavailable')+'. Conditional absence is not a failed model.\n';continue
  control='A00' if candidate.startswith('A') else 'B00';b=valid[valid.candidate.eq(control)];reasons=[]
  for row in g.itertuples():
   failed=[k for k,ok in [('A/B acquisition',row.familiar),('useful C',row.C_useful),('useful Mixed',row.Mixed_useful),('Fresh early preservation',row.early)] if not ok];reasons.append(f'seed{row.seed}: {row.classification}; '+('joint components pass' if not failed else 'fails '+', '.join(failed)))
  body+=f'- **{candidate} — {descriptions[candidate]}.** Matched {control}; mean C CE change {g.C_ce.mean()-b.C_ce.mean():+.6g}, normalized switching change {g.E_normalized.mean()-b.E_normalized.mean():+.6g}. '+ '; '.join(reasons)+'.\n'
 body+='''
A01–A03 test target plasticity allocation, not how the prior would develop under that rule. A04 calibration matches typical update magnitude, not Adam coordinates; A05 also changes post-switch step size. A06 must beat a frozen bias-aware readout re-expression before promotion. B01 matches input-cue energy, not hidden energy. B02 changes exact-cue familiarity as well as code diversity. B03 gives background experience with cue absence; final-target NoContext was never practiced. B04 changes depth, resets, task diversity and outer cadence together. Width controls output gain, not geometry or runtime. B06 is conditional computation, not two prescribed skills; its actual Mixed output is evaluated and the affine midpoint/Jensen identity is inapplicable.

## Exact seed-level pilot components

'''
 cols=['candidate','seed','own_A','cross_A','D_A','Q_A','own_B','cross_B','D_B','Q_B','E_normalized','C_ce','C_correct','Mixed_ce','NoContext_ce','familiar','joint'];body+=valid[cols].to_markdown(index=False,floatfmt='.6g')
 body+='\n\nSigned own/cross gains, starting losses, floor flags, raw E, directional/sustained costs and normalized effects are preserved in summary.csv. A positive D can include harm to the other familiar context; Q caps negative cross gains at zero and the positive familiar-net-gain gate prevents interference-dominated joint passes. Late familiar floors do not establish a transition to shared representation. The original early phenotype is separately reported; the original late counterfactual functional criterion is unassessed for new arms, because this protocol does not authorize those extra branches.\n'
 body+='\n## Background competence and budget qualification\n\n'+bgpilot[['candidate','seed','fresh','adaptation_viable','background_competent','tasks_useful','mean_gain','mean_shared_gain','mean_conflict_gain']].to_markdown(index=False,floatfmt='.6g')
 body+='\n\nValidation uses six sealed unrelated tasks per seed, three shared and three conflicting across AB/AC/BC. Each task requires both own cue roles for competence. Adaptation viability requires finite results, >1e-4 mean own gain on≥4/6 tasks, and positive shared/conflicting mean gains. Competence requires useful performance in≥4/6 including≥2 of each kind. Only B00 has a paired variant-matched Fresh-to-meta check. Other arms have post-meta comparisons with B00, not claimed untrained-to-trained benefits. A negative B comparison with an inadequate background control is budget_inconclusive; no dose was extended. Acquisition-plus-transfer is flagged when control acquisition differs.\n'
 body+='\n## Selection and confirmation\n\nFixed Fresh medians: '+json.dumps({k:ref[k] for k in ['Q_A','Q_B','E_normalized','resolved']})+'. Preservation uses50% of each positive reference median, both eligible positive Q, positive E and positive own+cross net gains. These are routing thresholds, not redefined biological effects.\n\nBoth pilot seeds must be joint-feasible, plus the registered transfer, splitting or joint-rescue route against the proper control. A06 also must beat its frozen scaling control. Ranking is strict-joint count, worst minimum Fresh retention, mean C CE, registry order. Complete input hashes, every route result and reasons are in selection.json.\n\n'
 body+=(pd.DataFrame(confirmation).to_markdown(index=False) if confirmation else 'No candidate advanced. There are no new reserved-seed outcomes; this is not four failed confirmations. Figure2 shows only the first two fixed-ranked partial signals, explicitly descriptive.')
 body+=f'''

## Algorithm and teach-back

h_t=R h_(t−1)+W_x x_t+W_c c_t+b_ih+b_hh; z_t=W_o h_t+b_o. The container's advertised tanh is not executed. Hidden state resets to zero for each seven-token sequence. Native prediction CE+.2 execution loss retains detached confidence/uncertainty, a fresh target policy/Adam.005, native parameter ordering, and joint norm1 clipping before all plasticity masks. Frozen entries are restored bitwise after an Adam proposal; shadow moments may evolve. A03 uses parameter-group learning rates rather than scaled gradients. SGD bypasses only the native Adam-moment assertion, with independent target/C guards; it does not invent Adam state.

Ordinary P2 Reptile copies prediction weights, creates fresh inner Adam/policy, takes four5-update support blocks, then interpolates prediction weights by epsilon0.1→0.01. No policy is meta-learned. B04 repeats those20 presentations3times without inner resets and uses80 outer moves. The source task and target-orbit exclusions remain. Task, parameter and augmentation RNG streams are separated and frozen before outcomes.

G_own=L_own(start)−L_own(end), G_cross likewise; D=G_own−G_cross. Q=1−max(G_cross,0)/G_own only when starting CE>.05 and G_own>max(1e-4,.05*startingCE). At10,20,…,60, adjusted switch=incomingCE−outgoingCE−the same difference at0. E is its mean; H0=max(.5,initial familiar meanCE), normalized E=E/H0. The unchanged source directional/sustained function cross-checks the explicit implementation. No normalized gain ratio is used at a learning floor.

Useful performance is CE≤.5 and≥6/7 correct, tested with integer counts. Familiar requires both A and B. Joint requires familiar, useful C, useful actual Mixed-average and early preservation in one seed. Strict joint additionally requires C CE≤.1 and7/7; strict Mixed and the inherited CE/gap rule remain separate. NoContext is never an added gate.

## Timing, reproducibility and limitations

Actual exposure: {json.dumps(x)}. Caps are124800 meta-inner,15180 target,5000 auxiliary attempts. Physical optimizer calls differ from sequence-gradient updates for SGD+policy Adam. One early gate-fixture attempt failed before any optimizer call; a separate SGD fixture failed an Adam-only assertion after its two real optimizer calls. All attempts remain charged, all failing logs retained; no source threshold/tolerance was relaxed. First-run timings and projections are in audit/14_checkpoint_pilot.log and audit/16_pretraining_pilot.log, with width/gate costs separately visible. Exact B00 reuse saves ordinary meta-inner/target updates, not an independent replication. Source A00/F00 histories are historical references, not new scientific updates.

Native logits are float32; frozen scoring reuses core.score in float64. All four probabilities, seven positions, margins/output bias and named full states are retained. Ungated Mixed checks retain strict double algebra and the inherited64eps float32 propagation bound separately; B06 is explicitly inapplicable. Three inherited strict query-forward failures remain source exceptions, not passing strict references. CPU/thread1 and native numerical dependencies were preserved. No new package, GPU/MPS, projection, scale, reward, cue search or meta-objective was introduced.

This screen compares complete procedures, not isolated human memory stores. Pilot selection is exploratory; confirmation seeds were reserved only from this intervention-selection process. A negative short screen does not prove linear networks cannot express the joint phenotype. No post-hoc parameter tuning, target-outcome calibration or selective extension to990 occurred. Reproduction reads saved tables/arrays; it never retrains. See SOURCE_AUDIT.md, REPRODUCIBILITY.md, candidate_registry.csv, selection.json, reference_thresholds.json, exact CSVs and the final manifest.
'''
 (OUT/'REPORT.md').write_text(body)
 handoff=f'''# Researcher handoff

{lead}

Question: can ordinary sequence learning retain the substantial early A/B acquisition phenotype and useful unpracticed C and physical Mixed transfer in the same model? We tested twelve fixed procedures at scale5/U using two pilot seeds. A00 ordinary1920 and B00 ordinary240 are separate, exact archived controls. No target-C/NoContext/Mixed practice or reward was added.

Read report_pilot_overview.csv for procedure means, summary.csv for signed per-seed components/floors, metrics.csv and runs/*/observations.npz for every checkpoint/position/class. selection.json is the immutable decision. Confirmation results: {json.dumps(confirmation)}. NoContext remains a diagnostic. Tests passing validates implementation, not the scientific phenotype.

Teach-back: Q measures the eligible own improvement that did not also help the other familiar cue, without rewarding cross-context damage. Raw D/own/cross remain. Normalized switching retains same-state incoming/outgoing loss differences relative to the initial difference. These functional measures can dissociate from representation distances. C competence in a different seed cannot rescue absent specificity. A fixed output transformation is controlled separately from learning. The gate changes cue-token linearity, so Mixed must actually be run.

Next action: {'inspect the exact failed confirmation gate before considering a separately registered replication; do not retune the confirmed configuration' if selection['selected'] else 'audit the failed gate of the first descriptive partial signal '+str(selection['partial_display'][:1])+' using its cached position errors, signed gains and matched-control history before proposing any further training'}. This is a recommendation, not a launched follow-up. Do not extend the ordinary budget or choose a favorable seed/scale after this result.

Exposure {json.dumps(x)}. Source results/checkpoints remain unchanged. All scientific work stops at the registered scope; reproduce with table-only figures and hashes. Inherited strict numerical exceptions and all technical/floor/background-competence qualifications remain in REPORT.md and SOURCE_AUDIT.md.
''';(OUT/'RESEARCHER_HANDOFF.md').write_text(handoff)
 assert sha(OUT/'registry/optimizer_events.jsonl')==before
 from .figures import main as figures
 figures();receipt=dict(passed=True,read_only_optimizer_ledger_sha=before,summary_rows=len(s),metric_rows=len(f['metrics']),confirmation=confirmation,report_sha=sha(OUT/'REPORT.md'));old=OUT/'audit/report_complete.json';name='report_complete_v2.json' if old.exists() and json.loads(old.read_text())!=receipt else 'report_complete.json';put(OUT/'audit'/name,receipt);return f
