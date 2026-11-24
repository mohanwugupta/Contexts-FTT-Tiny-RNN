from .common import *
def main():
 selection=json.loads((OUT/'audit/selection.json').read_text());screen=read('screen_outcomes');rep=read('replicate_outcomes');stats=read('replicate_paired_intervals');matched=read('replicate_matched_pairs');tables=[]
 for (a,s),g in screen.groupby(['arm','dose']):tables.append(dict(source=a,scale=s,n=len(g),acquired=int(g.strong_familiar.eq(1).sum()),early=int(g.early_phenotype.sum()),sustained=int(g.sustained.sum()),balanced=g.early_cost.mean(),A_to_B=g.A_to_B.mean(),B_to_A=g.B_to_A.mean(),later=g.sustained_cost.mean(),D_A=g.D_A.mean(),D_B=g.D_B.mean(),C_CE=g.ce.mean(),C_gap=g.gap.mean(),strict_C=int(g.strict_all.eq(1).sum()),Mixed_both_CE=g.Mixed_both_ce.mean(),Mixed_both_success=int(g.Mixed_both_success.eq(1).sum())))
 summary=pd.DataFrame(tables);write_frame(OUT/'tables/report_screen_summary.csv',summary);rr=rep.groupby('dose')[['early_cost','A_to_B','B_to_A','sustained_cost','D_A','D_B','strong_familiar','early_phenotype','initial_ce_A','initial_ce_B','initial_accuracy_A','initial_accuracy_B']].mean();write_frame(OUT/'tables/report_replication_summary.csv',rr.reset_index());primary=stats[stats.metric.isin(['early_cost','sustained_cost','A_to_B','B_to_A','D_A','D_B','strong_familiar'])];write_frame(OUT/'tables/report_replication_effects.csv',primary);acc=[]
 for stage in ['screen','replicate']:
  s=read(stage+'_switch_costs');q=s[s.boundary.le(60)].groupby(['seed','arm','dose','direction'])[['incoming_accuracy','outgoing_accuracy','accuracy_cost']].mean().reset_index();write_frame(OUT/f'tables/{stage}_early_accuracy_costs.csv',q)
 budget={stage:json.loads((OUT/f'audit/{stage}_complete.json').read_text()) for stage in ['screen','replicate']};steps=sum(b['target_updates']+b['clones'] for b in budget.values());report='''# Linear SRN scale sweep

**Scale4 independently increases balanced early transition costs and both first-block learning-specificity effects, but the full desired reliable bidirectional splitting regime is not established.** Scale4 learned familiar A/B in12/14 new seeds, versus14/14 at scale2. The average adjusted A→B cost increased strongly, while B→A remained slightly negative in the replication. All four matched-acquisition specificity intervals span zero. Larger CE spikes alone are not proof of useful separate skills.

The six-seed screen selected scale4 for both early and sustained ranking; it was frozen before14 new seeds. Replication includes scale2 as a separate necessary within-seed reference, not a second selected candidate. No new Reptile/FOMAML training occurred. U/P results remain exploratory and no scale satisfied the frozen persistence/transfer eligibility rule in either source.

## Protocol and source fidelity

Fresh uses the exact stored High initialization from the previous study: Gaussian SD2 reference with its existing initial nonrecurrent group floors and initial recurrence normalized once to spectral norm0.9. U/P use their own final FOMAML+1920 weights. All six original seed IDs and source tensors are verified against the complete predecessor seal. For every tested scale1,2,3,4,6,8,12,16, multiply the token matrix, all three cue columns and output matrix by α=s/2. No additional floor is applied, and recurrence/all biases stay unchanged. Scale2 is exact identity;18 archived histories are reused, not retrained or counted independently.

The native330-update A10/B10 sequence [0,1,2,0,3,1,2] predicts [1,2,0,3,1,2,3]. Each trial resets h0; fresh Adam0.005/policy and original CE+0.2 execution objective/joint gradient clip1 remain. No C/Mixed practice, target projection, recurrent renormalization, hidden clipping or learning-rate rescue. The C column is bitwise fixed throughout practice. All331 states, six cues, seven positions and four logits/probabilities are preserved. Stage1 uses all144 cells (126 new,18 reused). Replication uses28 new Fresh histories,14 each at scale2/4. Numerical failures and unsuccessful acquisition were not grounds to remove a cell; all runs were finite here.

## Six-seed screen, all scales

'''+summary.to_markdown(index=False,floatfmt='.6g')+'''

Large-scale Fresh6/8/12/16 acquisitions were4/6,2/6,1/6,1/6. Their enormous costs are not viable splitting candidates. Scale3/4 were eligible;4 maximized both early and sustained eligible rankings. `SCALE_SELECTION.md` preserves every source's eligibility and the immutable ranking. Its rules include≥5/6 familiar acquisition, bidirectional nonfloor first-block phenotype, sustained later cycles, larger balanced cost than scale2 and a positive matched comparison with at least two paired eligible seeds. Matched bins and signs are shown together; this exploratory rule is not proof of causal control for initial error.

Neither U nor P had a qualified splitting or joint splitting/transfer regime under the fixed reliability rule. Mean costs can increase at several scales, but persistence across at least five seeds fails; the highest raw scores also lose familiar acquisition. This does not mean scaling has no effects. Best eligible splitting/joint choices are explicitly absent, not replaced with a post hoc favorable score or a chosen seed.

## Independent Fresh replication

'''+rr.reset_index().to_markdown(index=False,floatfmt='.6g')+'''

Paired scale4−scale2 effects;14 complete seeds,10,000 whole-seed resamples, pointwise95% intervals:

'''+primary[['metric','n','mean','ci_low','ci_high','median','median_ci_low','median_ci_high','n_negative','n_positive']].to_markdown(index=False,floatfmt='.6g')+'''

Balanced early cost increases in all14 pairs (mean+3.81134 nats); sustained cycles2–3 also increase in all14 (mean+2.87047). Both first-A and first-B specificity increase in all pairs. However, familiar acquisition falls from14/14 to12/14, with all failures retained. Directional means are A→B2.649→10.473 nats and B→A+.0447→−.1562. A positive balanced mean can coexist with a negative directional cost. First-B own-minus-cross learning is not identical to the B→A incoming-versus-outgoing decrement; these must not be conflated.

Prespecified replication matched-acquisition results (all-start stratum), candidate−scale2 specificity:

'''+matched[matched.stratum.eq('all')][['performance_range','n','n_planned','mean','ci_low','ci_high','median']].to_markdown(index=False,floatfmt='.6g')+'''

No matched-bin interval excludes zero. Missing eligibility is retained (actual paired counts6–12 of14), not imputed as zero. All four bins and A/B strata are available. These conditional comparisons do not randomize starting performance or identify mediation; they also do not independently establish that amplified effects are unrelated to larger initial errors. Scale selection remains exploratory; replication validates only the tested Fresh effects, not U/P transfer.

## Original measures, accuracy and headroom

Every same-weight transition records incoming/outgoing CE, their raw difference, and subtraction of the original initial cue difference. A→B and B→A signs are preserved, including negative costs. Balanced cycles are the means of boundaries10/20,30/40,50/60. Persistence uses cycles2/3, headroom counts incoming CE>.05, and original both-context floor flags remain distinct from the registered own-context start floor. All33 boundaries including the censored end-of-run switch are preserved. Every seven-position contribution and incoming/outgoing accuracy is in `*_position_switch_costs.csv`; seed-averaged directional accuracy controls are in `*_early_accuracy_costs.csv`.

Own/cross improvements use frozen losses at identical start/end weights for all33 blocks. First-A is0→10 and first-B10→20. Original late170/330 gates remain unchanged and are not replaced with geometry or a new late threshold. Near-zero loss limits measurable late learning. Strong familiar means each A/B CE≤0.5 and accuracy≥6/7; acquisition times and failures stay visible. Strict C uses CE≤.1, perfect7/7 and gap≤.1; Mixed-both uses original CE≤.5 and accuracy≥6/7. Every absolute six-cue CE/accuracy and signed C gap is preserved; a gap is not absolute performance, nor is a smaller gap from worse familiar learning a successful transfer result.

## Mechanistic meaning of scale

s is nominal, not a guarantee of RMS=s. `*_initializations.csv` stores actual token/context/output RMS, recurrence and centered effective contrast for every condition. For a fixed state with h0=0, `K_t=W_o sum_(j=0)^(t−1)R^j` and `zA−zB=K_t(wA−wB)`. Since both W_o and the context columns scale by α, initial effective contrast scales by α². Fresh biases are zero, so all initial logits scale by α² as well: initial class accuracy is unchanged even though confidently wrong CE increases dramatically. For learned U/P, unscaled biases add linear/constant contributions to logits, so scaling is not simply a temperature change. Recurrence is unchanged by the intervention but remains learnable during original target adaptation.

These facts explain why larger RMS/contrast and CE spikes are not automatically stronger behavioral splitting. Input directions are preserved, not all output probabilities or learned calibration. Hidden separation, centered logits, K sensitivity and margins are explanatory diagnostics, not independent-skill evidence or causal mediators. Mixed-average(.5,.5,0) retains the linear logit midpoint/convex CE bound; Mixed-both(1,1,0) is different and lacks that bound. C is the learned third channel, unpracticed in the target task; no new cue vectors or hidden-state optimization were introduced.

## Figures and exact values

![Transition scale response](figures/screen_figure_1_scale_response.png)

![Replication response](figures/replicate_figure_1_scale_response.png)

![Fresh early curves](figures/screen_figure_2_Fresh_first20.png)

![Own/cross](figures/replicate_figure_3_Fresh.png)

![Persistence](figures/replicate_figure_4_Fresh.png)

![Post-meta transfer](figures/screen_figure_5_joint.png)

![Effective computation](figures/screen_figure_6_effective.png)

Each family has SVG and full/source/scale companions. Figure2 has full330/first100/first20 common y ranges and labeled symlog close-ups preserving small effects. Figures1/4 retain both directional signs, all seeds and source-separated intervals. Figure3 includes all matched bins. Figure5 x=C gap and y=balanced early cost identifies source, nominal scale, seed, familiar success(*) and Mixed-both(+); labeled symlog axes preserve signed values/zero. Figure6 shows actual RMS/effective contrast, not a geometry-based winner. Exact plotting coordinates are all saved CSVs; raw per-run probabilities/states remain source indexed.

## Integrity and limits

All data use the original Linear forward and target algorithms. Scale2 source/initial tensors match bitwise and native all-state audits verify four-class probabilities, loss/Adam/policy counters, C freezing and double/native affine identities. Source file hashes are immutable. Benchmark CPU1/2 native A-step copies used2 fixture updates separately. No meta-training or optimizer inheritance from meta checkpoints occurred.

The first selection JSON failed because NumPy bool was not serializable; partial JSON/log12 and RED13 are retained, native-bool-only correction14 completed the already-written scale4 decision before any replication outcomes. Rankings and thresholds did not change. The native pandas warning concerns nullable reporting dtype, not a scientific rule or numerical failure. All model runs completed without nonfinite rescue.

New scientific optimizer updates: '''+str(steps)+''' (screen41580 target+1512 restored-Adam copies; replication9240 target+336 copies).2 benchmark fixture updates separate.18 old histories/5940 target updates and216 clones are reused references, not new compute/subjects. Whole six screening seeds and fourteen replication seeds are never pooled; U/P remain six-seed exploratory. The evidence supports amplified balanced and practice-specific effects at scale4 with an acquisition trade-off, not a universally reliable bidirectional regime or demonstrated post-meta restoration preserving transfer. No favorable checkpoint, scale or seed was substituted after results.
'''
 (OUT/'SCALE_SWEEP_REPORT.md').write_text(report)
 (OUT/'RESEARCHER_HANDOFF.md').write_text('# Researcher handoff\n\nScale4 is the frozen exploratory Fresh choice. Fourteen new pairs show stronger balanced early cost (+3.811 nats) and later cycles (+2.870), and both first-block learning gains increase. But two scale4 models fail familiar acquisition, B→A average adjusted cost is slightly negative, and matched-acquisition intervals do not establish an effect beyond larger initial errors. Do not call this a proven reliable bidirectional splitting regime.\n\nU/P scaling raises some raw effects, but no scale meets the preregistered persistence/transfer reliability rule. Those six models were not independently replicated. See every scale and source, not only the best-looking trajectory.\n\nAlgorithm: copy stored Fresh/U/P prediction tensors; multiply only the input matrix (token+all3contexts) and output weights by s/2; leave recurrence/biases unchanged. Reset target Adam/policy, run330 original A10/B10 updates, evaluate six cues at all331 states. C is fixed during target. No meta-learning, projection, extra normalization or hidden rescue occurs. Eighteen scale2 histories are exact archived references. Fourteen new scale2 controls enable paired replication.\n\nCosts compare contexts at identical weights; learning specificity compares own and cross gains across identical block endpoints. A first-B learning advantage does not itself imply positive B→A switch cost. For Fresh, α² logit scaling leaves initial argmax accuracy unchanged while increasing CE, making headroom/accuracy/matched-bin controls essential. Geometry is not separate learning.\n\nRead SCALE_SWEEP_REPORT.md, SCALE_SELECTION.md and CLAIM_LEDGER.csv; use only hash verification/table-only plots for reproduction. All negative signs, acquisition failures, missing bins and corrected serialization logs remain. No more training or favorable-seed selection.\n')
 (OUT/'IMPLEMENTATION_AUDIT.md').write_text('# Implementation audit\n\nFull predecessor chain verified before runs. All6×3 stored directions available; scaling identity/directions/scope tested RED01/GREEN02. Source2 initial states equal archived starts bitwise;18 histories reused. Native target runner, original functional measures, criterion170/330, loss/Adam/policy/clipping remain source imports through process-local bindings. CPU configuration benchmarked for bitwise parity; native initializer bound to selected tested setting.\n\nRED06/GREEN07 tests balanced signed cycle arithmetic/persistence; RED08/GREEN09 fixed ranking controls. Serialization regression RED13/GREEN14 detects NumPy booleans; partial JSON retained, no result-dependent criterion change. All selected Fresh models are frozen before new seeds. No meta-training.\n\nAll172 logical histories (154new+18reused),331states, six cues and seven four-class positions remain. New optimizers/policies start empty, C moments remain zero and C columns fixed. Native source checks independently verify every valid frozen distribution, objective, initial model, model/policy/Adam counter, double affine identity and native arithmetic allowance. Position costs reconstruct original direction signs and initial-baseline subtraction. Event/position rows never become independent seeds.\n\nScientific updates52668, fixture updates2. Source/checkpoint hashes, exact scope, all seed grids and aliases are in registries/manifest. Complete failure/status fields remain; no nonfinite model failures occurred. Eligibility source C/Mixed fields never disqualify Fresh. Parent files/weights/ignore rules unchanged.\n')
 (OUT/'REPRODUCIBILITY.md').write_text('''# Reproduce without scientific training

From repository root:

```sh
python -m experiments.linear_scale_sweep.seal --verify
MPLCONFIGDIR=/tmp/linear_scale_sweep_mpl python -m experiments.linear_scale_sweep.figures --stage screen --output-dir /tmp/linear_scale_sweep_figures
MPLCONFIGDIR=/tmp/linear_scale_sweep_mpl python -m experiments.linear_scale_sweep.figures --stage replicate --output-dir /tmp/linear_scale_sweep_figures
python -m pytest tests/linear_scale_sweep --import-mode=importlib -q
```

Plots read CSVs only; fixed SVG hash salt/date metadata permit byte-identical PNG/SVG reproduction. Sealed outputs require an external destination. Do not rerun scientific runners, selection or benchmark. Their budgets are already spent. No meta-training code is used.

`registry/*_initializations.csv` indexes original/scaled states, actual RMS/contrasts and SHA256. `reused_scale2.csv` points to exact18 parent histories; `*_runs.csv` records every new/reused target and source hash. Complete per-history training/evaluation/geometry/position/decomposition/mathematical-check files sit beside331-state model/policy/Adam bundles. Compact summary/plot coordinates are `tables/*outcomes`, `summary`, `paired_seeds`, `paired_intervals`, `training`, `evaluations`, `blocks`, `switch_costs`, `position_switch_costs`, `matched`, and `geometry`.

The frozen selection JSON binds SCALE_SELECTION.md and screen outcomes. Initial Gaussian generator for new seeds is the same original High initialization/floor procedure; the sweep itself applies no floor. New14 scale2 baselines are paired controls, not a second selected candidate. Native source historical data are not independent replications. Every probability/logit and negative/missing outcome remains. New corrections require an explicit version and retained failures; no threshold tuning, numerical rescue, added recurrent normalization or favorable checkpoint selection. Post-seal receipts belong outside this package under scratchpads/. Publication policy remains PUBLICATION.md; no ignore rules changed.
''')
 claims=[dict(claim='Fresh4 increases balanced early/persistent costs',status='replicated directional effect across14 pairs',source='tables/replicate_paired_intervals.csv',limit='not sufficient to establish reliable positive costs in both directions'),dict(claim='Both-direction practice specificity',status='increases in14 pairs',source='tables/replicate_outcomes.csv',limit='larger initialCE; matched bins inconclusive'),dict(claim='Reliable acquisition at high scale',status='trade-off',source='tables/report_screen_summary.csv;tables/replicate_outcomes.csv',limit='scale4 12/14 versus14/14; larger scales frequently fail'),dict(claim='Post-meta restoration with C/Mixed transfer',status='no qualified regime under frozen rule',source='tables/scale_eligibility.csv;tables/screen_outcomes.csv',limit='U/P six seeds exploratory, not independently confirmed'),dict(claim='Nominal scale equals trained RMS',status='false; actual measured',source='registry/screen_initializations.csv;tables/screen_geometry.csv',limit='directions preserved at initialization only; target learning changes weights'),dict(claim='Norm/geometry identifies separate memories',status='not established',source='SCALE_SWEEP_REPORT.md',limit='descriptive; original criterion and floor limits retained')];write_frame(OUT/'CLAIM_LEDGER.csv',claims);dump(OUT/'audit/report_complete.json',dict(passed=True,scientific_updates=steps,meta_updates=0,screen_seeds=6,replication_seeds=14,selected=selection['selected'],all_negative_findings=True));append('## Researcher reports complete\nScale4 independently increases balanced costs/gains, butacquisition12/14 vs14/14, B→A nonpositiveonaverage, andmatchedCIs span0. NoqualifiedU/P regime. Allnumbers/limits recorded; no unsupported fullphenotype or mediation claim. Scientificwork stopped.')
if __name__=='__main__':main()
