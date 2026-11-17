from pathlib import Path
import json
import pandas as pd

QUESTIONS={
'A1':'Does contextual differentiation accompany reduced natural generalization?',
'A2':'Does the relation remain at matched familiar acquisition?',
'A3':'Is a familiar-derived shared state causally sufficient for changed-context behavior?',
'A4':'How does continuing context affect maintenance after one shared-state placement?',
'A5':'Can physical cues generate the useful state from natural prestates?',
'A6':'Do contrast routing and closure improve natural transfer while retaining differentiation?',
'A7':'Do paired changes in shared-state geometry and behavior agree?',
'A8':'Which raw contextual distinctions survive the one-dimensional contrast projection?'}


def md(frame):
    f=frame.copy()
    def val(x):return f'{x:.5g}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(val(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))


def write_report(out,tables,status,config):
    lines=['# Final four-experiment paper freeze','',
    '## Four questions and empirical answers','']
    if 'a1_canonical_phenotype_seed' in tables:
        a=tables['a1_canonical_phenotype_seed'];m=a.groupby(['condition','probe']).raw_ce.mean().unstack()
        lines+=['**1. Does differentiation accompany a generalization cost?** The final canonical natural CE means are below. High/Low differ in familiar acquisition as well as differentiation; this is association, not a causal manipulation of splitting.',md(m[['A','B','Mixed','NoContext','Novel30Mean','Novel60Mean']].reset_index()),'']
        a3=tables['a3_composite_sufficiency_seed'];v=a3[a3.family.eq('High_monotonic')&a3.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])&a3.intervention.isin(['natural','repeated_composite','single_composite'])]
        lines+=['**2. Does a useful shared solution remain available?** Frozen A/B-derived state placement produced the following effects in High networks. Positive rescue means lower CE. Acute controls and endpoint controls remain in the tables; state sufficiency does not imply natural construction or necessity.',md(v.groupby(['probe','intervention'])[['raw_ce','raw_rescue','immediate_rescue','remaining_rescue']].mean().reset_index()),'']
        v=tables['a4_maintenance_seed'];v=v[v.family.eq('High_monotonic')&v.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
        lines+=['**3. What limits access and maintenance?** After single placement, continuation-specific remaining rescue and post-injection task distance are:',md(v.groupby(['probe','continuation'])[['remaining_rescue','post_task_distance']].mean().reset_index()),'']
        v=tables['a5_cue_generation_seed'];v=v[v.family.eq('High_monotonic')&v.analysis_kind.eq('local')&v.intervention.eq('Mixed')&v.probe.isin(['A','B','NoContext','Novel30Mean','Novel60Mean'])]
        lines+=['The physical Mixed cue has these mean local task-generation effects (positive means closer to the midpoint). Partial movement and behavioral benefit are distinct from reliably recreating the exact state.',md(v.groupby('probe')[['task_generation','raw_rescue']].mean().reset_index()),'']
    if 'a6_constructive_seed' in tables:
        a=tables['a6_constructive_seed'];v=a[a.probe.isin(['A','B','Mixed','NoContext','Novel30Mean','Novel60Mean'])]
        lines+=['**4. Can the mechanisms construct natural generalization?** The existing sequence-only C0–C3 models yield:',md(v.groupby(['condition','probe']).raw_ce.mean().unstack().reset_index()),'']
        v=a[a.probe.eq('A')];lines += ['Familiar differentiation remains a separate outcome:',md(v.groupby('condition')[['hidden_separation','task_separation']].mean().reset_index()),'']
        b=tables['paper_bootstrap_summary'];v=b[b.analysis.eq('A6')&b.contrast.eq('C3_full minus C0_canonical')&b.metric.eq('raw_ce')&b.probe.isin(['A','B','Mixed','NoContext','Novel30Mean','Novel60Mean'])]
        lines+=['Primary paired C3−C0 CE effects (negative improves behavior):',md(v[['probe','mean','ci_low','ci_high','n_seeds']]),'',
                  'The contrast pathway aliases same-side Novel30 and Novel60 and also aliases Mixed with NoContext. Both raw novelty distance and discarded components are preserved in the audit. These equivalences are representational consequences, not post-result changes to cues.']
    lines+=['','## Analysis-by-analysis accountability','',md(status),'']
    source_names={'A1':'a1_canonical_phenotype_seed','A2':'a2_acquisition_robustness_seed','A3':'a3_composite_sufficiency_seed','A4':'a4_maintenance_seed','A5':'a5_cue_generation_seed','A6':'a6_constructive_seed','A7':'a7_geometry_behavior','A8':'a8_effective_cue_audit'}
    contrasts={
'A1':'High−Low; Novel30/60−familiar; Novel60−Novel30',
'A2':'High update170 versus familiar-CE-matched Low within the same seed',
'A3':'Composite versus no patch, wrong position, orthogonal and endpoints; repeated and acute rescue',
'A4':'D1−D0, D2−D0; native Novel60−Novel30, with lag support retained',
'A5':'Local Mixed versus native/A/B/zero; persistent Mixed versus pulse/A/B takeover',
'A6':'C1−C0, C2−C0, C3−C0, C3−C1, C3−C2',
'A7':'Within-seed C3−C0 absolute task distance versus CE, normalized sensitivity',
'A8':'Raw cue geometry versus computed alpha/discarded residual and effective drive'}
    controls={
'A1':'Low family and natural familiar A/B endpoints', 'A2':'Familiar-only 0.05-nat per-endpoint caliper; incoming-context match; historical architecture separate',
'A3':'Wrong-position, norm-matched orthogonal, A/B endpoint, no patch, all seven acute positions',
'A4':'Native/zero/Mixed continuation; Mixed D0=D2 and NoContext D0=D1 identities',
'A5':'Bit-identical prestates; native/A/B/zero branches; pulse and persistent endpoint takeover',
'A6':'Full C0–C3 component design; no hidden patches; C0 identity with sequence-only sources',
'A7':'Absolute distances and normalization denominators; no fitted correlation required',
'A8':'Label-blind A/B projection and identical-effective-drive checks'}
    limits={
'A1':'Familiar performance differs; no causal inference about splitting.',
'A2':'Only one of 20 final-High seeds meets the frozen caliper; its matched Low update110 weights are not saved. No matched new-Novel outcome is reconstructed or inferred.',
'A3':'State replacement establishes sufficiency, not natural access, necessity, or unique computation.',
'A4':'Seven teacher-forced transitions and varying lag support do not establish a long-run attractor.',
'A5':'Cue-driven behavioral rescue and exact midpoint generation can dissociate.',
'A6':'Sequence-only training differs from canonical monotonic training; closure uses A/B-derived physical Mixed cue without probe behavioral targets.',
'A7':'Descriptive paired dissociation; normalized residual can change through its denominator.',
'A8':'Operational out-of-support novelty is not a new semantic dimension; no rho pathway tested.'}
    for a in QUESTIONS:
        st=status[status.analysis.eq(a)].iloc[0]
        lines += [f'### {a} — {QUESTIONS[a]}','',
            f'- Cohort: {"constructive sequence-only C0–C3" if a in ["A6","A7","A8"] else "canonical High/Low monotonic"}; registered seeds 0–19; primary final checkpoint 170.',
            f'- Intervention/conditions and controls: {controls[a]}.',
            f'- Implementation status: `{st.implementation_status}`; evidence status: `{st.evidence_status}`; seed count: {st.n_seeds}; exclusions: {st.exclusions}.',
            f'- Primary metric: {"computed alpha, residual and context drive" if a=="A8" else "raw CE in nats; absolute centered-logit midpoint distance and normalized residual reported separately"}.',
            f'- Primary contrast: {contrasts[a]}.',
            '- Uncertainty: paired whole-seed percentile bootstrap (10,000 draws, seed 20260925) where estimable; actual matrix saved. A2 has no inferential Novel result; A7 is a paired scatter; A8 identities are deterministic.',
            '- Acquisition/familiar check: genuine canonical pre-update logs and natural A/B CE are retained. Constructive conditions have historical pre-update logs and frozen A/B CE; no retraining occurred.',
            '- Differentiation check: transition-matched hidden/task A/B separation is retained in the relevant natural and intervention tables.',
            f'- Result artifact: `tables/{source_names[a]}.csv`; paired estimates and intervals: `paper_primary_contrasts.csv` / `paper_bootstrap_summary.csv`.',
            f'- Interpretation and limitation: {limits[a]}',
            '- Next manuscript decision: report this prespecified result, including limitations and negative effects; do not tune or extend modeling.','']
    lines+=['## Prospective and numerical provenance','',
            f'Novel registry SHA256: `{config["novelty_registry_sha256"]}`. Contract SHA256: `{config["contract_sha256"]}`.',
            f'Evaluator: `{config["evaluator"]}`; {config["numerical_dtype"]}. Historical NumPy/Torch discrepancies and prospective tolerances are saved in `audit/evaluator_regression.csv`.',
            'The registry was committed before new Novel behavior was evaluated. Matching was saved before new-cue execution. Every retained source checkpoint hash was checked before and after; optimizer creation/steps and backward were trapped. Historical tables remain immutable. This final analysis freeze does not retroactively preregister the earlier exploratory modeling program.',
            '', '## Historical crosswalk','',
            '| Paper experiment | Historical implementation |','| --- | --- |',
            '| 1 | Canonical confirmatory / timeline; historical ordinal matching separately labeled |',
            '| 2 | Phase 1I |','| 3 | Phase 1J + 1K |','| 4 | Phase 1L |','',
            'LegacyNovel sensitivities, historical matching, neutral closure, individual cue directions, Low controls, endpoints, injection positions and lag summaries are exported. Human Results remain a separate author-selected analysis; this program did not refit human data.',
            '', '## Stop decision','',
            'The prespecified available frozen-model analyses are complete. Acquisition matching has insufficient common support and an unavailable matched intermediate state; it is reported as such. A positive, selective, or negative result does not trigger Phase 1M, new cues, retraining, or an additional context pathway. Stop modeling and write the manuscript.']
    (out/'FINAL_FOUR_EXPERIMENT_FREEZE_REPORT.md').write_text('\n'.join(lines)+'\n')
