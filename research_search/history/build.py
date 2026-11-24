"""Curated claim index with exact numerical paths, never an LLM-generated evidence score."""
from research_search.common import *

# Status describes the stated joint-phenotype hypothesis, not every result in
# the program. Budget-inconclusive screens remain distinct from failed tests.
STUDIES=[
 ('high_scale_initialization','partial','High initial scale restores Tanh Mixed impairment; Linear averaging alone does not establish joint splitting/transfer.','tables/cohort_summary.csv','recurrent_nonlinearity'),
 ('high_scale_novel_contexts','partial','Mixed interpolation does not guarantee off-segment Novel transfer; no model succeeds on every Novel cue.','tables/seed_values.csv','affine_extrapolation'),
 ('centered_context_linear','partial','Centering yields reliable zero-C prediction in its cohort, without demonstrating the full late separate-learning phenotype.','tables/frozen_evaluations.csv','context_centering'),
 ('nonzero_context_generalization','partial','Reliable zero-C prediction is not arbitrary nonzero-input robustness.','CLAIM_LEDGER.csv','nonzero_cue_robustness'),
 ('context_experience_prior','tested_joint_failed','Ordinary shared/context-bound priors improve C but no pretrained network passes the original late joint gate.','tables/','ordinary_context_prior'),
 ('meta_learning_scaling','partial','More ordinary Reptile experience does not by itself resolve joint splitting and transfer.','tables/','ordinary_reptile_dose'),
 ('structured_overlap_meta','partial','Structured overlap objectives retain useful partial signals, not established reliable joint success.','tables/','structured_overlap'),
 ('linear_scale_sweep','partial','Fresh scale4 strengthens balanced early costs; fully reliable bidirectional/headroom-independent splitting not established.','SCALE_SELECTION.md','nominal_scale'),
 ('high_scale_constrained_scaling_10k','tested_joint_failed','All final models acquire familiar tasks; high raw RMS and 10000 episodes do not reliably recover joint phenotype.','tables/','high_scale_constraint'),
 ('ab_update_transfer_analysis','partial','A/B updates usually help C early; later interference and optimizer history qualify transfer failures.','tables/gain_matrices.csv','optimizer_history'),
 ('joint_phenotype_hypothesis_screen','inconclusive','Short-pretraining B candidates lack competent controls; not evidence against their mechanisms. No pilot candidate promoted.','selection.json','cue_diversity'),
 ('a06_cached_readout_comparison','partial','Quarter readout initialization changes NoContext correctness beyond static bias-preserving scaling; C tradeoff remains.','comparison.csv','output_initialization'),
 ('a06_novel_cue_followup','partial','A06 improves unfamiliar-cue mean accuracy in all6; only24/72 useful cue tests and no all12 success. Late gate unassessed.','paired_summary.csv','output_initialization')]

def build():
    dest=PACKAGE/'history';inputs={};records=[];available=[]
    for name,status,claim,numeric,mechanism in STUDIES:
        base=ROOT/f'results/{name}/v1';manifest=base/'manifest.json'
        if not manifest.exists():
            records.append(dict(program=name,status='source_unavailable',hypothesis_status_not_inferred=status,claim=None,sources=[]));continue
        m=read(manifest);inputs[str(manifest.relative_to(ROOT))]=sha(manifest)
        reports=[p for p in [base/'REPORT.md',base/'FINAL_CONTEXT_EXPERIENCE_PRIOR_REPORT.md',base/'SCALE_SWEEP_REPORT.md',base/'FINAL_META_LEARNING_SCALING_REPORT.md',base/'RESEARCHER_HANDOFF.md'] if p.exists()]
        n=base/numeric
        # Select actual numerical artifacts from the sealed output index when
        # a historical program uses a different table name/directory layout.
        nums=[n] if n.is_file() else [base/p for p in m.get('outputs',{}) if p.endswith(('.csv','.csv.gz','.json')) and ('summary' in p or 'outcome' in p or 'joint' in p or 'paired' in p)][:6]
        if not nums:raise ValueError('No numerical support found for '+name)
        paths=[]
        for p in reports+nums:
            rel=str(p.relative_to(ROOT));h=sha(p)
            if str(p.relative_to(base)) in m.get('outputs',{}):assert h==m['outputs'][str(p.relative_to(base))],rel
            inputs[rel]=h;paths.append(rel)
        records.append(dict(program=name,status=status,mechanism=mechanism,claim=claim,sources=paths,numerical_sources=[str(p.relative_to(ROOT)) for p in nums],qualification='Read linked cohort/protocol; do not pool models or equate early flags with late functional evidence'))
    source=read(ROOT/'results/a06_novel_cue_followup/v1/registry/sources.json')
    for seed,s in source['controls'].items():
        p=ROOT/s['initializer'];inputs[s['initializer']]=sha(p);available.append(dict(label='ordinary-Reptile1920-scale5-U',seed=int(seed),path=s['initializer'],sha256=sha(p),uses='frozen baseline parity/history only until exclusion of new taskbank is verified',target_heldout_for_new_banks='not_assumed'))
        available.append(dict(label='A00-original-target-history',seed=int(seed),path=s['states'],sha256=source['inputs'][s['states']],updates='0..990; source first330 reused',new_task_reuse='not scientifically independent'))
    opened=[dict(id='prior-cue-coverage',research_area='experience',mechanism='cue_diversity',status='open_after_inconclusive_test',hypothesis='Broader cue-code coverage may reduce directional fragility if ordinary background learning is first competent.',alternatives=['cue-code coverage','optimization path/output gain'],source='results/a06_novel_cue_followup/v1/RESEARCHER_HANDOFF.md'),dict(id='direction-vs-curvature',research_area='learning',mechanism='optimizer_history',status='recommended_not_run',hypothesis='Cached update directional derivatives could distinguish unfavorable local directions from curvature.',source='results/ab_update_transfer_analysis/v1/RESEARCHER_HANDOFF.md'),dict(id='conditional-routing-ablation',research_area='architecture',mechanism='conditional_routing',status='unresolved',hypothesis='A small conditional routing mechanism requires a matched gate-neutralization ablation, not hidden-distance evidence.',source='results/joint_phenotype_hypothesis_screen/v1/summary.csv')]
    for name,obj in [('TESTED_HYPOTHESES.json',records),('OPEN_HYPOTHESES.json',opened),('AVAILABLE_CHECKPOINTS.json',available),('SOURCE_MANIFEST.json',dict(files=inputs,canonical_source='REPO_MAP.md; program-specific sealed reports; not a new experiment'))]:write(dest/name,obj,True)
    text='# Experiment history\n\nDefault researcher context. Status concerns the joint hypothesis and retains acquisition, floor and budget qualifications. Follow numerical links before interpreting any claim.\n\n'
    for r in records:text+=f"- **{r['program']} — {r['status']}**: {r.get('claim') or 'Source unavailable; no inferred result.'} "+' '.join(f'[{Path(p).name}](../../{p})' for p in r.get('sources',[]))+'\n'
    (dest/'EXPERIMENT_HISTORY.md').write_text(text)
    (dest/'KNOWN_FAILURES.md').write_text('# Known failure and uncertainty distinctions\n\n- Original selective-repair v1 R2 is superseded by v1_1; never reuse the initial geometry as current evidence.\n- Short B00/B02 background competence was inadequate: cue-diversity hypotheses remain inconclusive.\n- A06 canonical pilot seed1 improves C loss while losing one correct position; mean CE alone is insufficient.\n- A06 follow-up improves average transfer but leaves both C and unfamiliar-cue failures; late functional gate unassessed.\n- Near-zero familiar loss can make late update specificity unmeasurable. This is not evidence of shared representations.\n- Oracle states/output-compatible rescues do not establish natural retrieval.\n- Inherited strict precision failures remain separate from bounded float32 replay checks.\n\nEvery scientific claim above links through EXPERIMENT_HISTORY.md to sealed numerical data. New candidates need a distinct mechanism or an explicit discriminating reason to revisit an inconclusive branch.\n')
    return dict(studies=len(records),source_files=len(inputs),available_checkpoint_groups=len(available))
