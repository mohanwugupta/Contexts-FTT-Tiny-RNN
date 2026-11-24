"""Trusted table-only worker analysis. No model creation or forward/update."""
import numpy as np,pandas as pd
from research_search.common import ROOT,read,write,sha,clean
from research_search.evaluation.scoring import PROBES
from experiments.joint_phenotype_hypothesis_screen.analysis import seed_summary
from experiments.meta_objective_balance.statistics import interval
from .evaluation import four_outcome
from .protocol import SOURCE,OUT
TIMES=[0,10,20,40,80,120,170,330]


def validate_grid(frame):
    if set(frame.cue)!=set(PROBES) or len(frame)!=331*5 or frame.duplicated(['update','cue']).any():raise ValueError('Duplicate/missing/wrong cue cells')
    if set(frame['update'])!=set(range(331)):raise ValueError('Wrong saved-state indexing')
    if not np.isfinite(frame[['ce','correct_count','margin']].to_numpy(float)).all():raise ValueError('Missing/nonfinite retained as blocked, not imputed')
    if not frame.correct_count.between(0,7).all() or not (frame.correct_count%1==0).all():raise ValueError('Invalid correct counts')


def block_effects(frame):
    validate_grid(frame);p=frame.set_index(['update','cue']);rows=[]
    for block,start in enumerate(range(0,330,10),1):
        practiced='A' if block%2 else 'B'
        other='B' if practiced=='A' else 'A'
        own=p.loc[(start,practiced)].ce-p.loc[(start+10,practiced)].ce
        cross=p.loc[(start,other)].ce-p.loc[(start+10,other)].ce
        for cue in PROBES:
            a,b=p.loc[(start,cue)],p.loc[(start+10,cue)]
            rows.append(dict(own_gain=float(own),cross_gain=float(cross),specificity=float(own-cross),net_familiar_gain=float(own+cross),own_start_floor=bool(p.loc[(start,practiced)].ce<=.05),block=block,start=start,end=start+10,practiced=practiced,cue=cue,start_ce=float(a.ce),end_ce=float(b.ce),gain=float(a.ce-b.ce),start_correct=int(a.correct_count),end_correct=int(b.correct_count),correct_gain=int(b.correct_count-a.correct_count),start_floor=bool(a.ce<=.05),start_margin=float(a.margin),end_margin=float(b.margin)))
    return pd.DataFrame(rows)


def execute(campaign,analysis,dest):
    dest.mkdir(parents=True,exist_ok=True)
    definition=read(campaign/'definition.json')
    for rel,digest in definition['sources'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Changed frozen input: '+rel)
    rows=[];metrics=[];selected=[];positions=[];switches=[];matched=[]
    ids=read(SOURCE/'paired_jobs.json')
    for condition,jobs in ids.items():
        if len(jobs)!=8 or len(set(jobs))!=8:raise ValueError('Incomplete task/seed grid')
        for job in jobs:
            root=SOURCE/'jobs'/job;old=read(root/'evaluation.json');f=pd.read_csv(root/'metrics.csv');validate_grid(f)
            native=seed_summary(f,old['candidate'],old['seed'],'development',dict(resolved=False,Q_A=0,Q_B=0,E_normalized=0))
            for k,v in native.items():
                if isinstance(v,(bool,str)) and old.get(k)!=v:raise ValueError('Historical flag changed: '+k)
                if isinstance(v,(int,float)) and np.isfinite(v) and not np.isclose(v,old[k],rtol=1e-10,atol=1e-10):raise ValueError('Historical scalar changed: '+k)
            end=f[f['update']==330].set_index('cue').to_dict('index')
            new=four_outcome(old,end)
            meta=dict(condition=condition,task_id=old['task_id'],seed=old['seed'],job_id=job)
            rows.append(old|new|meta)
            metrics.append(f.assign(**meta))
            for name,collection in [('switch_costs.csv',switches),('matched_acquisition.csv',matched)]:collection.append(pd.read_csv(root/name).assign(**meta))
            p=pd.read_csv(root/'positions.csv')
            positions.append(p[p['update'].eq(330)].assign(**meta))
            if analysis=='acquisition_vs_interference':selected.append(block_effects(f).assign(**meta))
            elif analysis=='confidence_vs_correctness':
                sf=pd.read_csv(root/'static_readout_metrics.csv');validate_grid(sf)
                a=f[f['update'].isin(TIMES)].merge(sf[sf['update'].isin(TIMES)],on=['update','cue'],suffixes=('_natural','_static'),validate='one_to_one')
                a['ce_gain_static']=a.ce_natural-a.ce_static;a['correct_gain_static']=a.correct_count_static-a.correct_count_natural
                selected.append(a.assign(**meta))
                sp=pd.read_csv(root/'static_readout_positions.csv')
                q=p[p['update'].isin(TIMES)].merge(sp[sp['update'].isin(TIMES)],on=['update','cue','position'],suffixes=('_natural','_static'),validate='one_to_one')
                q['ce_gain_static']=q.ce_natural-q.ce_static
                positions.append(q.assign(**meta,comparison='natural_vs_static'))
            else:raise ValueError('Unregistered analysis')
    result=pd.DataFrame(rows)
    if len(result)!=16 or result.duplicated(['condition','task_id','seed']).any():raise ValueError('Invalid network grid')
    if sorted(result.seed.unique())!=[21000,21001] or result.groupby(['condition','seed']).size().ne(4).any():raise ValueError('Missing seed/task')
    for name,frame in [('seed_task_metrics',result),('frozen_metrics',pd.concat(metrics)),('selected_analysis',pd.concat(selected)),('position_values',pd.concat(positions)),('switch_costs',pd.concat(switches)),('matched_acquisition',pd.concat(matched))]:frame.to_csv(dest/(name+'.csv'),index=False)
    fields=['familiar_ce','A_ce','B_ce','C_ce','Mixed_ce','NoContext_ce','A_correct','B_correct','C_correct','Mixed_correct','NoContext_correct','own_A','cross_A','D_A','own_B','cross_B','D_B','net_A','net_B','A_to_B','B_to_A','E','sustained_cost']
    means=result.groupby('condition')[fields].mean();means.to_csv(dest/'condition_means.csv')
    counts=result.groupby('condition')[['familiar','original_early_phenotype','C_useful','Mixed_useful','NoContext_useful','four_outcome_feasible_v1']].sum();counts.to_csv(dest/'outcome_counts.csv')
    ci=[]
    for field in fields:
        a=result[result.condition.eq('candidate')].groupby('seed')[field].mean().reindex([21000,21001])
        b=result[result.condition.eq('control')].groupby('seed')[field].mean().reindex([21000,21001])
        ci.append(dict(metric=field,**interval((a-b).to_numpy(float),[21000,21001])))
    pd.DataFrame(ci).to_csv(dest/'paired_seed_intervals.csv',index=False)
    # Compact evidence packet includes all network endpoint outcomes, plus
    # signed seed/task effects in exact tables; no favorable examples selected.
    findings=[]
    s=pd.concat(selected)
    if analysis=='acquisition_vs_interference':
        for (condition,cue),g in s.groupby(['condition','cue']):
            findings.append(dict(condition=condition,cue=cue,positive_CE_gain_blocks=int(g.gain.gt(0).sum()),negative_CE_gain_blocks=int(g.gain.lt(0).sum()),floor_start_blocks=int(g.start_floor.sum()),net_correct_changes=int(g.correct_gain.sum()),total_blocks=len(g)))
    else:
        for (condition,cue),g in s[s['update'].eq(330)].groupby(['condition','cue']):findings.append(dict(condition=condition,cue=cue,mean_CE_gain_static=float(g.ce_gain_static.mean()),total_correct_gain_static=int(g.correct_gain_static.sum()),cells=len(g)))
    packet=dict(analysis=analysis,complete_grid=True,four_outcome_all_grid=bool(result.four_outcome_feasible_v1.notna().all() and result.four_outcome_feasible_v1.all()),source_histories=16,independent_initialization_seeds=2,repeated_tasks_per_seed=4,endpoint=330,means=means.reset_index().round(9).to_dict('records'),outcome_counts=counts.reset_index().to_dict('records'),selected_findings=findings,screen_reference='unresolved; screen_preservation_valid=null for all16; original flags retained',new_model_forwards=0,new_optimizer_steps=0,scientific_status='cached descriptive; trained-prior mechanism and persistent late functional splitting inconclusive',claims='No causal mediation, no natural retrieval or human mechanism established. Candidate/control are archived zero-prior conditions, not newly trained competent priors.')
    write(dest/'evaluation_result.json',packet,True)
    write(dest/'complete.json',dict(passed=True,analysis=analysis,new_optimizer_steps=0,new_model_forwards=0,files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}),True)
    return packet
