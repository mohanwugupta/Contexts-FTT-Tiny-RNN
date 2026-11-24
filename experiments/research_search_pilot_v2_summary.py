"""Read-only endpoint summary of the agent's predeclared pilot prediction."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from research_search.common import read,write

SEEDS=[21000,21001]


def hypothesis(frame):
    f=frame.copy();keys=['seed','task_id']
    if set(f.condition)!=set(['candidate','control']) or sorted(f.seed.unique())!=SEEDS:
        raise ValueError('Incomplete registered conditions/seeds')
    if f.duplicated(['condition']+keys).any():raise ValueError('Duplicate task-seed cell')
    task_ids=sorted(f.task_id.unique())
    if len(task_ids)!=4 or len(f)!=16:raise ValueError('All16 planned rows required')
    for condition in ['candidate','control']:
        for seed in SEEDS:
            if sorted(f.loc[f.condition.eq(condition)&f.seed.eq(seed),'task_id'])!=task_ids:raise ValueError('Unbalanced task grid')
    if not np.isfinite(f[['C_ce','C_correct']].to_numpy(float)).all():raise ValueError('Missing/nonfinite outcomes are incomplete')
    a=f[f.condition.eq('candidate')].set_index(keys);b=f[f.condition.eq('control')].set_index(keys)
    pairs=a[['C_ce','C_correct']].join(b[['C_ce','C_correct']],lsuffix='_candidate',rsuffix='_control',validate='one_to_one').reset_index()
    for field in ['C_ce','C_correct']:pairs['delta_'+field]=pairs[field+'_candidate']-pairs[field+'_control']
    means=pairs.groupby('seed')[['delta_C_ce','delta_C_correct']].mean().reset_index()
    means['CE_prediction']=means.delta_C_ce<0;means['correctness_prediction']=means.delta_C_correct>=0
    return dict(paired_rows=pairs.to_dict('records'),per_seed=means.to_dict('records'),
        predicted_joint_supported_descriptively=bool((means.CE_prediction&means.correctness_prediction).all()),
        scope='Exact agent prediction only, not a replacement for original phenotype gates or evidence of a human mechanism',
        gradient_ratio_measured=False,independent_initialization_seeds=2,task_seed_pairs=8)


def run(campaign,destination):
    campaign=Path(campaign);destination=Path(destination);r=read(campaign/'development_result.json')
    f=pd.DataFrame([dict(condition=label,**row) for label,key in [('candidate','individual_results'),('control','matched_control_results')] for row in r[key]])
    summary=hypothesis(f);write(destination/'AGENT_PREDICTION_CHECK.json',summary,True)
    pd.DataFrame(summary['paired_rows']).to_csv(destination/'agent_prediction_pairs.csv',index=False)
    pd.DataFrame(summary['per_seed']).to_csv(destination/'agent_prediction_seed_means.csv',index=False)
    ids=read(campaign/'paired_jobs.json');positions=[]
    for condition,jobs in ids.items():
        for id in jobs:
            job=campaign/'jobs'/id;task=read(job/'evaluation.json')['task_id']
            p=pd.read_csv(job/'positions.csv',float_precision='round_trip');p=p[p['update'].eq(330)&p.cue.eq('C')].copy();p['condition']=condition;p['task_id']=task;positions.append(p)
    pos=pd.concat(positions,ignore_index=True);pos.to_csv(destination/'C_endpoint_positions.csv',index=False)
    keys=['seed','task_id','position'];a=pos[pos.condition.eq('candidate')].set_index(keys);b=pos[pos.condition.eq('control')].set_index(keys)
    if len(a)!=56 or len(b)!=56:raise ValueError('Missing C endpoint positions')
    joined=a[['target','correct','ce','margin']].join(b[['target','correct','ce','margin']],lsuffix='_candidate',rsuffix='_control',validate='one_to_one')
    if not joined.target_candidate.equals(joined.target_control):raise ValueError('Different paired target labels')
    joined['became_correct']=joined.correct_candidate&~joined.correct_control
    joined['became_incorrect']=~joined.correct_candidate&joined.correct_control
    joined.reset_index().to_csv(destination/'C_position_changes.csv',index=False)
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='controlled-pilot-v2-summary'
    import matplotlib.pyplot as plt
    pairs=pd.DataFrame(summary['paired_rows']);fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for i,seed in enumerate(SEEDS):
        part=pairs[pairs.seed.eq(seed)]
        for ax,field in zip(axes,['C_ce','C_correct']):
            ax.scatter(part[field+'_control'],part[field+'_candidate'],label=f'seed {seed}: four tasks',marker=['o','^'][i],s=48,alpha=.75)
    bound=max(pairs.C_ce_control.max(),pairs.C_ce_candidate.max())*1.05
    axes[0].plot([0,bound],[0,bound],color='gray',lw=1)
    axes[0].axhline(np.log(4),color='gray',ls=':',lw=.8);axes[0].axvline(np.log(4),color='gray',ls=':',lw=.8)
    axes[0].set(xlabel='Control C cross-entropy',ylabel='Candidate C cross-entropy',title='Fixed update330: C loss',xlim=(0,bound),ylim=(0,bound))
    axes[1].plot([0,7],[0,7],color='gray',lw=1);axes[1].set(xlabel='Control correct C positions',ylabel='Candidate correct C positions',title='C correctness',xlim=(-.25,7.25),ylim=(-.25,7.25),xticks=range(8),yticks=range(8))
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('Eight task-seed pairs, two initialization seeds')
    fig.savefig(destination/'agent_prediction_endpoints.png',dpi=160);fig.savefig(destination/'agent_prediction_endpoints.svg',metadata={'Date':None});plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',required=True);p.add_argument('--destination',required=True);args=p.parse_args()
    result=run(args.campaign,args.destination);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
