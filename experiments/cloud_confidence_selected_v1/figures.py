"""One fixed-time figure from completed saved-table products only."""
from pathlib import Path
import pandas as pd


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='cloud-confidence-selected-v1'
    import matplotlib.pyplot as plt
    out=Path(out);m=pd.read_csv(out/'initialization_seed_means.csv');ci=pd.read_csv(out/'initialization_seed_intervals.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
    for row,cue in enumerate(['C','NoContext']):
        for col,metric in enumerate(['CE_gain','correct_count_change']):
            ax=axes[row,col]
            for view,color in [('native','#1768a0'),('quarter','#8751a6')]:
                part=m[m.view.eq(view)&m.cue.eq(cue)].sort_values('update')
                for _,seed in part.groupby('seed'):ax.plot(seed['update'],seed[metric],color=color,alpha=.25,lw=.8)
                c=ci[ci.view.eq(view)&ci.cue.eq(cue)&ci.metric.eq(metric)].sort_values('update')
                ax.plot(c['update'],c['mean'],label=view,color=color)
                ax.fill_between(c['update'],c.ci_low,c.ci_high,color=color,alpha=.12)
            ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Fixed saved update',ylabel='Control CE − candidate CE (nats)' if col==0 else 'Candidate − control correct positions',title=cue);ax.legend(fontsize=8)
    fig.suptitle('All tasks averaged within two seeds; initialization contrast')
    fig.savefig(out/'fixed_time_initialization.png',dpi=160);fig.savefig(out/'fixed_time_initialization.svg',metadata={'Date':None});plt.close(fig)

    # Native outcomes only: a static expression view is not another trained
    # network and never supplies the four-outcome phenotype flag.
    from experiments.meta_objective_balance.statistics import interval
    f=pd.read_csv(out/'native/seed_task_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    labels=['A','B','C','Mixed','NoContext']
    for condition,offset,color in [('candidate',-.12,'#1768a0'),('control',.12,'#cf741c')]:
        g=f[f.condition.eq(condition)]
        early=int(g.original_early_phenotype.sum())
        for ax,field in zip(axes,['ce','correct']):
            means=[];xs=[]
            for i,cue in enumerate(labels):
                values=g[cue+'_'+field].to_numpy(float)
                ax.scatter([i+offset]*len(values),values,s=14,alpha=.25,color=color)
                by=g.groupby('seed')[cue+'_'+field].mean().reindex([21000,21001])
                ci=interval(by.to_numpy(float),[21000,21001]);xs.append(i+offset);means.append(ci['mean'])
                ax.errorbar(i+offset,ci['mean'],yerr=[[ci['mean']-ci['ci_low']],[ci['ci_high']-ci['mean']]],fmt='o',color=color,capsize=3)
            ax.plot(xs,means,color=color,label=condition+f' (early flag {early}/8 cells)')
    axes[0].axhline(1.3862943611198906,ls=':',color='gray');axes[0].set(ylabel='Native endpoint CE (nats)')
    axes[1].axhline(6,ls=':',color='gray');axes[1].set(ylabel='Native correct positions',ylim=(-.2,7.3))
    for ax in axes:ax.set(xticks=range(5),xticklabels=labels);ax.legend(fontsize=8)
    fig.suptitle('Fixed330; each condition has two seeds × four repeated tasks')
    fig.savefig(out/'native_outcomes.png',dpi=160);fig.savefig(out/'native_outcomes.svg',metadata={'Date':None});plt.close(fig)
