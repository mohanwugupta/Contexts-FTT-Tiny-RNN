"""Table-driven paired-seed plots and editable vector exports."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import CONTRACT


def save_figure(fig,out,name):
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    for suffix in ['png','pdf','svg']:fig.savefig(Path(out)/'figures'/f'{name}.{suffix}',dpi=160,bbox_inches='tight')
    plt.close(fig)


def band(ax,frame,x,y,label,color,seed_lines=False,style='-'):
    wide=frame.pivot(index='seed',columns=x,values=y).sort_index();v=wide.to_numpy(float)
    if seed_lines:
        for row in v:ax.plot(wide.columns,row,color='gray',alpha=.12,lw=.5)
    idx=np.random.default_rng(CONTRACT['bootstrap_seed']).integers(0,len(v),(10000,len(v)))
    b=v[idx].mean(1);mean=v.mean(0)
    ax.plot(wide.columns,mean,label=label,color=color,ls=style,lw=1.7)
    ax.fill_between(wide.columns,np.quantile(b,.025,axis=0),np.quantile(b,.975,axis=0),color=color,alpha=.12)


def figure_A(out,fixed):
    probes=['Mixed','NoContext','Novel30Mean','Novel60Mean'];fig,axes=plt.subplots(4,4,figsize=(16,12),layout='constrained')
    f=fixed[fixed.condition.eq('High_monotonic')]
    colors={'natural':'#777777','single_midpoint':'#2166ac','repeated_midpoint':'#339566'}
    for j,cue in enumerate(probes):
        for i,k in enumerate([1,3]):
            g=f[f.cue.eq(cue)&f.k.eq(k)]
            for row,metric in [(2*i,'raw_ce'),(2*i+1,'midpoint_task_distance')]:
                ax=axes[row,j]
                for branch in ['natural','single_midpoint','repeated_midpoint']:
                    q=g[g.branch.eq(branch)];band(ax,q,'lag',metric,branch.replace('_',' '),colors[branch],seed_lines=branch=='single_midpoint')
                ax.axvline(0,color='#999999',ls=':',lw=.7);ax.set(title=f'{cue}, k={k}' if row%2==0 else '',xlabel='Lag after placement',ylabel='CE (nats)' if metric=='raw_ce' else 'Absolute task midpoint distance')
                if row==0 and j==0:ax.legend(fontsize=7)
    fig.suptitle('What happens when we stop helping?  Lag 0 is an external state placement',fontsize=15)
    save_figure(fig,out,'fig_p1_single_patch_timecourse')
