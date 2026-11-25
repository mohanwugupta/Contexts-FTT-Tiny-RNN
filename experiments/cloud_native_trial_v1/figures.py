"""Compact exact native A/B, specificity and transfer plots, saved results only."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def draw(root,coverage):
    root=Path(root);frames=[]
    for r in coverage:
        p=root/'jobs'/r['job']/'metrics.csv'
        if r['kind']=='target' and p.exists():
            f=pd.read_csv(p);f['arm']=r['arm'];f['task_id']=r['task_id'];frames.append(f)
    if not frames:return
    f=pd.concat(frames,ignore_index=True);f.to_csv(root/'plot_values.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for arm,style in [('control','-'),('candidate','--')]:
        for cue,color in [('A','tab:blue'),('B','tab:orange')]:
            g=f[f.arm.eq(arm)&f.cue.eq(cue)&f['update'].le(100)]
            for _,v in g.groupby(['seed','task_id']):axes[0].plot(v['update'],v.ce,color=color,ls=style,alpha=.1,lw=.6)
            v=g.groupby('update').ce.mean();axes[0].plot(v.index,v,color=color,ls=style,label=arm+' '+cue)
    for n in range(10,101,10):axes[0].axvline(n,color='.7',lw=.4)
    axes[0].set(xlabel='Frozen update',ylabel='CE',title='Early A/B: original curves')
    summary=pd.read_csv(root/'seed_task_results.csv')
    for arm,offset in [('control',-.12),('candidate',.12)]:
        g=summary[summary.arm.eq(arm)]
        for i,metric in enumerate(['D_A','D_B']):
            axes[1].scatter([i+offset]*len(g),g[metric],alpha=.35,label=arm if i==0 else None)
        for i,cue in enumerate(['A','B','C','Mixed','NoContext']):
            axes[2].scatter([i+offset]*len(g),g[cue+'_ce'],alpha=.35,label=arm if i==0 else None)
    axes[1].axhline(0,color='.6',lw=.7);axes[1].set(xticks=[0,1],xticklabels=['D_A','D_B'],ylabel='Signed gain difference',title='Functional specificity')
    axes[2].axhline(1.38629436112,color='.6',lw=.7);axes[2].set(xticks=range(5),xticklabels=['A','B','C','Mixed','None'],ylabel='CE at330',title='All development networks')
    for ax in axes:ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(root/('trial_summary.'+ext),dpi=160)
    plt.close(fig)
