"""Predetermined scientific figures regenerated only from exported tables."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,out,name):
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    for ext in ['png','pdf','svg']:fig.savefig(Path(out)/'figures'/f'{name}.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def e1_figures(out,summary,patch,changes):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for cohort,ax in zip(['canonical','constructive'],axes[0]):
        f=summary[summary.cohort.eq(cohort)&summary.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
        for name,g in f.groupby('condition'):ax.scatter(g.closure_task_norm,g.raw_ce,s=13,alpha=.5,label=name)
        ax.set(title=cohort.title(),xlabel='Mean output-relevant closure defect',ylabel='Natural CE (nats)');ax.legend(fontsize=7)
    ax=axes[1,0];f=patch[patch.condition.eq('High_monotonic')&~patch.probe.isin(['A','B'])].copy()
    f['plot_probe']=f.probe.replace({'Novel30_A':'Novel30Mean','Novel30_B':'Novel30Mean','Novel60_A':'Novel60Mean','Novel60_B':'Novel60Mean'})
    for name,g in f.groupby('plot_probe'):
        # Equal direction/position means are taken within seed before the across-seed mean.
        v=g.groupby(['seed','probe','lag']).error_task_norm.mean().groupby(['seed','lag']).mean().groupby('lag').mean()
        ax.plot(v.index,v.values,marker='o',label=name)
    ax.set(title='Single placement: subsequent task error',xlabel='Transitions after placement',ylabel='Absolute task error');ax.legend(fontsize=8)
    ax=axes[1,1]
    f=changes[changes.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
    for name,g in f.groupby('contrast'):
        x=g[g.metric.eq('closure_task_norm')].set_index(['seed','probe']).effect
        y=g[g.metric.eq('raw_ce')].set_index(['seed','probe']).effect
        ax.scatter(x,y,s=13,alpha=.5,label=name)
    ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7);ax.set(title='Matched closure-arm changes',xlabel='Δ output closure defect',ylabel='Δ CE (nats)');ax.legend(fontsize=8)
    save(fig,out,'e1_accessibility')


def e2_figures(out,grid):
    conditions=['High_monotonic','Low_monotonic','C0_canonical','C1_contrast_only','C2_closure_only','C3_full']
    fig,axes=plt.subplots(2,6,figsize=(20,7),layout='constrained')
    for j,c in enumerate(conditions):
        f=grid[grid.condition.eq(c)&grid.cue_kind.eq('grid')]
        for i,metric in enumerate(['raw_ce','closure_task_norm']):
            p=f.groupby(['cue_beta','cue_alpha'])[metric].mean().unstack().sort_index()
            ax=axes[i,j];im=ax.imshow(p.values,origin='lower',aspect='auto',cmap='viridis')
            ax.set_xticks(range(len(p.columns)),[f'{x:.2g}' for x in p.columns],rotation=60);ax.set_yticks(range(len(p.index)),[f'{x:.2g}' for x in p.index]);ax.set(title=c if i==0 else metric,xlabel='alpha',ylabel='beta');fig.colorbar(im,ax=ax,shrink=.7)
    save(fig,out,'e2_context_grid')


def factorial_figure(out,prefix,familiar,summary):
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    arms=['F00','F01','F10','F11'];colors=['#467ca7','#5d9e8e','#ca814b','#915998']
    for ax,metric,title in [(axes[0,0],'familiar_mean_ce','Familiar A/B competence'),(axes[0,1],'hidden_separation','Realized hidden separation'),(axes[1,0],'closure_fixed_H','Familiar-derived neutral closure')]:
        for i,c in enumerate(arms):
            g=familiar[familiar.condition.eq(c)];ax.bar(i,g[metric].mean(),color=colors[i]);ax.scatter(np.full(len(g),i),g[metric],s=8,color='black',alpha=.3)
        ax.set_xticks(range(4),arms);ax.set(title=title,ylabel=metric)
    ax=axes[1,1]
    for i,c in enumerate(arms):
        f=summary[summary.condition.eq(c)&summary.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
        m=f.groupby('probe').raw_ce.mean().reindex(['Mixed','NoContext','Novel30Mean','Novel60Mean']);ax.plot(range(4),m,marker='o',label=c,color=colors[i])
    ax.set_xticks(range(4),['Mixed','NoContext','Novel30','Novel60']);ax.set(title='Natural changed-context predictions',ylabel='CE (nats)');ax.legend()
    save(fig,out,prefix+'_factorial')


def pilot_figure(out,audit):
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].scatter(audit.min_separation_gap,audit.min_closure_reduction,c=audit.competence_passed.astype(int),cmap='coolwarm',s=35)
    axes[0].axvline(.25,color='gray',ls='--');axes[0].axhline(.01,color='gray',ls='--');axes[0].set(xlabel='Minimum strong − weak hidden gap',ylabel='Minimum relative closure reduction',title='All 48 familiar-only candidates')
    fields=['competence_passed','learning_passed','separation_passed','closure_passed','gradient_passed','feasible'];axes[1].barh(fields,[int(audit[x].sum()) for x in fields]);axes[1].set(xlabel='Number of candidates',title='Prespecified feasibility checks')
    save(fig,out,'e3_pilot_feasibility')


def replication_figure(out,effects,registry,blocked=False):
    sequences=registry['sequences'][1:];fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,seq in zip(axes,sequences):
        sid=seq['sequence_id'];ax.set_title(sid+'  '+''.join(str(x) for x in seq['tokens']))
        if blocked:
            ax.axis('off');ax.text(.1,.55,'Design frozen\nNo model run\nBlocked by E3 feasibility',va='center',transform=ax.transAxes);continue
        f=effects[effects.sequence_id.eq(sid)&effects.contrast.eq('interaction')&effects.metric.eq('raw_ce')&effects.probe.isin(['Mixed','NoContext','Novel30Mean','Novel60Mean'])]
        for i,(_,r) in enumerate(f.iterrows()):ax.errorbar(i,r['mean'],yerr=[[max(0,r['mean']-r.ci_low)],[max(0,r.ci_high-r['mean'])]],fmt='o',capsize=3)
        ax.axhline(0,color='gray',lw=.7);ax.set_xticks(range(len(f)),f.probe,rotation=30,ha='right');ax.set_ylabel('Closure × separation interaction (CE)')
    save(fig,out,'e4_replication_grid')
