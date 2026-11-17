"""Unsmoothed, table-derived scientific figures with deterministic exports."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .config import PROBES

POLICIES=['native','best_familiar_line','best_familiar_square','best_extended_grid','bounded_oracle']
LABELS=['Native','Line best','Square best','Box grid best','Cue oracle']
ARMS=['N0','E','L','R','I']


def export(fig,destination,name):
    path=Path(destination)/'figures';path.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        p=path/(name+'.'+ext)
        if p.exists():raise FileExistsError('Figure output exists: '+str(p))
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else {'Date':None} if ext=='svg' else {'Software':'context-cue-access-v1'}
        fig.savefig(p,dpi=150,bbox_inches='tight',metadata=metadata)
    plt.close(fig)


def render_all(data,destination,tables=None):
    from .analysis import derived_tables
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'context-cue-access-v1'})
    tables=derived_tables(data) if tables is None else tables
    figure_a(data,destination,tables['s4_s1_intervals'],False)
    figure_a(data,destination,tables['s4_s1_intervals'],True)
    for k,name in [(3,'fig_B_cue_history'),(5,'fig_B_cue_history_k5')]:figure_b(data,destination,tables['s4_paired_intervals'],k,name)
    figure_c(data,destination)


def figure_a(data,destination,boot,exemplar=False):
    data=Path(data);cols=['seed','cue','k','kind','cue_id','u','v','alpha','beta','future_ce','viable_continuation']
    f=pd.read_csv(data/'tables/s1_grid_seed.csv',usecols=cols,float_precision='round_trip');f=f[f.k.eq(1)&f.kind.eq('grid')]
    best=pd.read_csv(data/'tables/s1_best_cues.csv',float_precision='round_trip');reg=pd.read_csv(data/'tables/input_registry.csv',float_precision='round_trip');anchors=reg[reg.kind.eq('exact_anchor')].set_index('anchor_name')
    grouped=[];seeds={}
    for cue in PROBES:
        g=f[f.cue.eq(cue)]
        if exemplar:
            b=best[best.cue.eq(cue)&best.k.eq(1)];native=b[b.policy.eq('native')].set_index('seed').viable_continuation;grid=b[b.policy.eq('best_extended_grid')].set_index('seed').viable_continuation
            eligible=grid[grid&~native].index
            if not len(eligible):grouped.append(pd.DataFrame());seeds[cue]=None;continue
            seeds[cue]=int(min(eligible));g=g[g.seed.eq(seeds[cue])]
        grouped.append(g.groupby(['u','v','alpha','beta'],as_index=False).agg(future_ce=('future_ce','mean'),viable=('viable_continuation','mean')))
    vmax=max(float(g.future_ce.max()) for g in grouped if len(g));norm=Normalize(vmin=0,vmax=vmax)
    fig,axes=plt.subplots(4,3,figsize=(15,15),gridspec_kw={'height_ratios':[1,1,.4,.4]},layout='constrained')
    centers=np.linspace(-1.5,1.5,13);edges=np.r_[-1.5,(centers[:-1]+centers[1:])/2,1.5];uu,vv=np.meshgrid(edges,edges,indexing='ij')
    for i,(cue,g) in enumerate(zip(PROBES,grouped)):
        ax=axes[i//3,i%3]
        if not len(g):ax.text(.1,.5,'No qualifying rescued seed',transform=ax.transAxes);continue
        matrix=g.pivot(index='u',columns='v',values='future_ce').reindex(index=centers,columns=centers).to_numpy()
        mesh=ax.pcolormesh(uu-vv,uu+vv,matrix,cmap='viridis_r',norm=norm,shading='flat')
        v=g[g.viable.gt(0)];ax.scatter(v.alpha,v.beta,s=35*v.viable,facecolors='none',edgecolors='white',linewidth=.7)
        ax.plot([0,1,0,-1,0],[0,1,2,1,0],color='red',ls='--',lw=.9,label='familiar square')
        ax.plot([-1,1],[1,1],color='red',lw=1.8,label='familiar line')
        ax.plot([0,3,0,-3,0],[-3,0,3,0,-3],color='black',lw=.8)
        for name,marker,color in [('A','s','black'),('B','s','black'),(cue,'*','#ff3333')]:
            a=anchors.loc[name];ax.scatter(a.alpha,a.beta,marker=marker,c=color,s=50 if marker=='*' else 22,zorder=5)
            if name in ['A','B']:ax.annotate(name,(a.alpha,a.beta),xytext=(4,3),textcoords='offset points',color='black')
        title=cue+(f'; lowest rescued seed {seeds[cue]}' if exemplar else '; mean over 20 seeds')
        ax.set(title=title,xlabel='alpha = u − v',ylabel='beta = u + v',xlim=(-3.1,3.1),ylim=(-3.1,3.1));ax.set_aspect('equal')
        fig.colorbar(mesh,ax=ax,label='Future CE (nats)',shrink=.8)
        if i==0:ax.legend(loc='lower left',fontsize=6)
    for i,cue in enumerate(PROBES):
        ax=axes[2+i//3,i%3];g=boot[boot.cue.eq(cue)&boot.k.eq(1)&boot.metric.eq('viable_continuation')].set_index('policy').reindex(POLICIES)
        rates=g['mean'].to_numpy(float);err=np.c_[rates-g.ci_low.to_numpy(),g.ci_high.to_numpy()-rates].T*20
        ax.bar(np.arange(5),rates*20,yerr=err,color=['#666666','#8fc0a9','#5ba2a5','#3b7b9c','#c17d49'],capsize=2)
        ax.set(title=cue+'; full cohort',ylim=(0,22),ylabel='Viable seeds / 20',xticks=np.arange(5),xticklabels=LABELS);ax.tick_params(axis='x',labelrotation=20)
        for j,value in enumerate(rates):
            if np.isfinite(value):ax.text(j,value*20+.3,str(round(value*20)),ha='center',fontsize=8)
    fig.suptitle('Single-pulse cue access: k=1; six identical future target-cue predictions\nWhite ring size = viable fraction at that input; star = target. Signed box is numeric capacity. Native is fixed; other bars use future-informed selection.',fontsize=12)
    export(fig,destination,'fig_A_seed_exemplars' if exemplar else 'fig_A_single_cue_accessibility')


def figure_b(data,destination,boot,k,name):
    data=Path(data);f=pd.read_csv(data/'tables/s2_seed_summary.csv',float_precision='round_trip');f=f[f.k.eq(k)&(f.policy.eq('bounded_oracle')|f.arm.eq('N0'))]
    fig=plt.figure(figsize=(15,12),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[1,1,.8]);colors=['#777777','#8aaa55','#3883b8','#b95b60','#825c9b']
    for i,cue in enumerate(PROBES):
        ax=fig.add_subplot(gs[i//3,i%3]);g=f[f.cue.eq(cue)];wide=g.pivot(index='seed',columns='arm',values='future_ce').reindex(columns=ARMS).sort_index()
        for _,row in wide.iterrows():ax.plot(range(5),row,color='gray',alpha=.25,lw=.6)
        for j,arm in enumerate(ARMS):
            v=wide[arm].to_numpy();ax.scatter(j+np.linspace(-.12,.12,len(v)),v,s=14,color=colors[j],alpha=.8)
        ax.axvline(3.5,color='black',ls=':',lw=.8);ax.set(title=cue,xlabel='Native; E/L/R: 2 parameters; I: 4',ylabel='Mean future CE (nats)',xticks=range(5),xticklabels=ARMS)
    ax=fig.add_subplot(gs[2,0]);ax.axis('off');ax.text(.02,.95,f'Same final probe, k={k}\n\nSame natural anchor\n↓ same token, prep cue1\n↓ same token, prep cue2\n↓ same token, FIXED TARGET CUE\n↓ identical target-cue future inputs\n\nScore only {7-k} subsequent predictions.\nR ties prep values; L changes only late.\nI has four parameters + larger finite menu.',va='top',fontsize=11)
    ax=fig.add_subplot(gs[2,1]);counts=f.groupby(['cue','arm']).viable_continuation.agg(lambda v:float(v.sum()) if v.notna().all() else np.nan).unstack().reindex(index=PROBES,columns=ARMS)
    im=ax.imshow(counts.to_numpy(),vmin=0,vmax=20,cmap='Blues');ax.set(title='Viable source futures (n / 20)',xticks=range(5),xticklabels=ARMS,yticks=range(6),yticklabels=PROBES)
    for i in range(6):
        for j in range(5):
            value=counts.iloc[i,j];ax.text(j,i,str(int(value)) if np.isfinite(value) else 'NA',ha='center',va='center',color='white' if value>10 else 'black')
    ax=fig.add_subplot(gs[2,2]);g=boot[boot.k.eq(k)&boot.policy.eq('bounded_oracle')&boot.metric.eq('R_minus_L_future_ce')].set_index('cue').reindex(PROBES)
    ax.errorbar(g['mean'],np.arange(6),xerr=np.c_[g['mean']-g.ci_low,g.ci_high-g['mean']].T,fmt='o',capsize=3,color='#b95b60');ax.axvline(0,color='gray',ls='--');ax.set(title='Paired R − L, 95% seed interval',xlabel='Future CE difference (negative favors R)',yticks=range(6),yticklabels=PROBES);ax.invert_yaxis()
    fig.suptitle('Preparation with identical final physical target and future tokens\nBounded cue oracles use correct future labels: six starts × 80 updates for every controlled arm.',fontsize=13)
    export(fig,destination,name)


def figure_c(data,destination):
    data=Path(data);s=pd.read_csv(data/'tables/s3_sensitivities.csv');s=s[s.k.eq(3)&s.schedule_kind.eq('N0')]
    p=pd.read_csv(data/'tables/s3_paired_control_diagnostic.csv');p=p[p.k.eq(3)&p.schedule_kind.eq('N0')]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');names=['late','tied','two'];colors=['#3883b8','#b95b60','#825c9b']
    for j,(name,color) in enumerate(zip(names,colors)):
        f=s[s.matrix.eq(name)].sort_values(['seed','cue']);x=j+np.linspace(-.2,.2,len(f));axes[0].scatter(x,f.rank_combined,s=12,color=color,alpha=.5)
        axes[1].scatter(x,f.gradient_alignment,s=12,color=color,alpha=.5)
    axes[0].set(title='Local effective rank at baseline',ylabel='Rank (absolute AND relative cutoff)',xticks=range(3),xticklabels=['Late (2)','Tied repeat (2)','Independent (4)'],ylim=(-.2,4.2))
    axes[1].set(title='Future-label gradient support',ylabel='Gradient norm fraction in control span',xticks=range(3),xticklabels=['Late','Tied repeat','Independent'],ylim=(-.02,1.02))
    for cue in PROBES:
        f=p[p.cue.eq(cue)];axes[2].scatter(f.alignment_tied_minus_late,-f.R_minus_L_future_ce,label=cue,s=20,alpha=.7)
    axes[2].axhline(0,color='gray',ls='--');axes[2].axvline(0,color='gray',ls='--');axes[2].set(title='Paired within seed and target',xlabel='Baseline alignment: tied − late',ylabel='Actual CE benefit: late − repeated');axes[2].legend(fontsize=7)
    fig.suptitle('Local finite-horizon input sensitivity, k=3. Points are repeated seed/probe cases; association is not mediation.',fontsize=12)
    export(fig,destination,'fig_C_local_control')
