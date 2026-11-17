"""Three table-derived figures; no training, decoder fitting or model evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.context_cue_access.plots import export


def line_band(ax,boot,selector,label,color,style='-'):
    f=boot.copy()
    for k,v in selector.items():f=f[f[k].eq(v)]
    f=f.sort_values('transition');x=f.transition.to_numpy();ax.plot(x,f['mean'],label=label,color=color,ls=style)
    ax.fill_between(x,f.ci_low,f.ci_high,color=color,alpha=.13)


def render_all(data,destination,tables=None):
    from .analysis import derived_tables
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'svg.fonttype':'none','svg.hashsalt':'context-mixing-sequence-v1','pdf.fonttype':42})
    tables=derived_tables(data) if tables is None else tables
    figure1(data,destination,tables);figure2(data,destination,tables);figure3(data,destination,tables)


def figure1(data,destination,tables):
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for cue,color in [('A','#2878a8'),('B','#c17d49'),('Mixed','#b34a59')]:line_band(axes[0],tables['d4_native_position_intervals'],{'cue':cue,'metric':'raw_ce'},cue,color)
    axes[0].set(title='Natural source prediction',xlabel='Scored position',ylabel='Next-token CE (nats)');axes[0].legend()
    for metric,label,color in [('e_task_norm','Observed Mixed − midpoint','#222222'),('j_task_norm','New nonlinear mismatch j','#4b9b70'),('r_task_norm','Propagation r','#8263a0')]:line_band(axes[1],tables['d4_mixing_position_intervals'],{'metric':metric},label,color)
    axes[1].set(title='Output-relevant components',xlabel='Scored position',ylabel='Centered-logit component norm');axes[1].legend(fontsize=8)
    for branch,label,color,style in [('natural_Mixed','Natural Mixed','#b34a59','-'),('one_time_correction','Single t1 midpoint correction','#2878a8','-'),('external_midpoint_readout','External midpoint readout','#4b9b70','--')]:
        line_band(axes[2],tables['d4_correction_curve_intervals'],{'branch':branch,'metric':'raw_ce'},label,color,style)
    axes[2].axvline(1,color='gray',ls=':');axes[2].set(title='Stop helping after position1',xlabel='Position (lag0 at t=1)',ylabel='CE (nats)');axes[2].legend(fontsize=8)
    fig.suptitle('Frozen High: same sequence and physical Mixed cue; 20 seed-cluster intervals\nThe exact vector identity is not additive CE causation. Correction at t1 is exogenous; midpoint readout is not a trajectory.',fontsize=12)
    export(fig,destination,'fig_1_mixing_failure')


def figure2(data,destination,tables):
    data=Path(data);f=pd.read_csv(data/'tables/d2_transition_history.csv');s=pd.read_csv(data/'tables/d2_history_seed.csv');decoder=pd.read_csv(data/'tables/d2_decoder_seed.csv');sanity=pd.read_csv(data/'tables/d2_decoder_sanity.csv')
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained');boot=tables['d4_mixing_position_intervals'];g=boot[boot.metric.eq('mixed_excess_ce')].sort_values('transition')
    colors=['#b34a59' if t in [1,3,4,7] else '#2878a8' for t in g.transition];x=np.arange(1,8);v=g['mean'].to_numpy()
    axes[0].bar(x,v,color=colors,yerr=np.c_[v-g.ci_low.to_numpy(),g.ci_high.to_numpy()-v].T,capsize=3);axes[0].axhline(0,color='gray',lw=.8)
    axes[0].set(title='Mixed excess over mean familiar loss',xlabel='Position: current token → target',ylabel='CE excess (nats)',xticks=x,xticklabels=['1:0→1','2:1→2','3:2→0','4:0→3','5:3→1','6:1→2','7:2→3']);axes[0].tick_params(axis='x',labelrotation=35)
    axes[0].text(.02,.97,'Red: history-dependent\nBlue: token-diagnostic',transform=axes[0].transAxes,va='top',fontsize=9)
    h=tables['d4_history_intervals'];h=h[h.metric.eq('history_contrast')].iloc[0]
    axes[1].scatter(s.seed,s.history_contrast,color='#555555',s=26);axes[1].axhline(0,color='gray',ls='--');axes[1].axhline(h['mean'],color='#b34a59');axes[1].axhspan(h.ci_low,h.ci_high,color='#b34a59',alpha=.12)
    axes[1].set(title=f'History − diagnostic excess\nMean {h["mean"]:.3f},95% CI[{h.ci_low:.3f},{h.ci_high:.3f}]',xlabel='Original trained seed',ylabel='Within-seed contrast (nats)')
    ax=axes[2];counts=[]
    for metric,status,color,offset,label in [('prototype_correct','prototype_status','#4b9b70',-.17,'Prototype'),('ridge_correct','ridge_status','#8263a0',.17,'Ridge λ1')]:
        valid=sanity.loc[sanity[status].eq('shared_decoder_sanity_passed'),'seed'];n=len(valid);counts.append(f'{label}: {n}/20 validated')
        if n:
            d=decoder[decoder.seed.isin(valid)].groupby('cue')[metric].mean().reindex(['A','B','Mixed']);ax.bar(np.arange(3)+offset,d.values,width=.32,color=color,label=f'{label} (n={n})')
    if not any(sanity.prototype_status.eq('shared_decoder_sanity_passed')) and not any(sanity.ridge_status.eq('shared_decoder_sanity_passed')):ax.axis('off');ax.text(.1,.5,'diagnostic_nonidentifiable\nFamiliar transfer failed',transform=ax.transAxes)
    else:ax.set(ylim=(0,1.25),xticks=range(3),xticklabels=['A (pooled fit)','B (pooled fit)','Mixed'],ylabel='Position accuracy; conditional diagnostic',title='Shared-readout sanity is rare');ax.legend(fontsize=8)
    ax.text(.02,.98,'\n'.join(counts)+'\nOther cases: nonidentifiable, not lost information',transform=ax.transAxes,va='top',fontsize=8)
    fig.suptitle('Sequence-information diagnostic: all20 seeds in primary contrasts; only fixed A/B states fit decoders\nGrammar masks also confound position, token and difficulty.14 states provide little independent support.',fontsize=12)
    export(fig,destination,'fig_2_sequence_information')


def figure3(data,destination,tables):
    data=Path(data);plan=pd.read_csv(data/'tables/d3_training_schedule.csv');raw=pd.read_csv(data/'tables/d3_natural_seed.csv');fam=pd.read_csv(data/'tables/d3_familiar_validation.csv');effect=pd.read_csv(data/'tables/d3_effect_intervals.csv');paired=pd.read_csv(data/'tables/d3_paired_effects.csv')
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained');arms=['B','I','S'];colors=['#777777','#2878a8','#b34a59']
    mats=[]
    for arm in arms:
        g=plan[plan.arm.eq(arm)&plan.continuation_update.le(20)].copy();g['A_code']=g.cue.eq('A').astype(int);mats.append(g.pivot(index='transition',columns='continuation_update',values='A_code').to_numpy())
    ax=axes[0,0];ax.imshow(np.r_[mats[0],mats[1],mats[2]],aspect='auto',cmap='coolwarm',vmin=0,vmax=1,interpolation='none');ax.axhline(6.5,color='black');ax.axhline(13.5,color='black')
    ax.set(title='Familiar-only curricula (first20 of160)',xlabel='Continuation sequence',ylabel='Seven scored steps per arm',yticks=[3,10,17],yticklabels=['B: A10/B10','I: whole alternation','S: switch3→4'],xticks=[0,9,19],xticklabels=[1,10,20]);ax.text(.01,-.23,'Red=A, blue=B;80 A/80 B at each position\nSame tokens/targets; joint source loss and160 updates',transform=ax.transAxes,fontsize=9)
    ax=axes[0,1];g=raw[raw.cue.eq('Mixed')].pivot(index='seed',columns='arm',values='raw_ce').reindex(columns=arms).sort_index()
    for _,r in g.iterrows():ax.plot(range(3),r,color='gray',alpha=.3,lw=.7)
    for i,(arm,color) in enumerate(zip(arms,colors)):ax.scatter(i+np.linspace(-.09,.09,len(g)),g[arm],color=color,s=22,alpha=.8)
    ax.set(title='Held-out natural Mixed, fixed final330',xticks=range(3),xticklabels=arms,ylabel='Mean seven-step CE (nats)')
    ax=axes[0,2];p=paired[paired.cue.eq('Mixed')];e=effect[effect.cue.eq('Mixed')&effect.metric.eq('S_minus_I')].iloc[0]
    ax.scatter(p.seed,p.S_minus_I,color=np.where(p.S_minus_I<0,'#2878a8','#b34a59'));ax.axhline(0,color='gray',ls='--');ax.axhline(e['mean'],color='black');ax.axhspan(e.ci_low,e.ci_high,color='gray',alpha=.15)
    ax.set(title=f'Paired S−I: {e["mean"]:.3f}\n95% seed interval[{e.ci_low:.3f},{e.ci_high:.3f}]',xlabel='Original trained seed',ylabel='Mixed CE difference (negative favors S)')
    ax=axes[1,0];boot=tables['d4_familiar_intervals']
    for i,(arm,color) in enumerate(zip(arms,colors)):
        for j,metric in enumerate(['A_ce','B_ce']):
            b=boot[boot.arm.eq(arm)&boot.metric.eq(metric)].iloc[0];x=i+(j-.5)*.26;ax.errorbar([x],[b['mean']],yerr=[[b['mean']-b.ci_low],[b.ci_high-b['mean']]],fmt='o' if j==0 else 's',color=color,capsize=3)
    ax.set(title='Familiar A (circle) and B (square)',xticks=range(3),xticklabels=arms,ylabel='Natural CE (nats)');ax.text(.02,.98,'Familiar mean guard: |S−I|≤.20\nDescriptive; never an exclusion gate',transform=ax.transAxes,va='top',fontsize=9)
    for ax,metric,original,title in [(axes[1,1],'hidden_separation',tables['d4_original_separation_seed'].hidden_separation.mean(),'Raw A/B hidden separation'),(axes[1,2],'task_separation',tables['d4_original_separation_seed'].task_separation.mean(),'Centered-logit A/B separation')]:
        for i,(arm,color) in enumerate(zip(arms,colors)):
            values=fam[fam.arm.eq(arm)].sort_values('seed')[metric].to_numpy();ax.scatter(i+np.linspace(-.12,.12,len(values)),values,color=color,s=20,alpha=.65)
            b=boot[boot.arm.eq(arm)&boot.metric.eq(metric)].iloc[0];ax.errorbar([i],[b['mean']],yerr=[[b['mean']-b.ci_low],[b.ci_high-b['mean']]],fmt='D',color='black',capsize=3)
        ax.axhline(original,color='gray',ls='--',label='Original High mean');ax.set(title=title,xticks=range(3),xticklabels=arms,ylabel='Mean matched-position distance');ax.legend(fontsize=8)
    fig.suptitle('Same original High model + policy per seed; fresh matched Adam; only familiar cue schedules differ\nAll60 endpoints and9600 updates fixed before protected evaluation. Total curriculum effect is not causal mediation.',fontsize=13)
    export(fig,destination,'fig_3_continuity_training')
