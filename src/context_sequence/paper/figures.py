"""Predetermined figures and exact panel data, with seed-level uncertainty."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from context_sequence.statistics.bootstrap import bootstrap_indices

COLORS=['#225c96','#d47732','#33856d','#825b9e']
PROBES=['Mixed','NoContext','Novel30Mean','Novel60Mean']
LABELS={'Mixed':'Mixed','NoContext':'No context','Novel30Mean':'Novel 30°','Novel60Mean':'Novel 60°','A':'A','B':'B'}


def make_figures(out,tables,config):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.titleweight':'bold'})
    idx=bootstrap_indices();claims=[]
    checkpoint_hashes=json.dumps([r['checkpoint_sha256'] for r in config['checkpoints']])
    def panel(figid,name,df,source,filter_description,estimator):
        path=f'tables/panel_{figid}_{name}.csv';df.to_csv(out/path,index=False)
        claims.append({'figure':figid,'panel':name,'source_table':source,'panel_data':path,'filter':filter_description,'estimator':estimator,'source_checkpoint_hashes':checkpoint_hashes,'registry_sha256':config['novelty_registry_sha256'],'claim':estimator})
    def finish(fig,number,caption):
        fig.savefig(out/'figures'/f'figure_{number}.png',dpi=180,bbox_inches='tight')
        fig.savefig(out/'figures'/f'figure_{number}.pdf',bbox_inches='tight')
        fig.savefig(out/'figures'/f'figure_{number}.svg',bbox_inches='tight')
        plt.close(fig);(out/'captions'/f'figure_{number}.md').write_text(caption+'\n')
    def bars(ax,df,x,groups,metric,order=None):
        order=list(order or df[x].unique());levels=list(df[groups].unique()) if groups else [None];width=.78/len(levels)
        for j,level in enumerate(levels):
            means=[];lo=[];hi=[]
            for value in order:
                f=df[df[x].eq(value)];f=f[f[groups].eq(level)] if groups else f
                v=f.groupby('seed')[metric].mean().sort_index().to_numpy()
                if len(v)!=20:raise ValueError('Figure uncertainty requires 20 seeds')
                boot=v[idx].mean(1);m=v.mean();means.append(m);lo.append(m-np.quantile(boot,.025));hi.append(np.quantile(boot,.975)-m)
            xx=np.arange(len(order))+(j-(len(levels)-1)/2)*width
            ax.bar(xx,means,width,color=COLORS[j],alpha=.88,label=str(level).replace('_monotonic','').replace('_canonical','').replace('_contrast_only',' contrast').replace('_closure_only',' closure').replace('_full',' full') if level is not None else None)
            ax.errorbar(xx,means,yerr=[np.maximum(lo,0),np.maximum(hi,0)],fmt='none',color='#333333',capsize=3,lw=1)
        ax.set_xticks(range(len(order)),[LABELS.get(v,v) for v in order],rotation=20,ha='right')
        if groups:ax.legend(frameon=False,fontsize=8)
    if 'a1_canonical_phenotype_seed' in tables:
        s=tables['a1_canonical_phenotype_seed'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        acq=tables['s1_genuine_acquisition'];ax=axes[0,0]
        for j,(family,g) in enumerate(acq.groupby('family')):
            wide=g.pivot(index='seed',columns='training_sequence_index',values='raw_sequence_loss_pre_update').sort_index();a=wide.to_numpy();m=a.mean(0);bs=a[idx].mean(1)
            ax.plot(wide.columns,m,color=COLORS[j],label=family.replace('_monotonic',''));ax.fill_between(wide.columns,np.quantile(bs,.025,axis=0),np.quantile(bs,.975,axis=0),color=COLORS[j],alpha=.16)
        ax.set(title='A  Genuine acquisition',xlabel='Training sequence',ylabel='Pre-update CE (nats)');ax.legend(frameon=False)
        panel('1','A',acq,'s1_genuine_acquisition','all canonical High/Low acquisition rows','Mean pre-update CE; whole-seed bootstrap at each update')
        sep=s[s.probe.eq('A')].melt(id_vars=['seed','condition'],value_vars=['hidden_separation','task_separation'],var_name='space',value_name='separation')
        bars(axes[0,1],sep,'space','condition','separation');axes[0,1].set(title='B  Familiar differentiation',ylabel='A/B Euclidean distance')
        panel('1','B',sep,'a1_canonical_phenotype_seed','probe=A; hidden/task separation','Transition-averaged A/B distance, paired seeds')
        f=s[s.probe.isin(['A','B',*PROBES])];bars(axes[1,0],f,'probe','condition','raw_ce',['A','B',*PROBES]);axes[1,0].set(title='C  Natural sequence expression',ylabel='Sequence CE (nats)')
        panel('1','C',f,'a1_canonical_phenotype_seed','A,B,Mixed,NoContext,Novel30Mean,Novel60Mean','Raw sequence CE; Novel directions averaged within seed')
        f=s[s.probe.isin(PROBES)];bars(axes[1,1],f,'probe','condition','transfer_deficit',PROBES);axes[1,1].axhline(0,color='black',lw=.7);axes[1,1].set(title='D  Deficit relative to familiar performance',ylabel='Probe CE − familiar CE (nats)')
        panel('1','D',f,'a1_canonical_phenotype_seed','four primary probe families','Probe CE minus equal A/B mean within seed')
        finish(fig,1,'# Figure 1. Differentiation and graded generalization\nCanonical High/Low monotonic-objective cohorts, 20 paired seeds each. Acquisition uses changing-weight pre-update logs, not terminal-checkpoint replay. Natural probes use update 170, zero hidden state and seven teacher-forced transitions. Novel30/60 are within-seed means of two raw directions. Intervals use the shared 10,000 whole-seed bootstrap draws. This experiment establishes association, not a causal effect of splitting.')
    if 'a3_composite_sufficiency_seed' in tables:
        s=tables['a3_composite_sufficiency_seed'];high=s[s.family.eq('High_monotonic')];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        axes[0,0].axis('off');axes[0,0].text(.04,.96,'A  Familiar-derived state',weight='bold',fontsize=13,va='top')
        axes[0,0].text(.08,.78,'A trajectory  +  B trajectory\n              ↓  average\n     shared hidden state\n              ↓  output head\n     next-token prediction',fontsize=11,linespacing=1.5,va='top')
        schematic=pd.DataFrame({'target':['(h_A + h_B) / 2'],'construction_inputs':['A,B'],'strength':[1.]});panel('2','A',schematic,'analysis_contract','fixed familiar-only target','Hidden midpoint decoded through original output weights')
        f=high[high.probe.isin(PROBES)&high.intervention.isin(['natural','repeated_composite'])];bars(axes[0,1],f,'probe','intervention','raw_ce',PROBES);axes[0,1].set(title='B  Repeated state placement',ylabel='Sequence CE (nats)')
        panel('2','B',f,'a3_composite_sufficiency_seed','High; natural/repeated composite; primary probes','Raw CE after repeated placement versus natural')
        f=high[high.probe.isin(PROBES)&high.intervention.isin(['single_composite','single_wrong_position','single_orthogonal'])];bars(axes[1,0],f,'probe','intervention','immediate_rescue',PROBES);axes[1,0].set(title='C  Acute positional specificity',ylabel='Lag-zero CE rescue (nats)')
        panel('2','C',f,'a3_composite_sufficiency_seed','High; single composite/wrong/orthogonal','Lag-zero rescue averaged over seven injection positions within seed')
        f=high[high.probe.isin(PROBES)&high.intervention.eq('repeated_composite')];bars(axes[1,1],f,'probe',None,'raw_rescue',PROBES);axes[1,1].set(title='D  Sufficiency across novelty levels',ylabel='Natural − patched CE (nats)')
        panel('2','D',f,'a3_composite_sufficiency_seed','High; repeated composite','Raw rescue, not normalized rescue fraction')
        finish(fig,2,'# Figure 2. Availability of a useful shared solution\nA/B-only hidden midpoint, fixed intervention strength one, frozen canonical High checkpoints (20 seeds). Repeated replacement is a sufficiency ceiling; acute effects average all seven injection positions within seed. Wrong-position and norm-matched orthogonal controls test specificity. Low networks, endpoint controls and individual directions are included in the complete source tables. No claim of necessity, unique mechanism, or natural midpoint generation follows from rescue.')
    if 'a4_maintenance_seed' in tables:
        lag=tables['a4_maintenance_lag_seed'];fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,metric,title in [(axes[0,0],'task_distance','A  Task-space drift after one injection'),(axes[0,1],'raw_rescue','B  Behavioral persistence')]:
            f=lag[lag.family.eq('High_monotonic')&lag.probe.isin(['Novel30Mean','Novel60Mean'])]
            for j,(key,g) in enumerate(f.groupby(['probe','continuation'])):
                means=g.groupby('lag')[metric].mean();ax.plot(means.index,means.values,label=LABELS[key[0]]+' '+key[1],color=COLORS[j//3],ls=['-','--',':'][j%3])
                ci=[]
                for l in means.index:
                    v=g[g.lag.eq(l)].sort_values('seed')[metric].to_numpy();bs=v[idx].mean(1);ci.append(np.quantile(bs,[.025,.975]))
                ci=np.array(ci);ax.fill_between(means.index,ci[:,0],ci[:,1],color=COLORS[j//3],alpha=.06)
            ax.set(title=title,xlabel='Transitions after injection',ylabel='Absolute centered-logit distance' if metric=='task_distance' else 'CE rescue (nats)');ax.legend(frameon=False,fontsize=7,ncol=2)
            panel('3','A' if metric=='task_distance' else 'B',f,'a4_maintenance_lag_seed','High; Novel30Mean/Novel60Mean; D0/D1/D2','Available injection positions averaged within seed at each lag; shared bootstrap')
        s=tables['a5_cue_generation_seed'];f=s[s.family.eq('High_monotonic')&s.analysis_kind.eq('local')&s.intervention.eq('Mixed')&s.probe.isin(['A','B',*PROBES])]
        bars(axes[1,0],f,'probe',None,'task_generation',['A','B',*PROBES]);axes[1,0].axhline(0,color='black',lw=.7);axes[1,0].set(title='C  Physical cue generation',ylabel='Native − Mixed task distance')
        panel('3','C',f,'a5_cue_generation_seed','High; identical-prestate Mixed branch','Positive values mean movement toward the familiar midpoint; seven positions averaged within seed')
        f=s[s.family.eq('High_monotonic')&s.analysis_kind.eq('takeover')&s.intervention.isin(['pulse_Mixed','persistent_Mixed'])&s.probe.isin(PROBES)]
        bars(axes[1,1],f,'probe','intervention','remaining_rescue',PROBES);axes[1,1].set(title='D  Pulse versus persistent cue takeover',ylabel='Mean remaining CE rescue (nats)')
        panel('3','D',f,'a5_cue_generation_seed','High; pulse/persistent Mixed takeover','Within-position remaining mean including takeover transition, then equal mean over positions within seed')
        finish(fig,3,'# Figure 3. Generation and maintenance\nSingle midpoint placement versus physical cue manipulation, canonical High cohort (20 seeds). D0 continues native context, D1 removes cue after injection, D2 continues physical Mixed. At lag l, 7−l positions contribute per seed; confidence intervals always resample seeds. Local branches share bit-identical prestates and never overwrite hidden state. Cue takeover retains the natural prefix. Improvement in behavior and movement toward the exact midpoint are separate outcomes. All rollouts are seven-transition teacher-forced trajectories, not evidence for autonomous attractors.')
    if 'a6_constructive_seed' in tables:
        s=tables['a6_constructive_seed'];scatter=tables['a7_geometry_behavior'];fig=plt.figure(figsize=(16,11),layout='constrained');gs=fig.add_gridspec(3,4)
        ax=fig.add_subplot(gs[0,0]);ax.axis('off');ax.text(0,.98,'A  Constructive design',weight='bold',fontsize=12,va='top')
        ax.text(0,.78,'C0  Sequence CE\nC1  + contrast pathway\nC2  + trajectory closure\nC3  Both mechanisms\n\nNo hidden-state patches\n20 matched seeds / condition',linespacing=1.5,fontsize=9,va='top')
        panel('4','A',pd.DataFrame({'condition':['C0','C1','C2','C3'],'contrast':[0,1,0,1],'closure':[0,0,1,1]}),'paper_contract','constructive cohort','Prospective factorial design; sequence-only C0 distinct from monotonic High')
        ax=fig.add_subplot(gs[0,1:3]);f=s[s.probe.isin(PROBES)];bars(ax,f,'probe','condition','raw_ce',PROBES);ax.set(title='B  Natural generalization',ylabel='Sequence CE (nats)');panel('4','B',f,'a6_constructive_seed','all C0–C3; primary probes','Natural CE; matched sequence-only seeds')
        ax=fig.add_subplot(gs[0,3]);f=s[s.probe.eq('A')];bars(ax,f,'condition',None,'task_separation');ax.set(title='C  A/B task separation',ylabel='Centered-logit distance');panel('4','C',f,'a6_constructive_seed','probe=A; denominator common to probes','Task A/B separation; hidden separation also retained')
        for i,p in enumerate(PROBES):
            ax=fig.add_subplot(gs[1,i]);f=scatter[scatter.probe.eq(p)];ax.axhline(0,color='#888888',lw=.7);ax.axvline(0,color='#888888',lw=.7);ax.scatter(f.delta_task_distance,f.delta_raw_ce,s=27,color=COLORS[i],alpha=.8)
            ax.set(title=f'{chr(68+i)}  {LABELS[p]}',xlabel='Δ absolute task distance',ylabel='Δ CE (nats)')
            panel('4',chr(68+i),f,'a7_geometry_behavior','probe='+p,'C3−C0, one paired point per seed; no fitted correlation')
        audit=tables['a8_effective_cue_audit'];f=audit[audit.condition.eq('C3_full')&audit.seed.eq(0)].copy()
        ax=fig.add_subplot(gs[2,:2]);ax.axhline(0,color='#888',lw=.7);ax.axvline(0,color='#888',lw=.7)
        for _,r in f.iterrows():
            v=json.loads(r.raw_vector);ax.scatter(v[0],v[1],s=40);ax.annotate(r.probe,(v[0],v[1]),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.plot([1,0],[0,1],ls='--',color='#777');ax.set(xlim=(-1.1,1.2),ylim=(-1.1,1.2),aspect='equal',title='H  Raw cue geometry',xlabel='A channel',ylabel='B channel');panel('4','H',f,'a8_effective_cue_audit','C3 seed 0; cue geometry shared across seeds','Raw vectors and familiar convex interpolation segment')
        ax=fig.add_subplot(gs[2,2:]);ax.scatter(f.cue_alpha,range(len(f)),color=COLORS[0]);ax.set_yticks(range(len(f)),f.probe);ax.axvline(0,lw=.7,color='#888');ax.set(title='I  Information retained by contrast pathway',xlabel='Effective alpha');panel('4','I',f,'a8_effective_cue_audit','C3 seed 0; projection fixed across seeds','Same-side Novel30/60 alias under A/B-only projection; discarded components exported')
        finish(fig,4,'# Figure 4. Constructive sufficiency and its boundary\nExisting Phase 1L sequence-only C0–C3 checkpoints, 20 matched seeds per condition, natural evaluation only. C0 is bit-identical to the saved High sequence-only cohort, not canonical High monotonic. C2/C3 closure uses a familiar-derived neutral cue numerically equal to Mixed, without protected probe behavioral supervision. Scatter axes show C3−C0 absolute task distance and CE; lower-left indicates joint improvement. Normalized-residual sensitivity is supplied separately. Alpha is calculated from cue vectors without label-specific rules; same-side 30°/60° cues become identical effective drives under C1/C3. Intervals use the shared paired-seed bootstrap; no outcome-driven panel selection.')
        fig,axes=plt.subplots(1,4,figsize=(14,3.5),layout='constrained')
        for ax,p in zip(axes,PROBES):
            f=scatter[scatter.probe.eq(p)];ax.scatter(f.delta_task_residual,f.delta_raw_ce,color=COLORS[0]);ax.axhline(0,color='#888',lw=.7);ax.axvline(0,color='#888',lw=.7);ax.set(title=LABELS[p],xlabel='Δ normalized residual',ylabel='Δ CE')
            panel('S1',p,f,'a7_geometry_behavior','probe='+p,'Normalized-residual sensitivity; paired C3−C0')
        finish(fig,'S1','# Supplement S1. Normalized geometry sensitivity\nThe same paired seeds and CE effects as Figure 4, using transitionwise-normalized task residual. Absolute distances and A/B denominators remain separately available.')
    return pd.DataFrame(claims)
