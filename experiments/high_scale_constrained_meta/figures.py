"""All figure families from saved tables only; no model imports or evaluations."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/high_scale_constrained_meta/v1';COL={'A':'#2166ac','B':'#ed8b23','C':'#ce3451','Mixed-average':'#27934c','Mixed-both':'#8a5bb4','NoContext':'#777777'};ARM={'U':'#557fa5','P':'#a64d35'}
plt.rcParams.update({'svg.hashsalt':'high_scale_constrained_meta_v1','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':140})
def read(name):return pd.read_csv(OUT/f'tables/{name}',float_precision='round_trip',low_memory=False)
def destination(value):
 p=Path(value) if value is not None else OUT/'figures'
 if (OUT/'manifest.json').exists() and p.resolve().is_relative_to(OUT.resolve()):raise ValueError('Sealed plots require an external directory')
 return p

def save(fig,out,name):
 fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/f'{name}.png',metadata={'Software':'high_scale_constrained_meta_v1'});fig.savefig(out/f'{name}.svg',metadata={'Date':None,'Creator':'high_scale_constrained_meta_v1'});plt.close(fig)
def band(ax,raw,summary,t,m,c,label=None):
 for _,g in raw.groupby('seed'):g=g.sort_values(t);ax.plot(g[t],g[m],color=c,alpha=.14,lw=.6)
 q=summary[summary.metric.eq(m)].sort_values(t);ax.plot(q[t],q['mean'],color=c,lw=1.6,label=label);ax.fill_between(q[t],q.ci_low,q.ci_high,color=c,alpha=.14,lw=0)
def pick(f,a,phase,dose):return f[f.arm.eq(a+'-'+phase)&f.dose.eq(dose)]

def scales(out):
 p=read('meta_group_scale_projection.csv.gz');t=read('meta_training_histories.csv');p=p.drop_duplicates(['seed','arm','group','cumulative_episode']);fig,axs=plt.subplots(2,3,figsize=(16,8));fig.suptitle('Weight RMS after each outer update: U unconstrained, P projected')
 for ax,g in zip(axs.flat,list(p.group.unique())):
  q=p[p.group.eq(g)]
  for a,c in ARM.items():
   for _,z in q[q.arm.eq(a)].groupby('seed'):ax.plot(z.cumulative_episode,z.after_rms,color=c,alpha=.16,lw=.6)
   z=q[q.arm.eq(a)].groupby('cumulative_episode').after_rms.mean();ax.plot(z.index,z,color=c,label=a,lw=1.6)
  for _,z in q[q.arm.eq('P')].groupby('seed'):ax.axhline(z.floor.iloc[0],color=ARM['P'],ls=':',lw=.65,alpha=.35)
  ax.axvline(1920,color='.5',ls='--',lw=.7);ax.set(title=g.replace('_',' '),xlabel='Cumulative meta episodes',ylabel='Group RMS');ax.legend(frameon=False)
 ax=axs.flat[-1]
 for a,c in ARM.items():
  q=t[t.arm.eq(a)]
  for _,z in q.groupby('seed'):ax.plot(z.cumulative_episode,z.recurrent_spectral_norm,color=c,alpha=.14,lw=.6)
  z=q.groupby('cumulative_episode').recurrent_spectral_norm.mean();ax.plot(z.index,z,color=c,label=a)
 ax.axvline(1920,color='.5',ls='--',lw=.7);ax.axhline(.9,color='.6',ls=':',lw=.6);ax.set(title='Recurrent spectral norm (unprojected)',xlabel='Reptile → FOMAML at1920',ylabel='Spectral norm');ax.legend(frameon=False);fig.text(.5,.015,'Faint lines: all six paired seeds. Dotted floors are seed/group-specific. Recurrence and biases are never projected.',ha='center');save(fig,out,'figure_1_weight_scale')
 old=read('historical_scale_trajectories.csv');fig,axs=plt.subplots(1,2,figsize=(12,4))
 for phase,ls in [('Reptile','-'),('FOMAML','--')]:
  for g in ['token','context_A','context_B','context_C','output']:
   q=old[old.phase.eq(phase)].groupby('episode')[g].mean();axs[0].plot(q.index,q,ls=ls,label=phase+' '+g)
  q=old[old.phase.eq(phase)].groupby('episode').recurrent_spectral_norm.mean();axs[1].plot(q.index,q,ls=ls,label=phase)
 axs[0].set(xlabel='Historical within-phase episode',ylabel='Actual trained RMS',title='Archived20-seed scale audit');axs[1].set(xlabel='Historical within-phase episode',ylabel='Recurrent spectral norm');axs[0].legend(fontsize=6,ncol=2);axs[1].legend();save(fig,out,'supplement_historical_scale')

def practiced(out,pre=False):
 f=read('preflight_training.csv' if pre else 'meta_training.csv');ci=read('preflight_training_intervals.csv' if pre else 'meta_training_intervals.csv')
 for phase in (['Initial'] if pre else ['R','F']):
  for stop,suffix in [(330,''),(100,'_first100'),(20,'_first20')]:
   fig,axs=plt.subplots(1 if pre else 2,2 if pre else 3,figsize=(12,4) if pre else (16,7),squeeze=False);fig.suptitle(('Mandatory preflight' if pre else ('Reptile' if phase=='R' else 'FOMAML'))+': actual A10/B10 pre-update CE')
   cases=[('High',0),('Low',0)] if pre else [(a,d) for a in ['U','P'] for d in [240,960,1920]]
   for ax,(a,d) in zip(axs.flat,cases):
    q=f[f.arm.eq(a)] if pre else pick(f,a,phase,d);s=ci[ci.arm.eq(a)] if pre else pick(ci,a,phase,d);q=q[q['update'].le(stop)];s=s[s['update'].le(stop)]
    for _,v in q.groupby('seed'):ax.plot(v['update'],v.ce_before,color='.5',alpha=.1,lw=.5)
    ax.plot(s['update'],s['mean'],color='.5',lw=.65)
    for start in range(1,stop+1,10):
     c='A' if ((start-1)//10)%2==0 else 'B';band(ax,q[q['update'].between(start,start+9)],s[s['update'].between(start,start+9)],'update','ce_before',COL[c],c if start<12 else None)
    ax.set(title=a if pre else f'Arm {a} | +{d}',xlabel='Target repetition',ylabel='Practiced CE (nats)');ax.legend(frameon=False)
   fig.text(.5,.012,'All six seeds; mean and95% whole-seed intervals. Switch jumps preserved; no target projection or smoothing.',ha='center');save(fig,out,('preflight_learning' if pre else 'figure_2_'+phase)+suffix)

def frozen(out):
 f=read('meta_evaluations.csv');ci=read('meta_evaluations_intervals.csv')
 for phase in ['R','F']:
  for stop,suffix in [(330,''),(120,'_first120')]:
   fig,axs=plt.subplots(2,3,figsize=(16,8));fig.suptitle(('Reptile' if phase=='R' else 'FOMAML')+': frozen six-cue sequence CE')
   for ax,(a,d) in zip(axs.flat,[(a,d) for a in ['U','P'] for d in [240,960,1920]]):
    q=pick(f,a,phase,d);s=pick(ci,a,phase,d)
    for c,col in COL.items():band(ax,q[q.cue.eq(c)&q['update'].le(stop)],s[s.cue.eq(c)&s['update'].le(stop)],'update','ce',col,c)
    ax.axhline(np.log(4),color='.5',ls=':',lw=.7);ax.set(title=f'Arm {a} | +{d}',xlabel='Frozen target state',ylabel='CE (nats)');ax.legend(fontsize=6,ncol=2,frameon=False)
   fig.text(.5,.012,'Six paired seeds; no C/Mixed target practice. Dotted ln(4). Mixed-average and Mixed-both are different physical cues.',ha='center');save(fig,out,'figure_3_'+phase+suffix)

def functional(out):
 f=read('meta_seed_outcomes.csv');summary=read('meta_outcome_summary.csv');sw=read('meta_switch_costs.csv');sc=read('meta_switch_intervals.csv');matched=read('meta_matched_acquisition.csv');ms=read('meta_matched_summary.csv')
 for phase in ['R','F']:
  fig,axs=plt.subplots(2,3,figsize=(15,7));fig.suptitle(phase+': first-A/B own, cross and signed specificity')
  for i,c in enumerate(['A','B']):
   for j,prefix in enumerate(['own_','cross_','D_']):
    ax=axs[i,j];metric=prefix+c
    for a,off in [('U',-.12),('P',.12)]:
     for x,d in enumerate([240,960,1920]):
      q=pick(f,a,phase,d);s=pick(summary,a,phase,d);s=s[s.metric.eq(metric)].iloc[0];ax.scatter(x+off+np.linspace(-.035,.035,6),q[metric],color=ARM[a],alpha=.4,s=12);ax.errorbar(x+off,s['mean'],yerr=[[max(0,s['mean']-s.ci_low)],[max(0,s.ci_high-s['mean'])]],fmt='o',color=ARM[a],capsize=3,label=a if x==0 else None)
    ax.axhline(0,color='.6',lw=.6);ax.set(xticks=[0,1,2],xticklabels=[240,960,1920],xlabel='Meta dose',title=c+' '+{'own_':'own-context gain','cross_':'cross-context gain','D_':'signed specificity'}[prefix],ylabel='CE gain / own−cross (nats)');ax.legend(frameon=False)
  fig.text(.5,.012,'Signed effects and all seeds retained. Starting losses/floors and missing matched bins remain in exact tables.',ha='center');save(fig,out,'figure_2_'+phase+'_functional')
  for metric in ['raw_cost','adjusted_cost']:
   fig,axs=plt.subplots(2,3,figsize=(16,7));fig.suptitle(phase+' same-weight directional '+metric.replace('_',' '))
   for ax,(a,d) in zip(axs.flat,[(a,d) for a in ['U','P'] for d in [240,960,1920]]):
    q=pick(sw,a,phase,d);s=pick(sc,a,phase,d)
    for direction,col in [('A_to_B',COL['A']),('B_to_A',COL['B'])]:band(ax,q[q.direction.eq(direction)],s[s.direction.eq(direction)],'boundary',metric,col,direction)
    ax.axhline(0,color='.5',lw=.6);ax.set(title=f'{a} | +{d}',xlabel='Target boundary',ylabel='Signed CE difference');ax.legend(fontsize=7,frameon=False)
   save(fig,out,'figure_2_'+phase+'_'+metric)
  for stratum in ['all','A','B']:
   fig,axs=plt.subplots(3,4,figsize=(16,10));fig.suptitle(phase+' fixed matched-acquisition bins: '+stratum)
   for i,d in enumerate([240,960,1920]):
    for j,b in enumerate(['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5']):
     ax=axs[i,j];ns=[]
     for x,a in enumerate(['U','P']):
      q=pick(matched,a,phase,d);q=q[q.stratum.eq(stratum)&q.performance_range.eq(b)];s=pick(ms,a,phase,d);s=s[s.stratum.eq(stratum)&s.performance_range.eq(b)&s.metric.eq('specificity')].iloc[0];ns.append(str(int(q.specificity.notna().sum())));ax.scatter(x+np.linspace(-.04,.04,6),q.specificity,color=ARM[a],alpha=.4,s=12)
      if np.isfinite(s['mean']):ax.errorbar(x,s['mean'],yerr=[[max(0,s['mean']-s.ci_low)],[max(0,s.ci_high-s['mean'])]],fmt='o',color=ARM[a],capsize=3)
     ax.axhline(0,color='.5',lw=.6);ax.set(xticks=[0,1],xticklabels=['U','P'],title=f'+{d} | {b} | n='+','.join(ns),ylabel='Signed specificity')
   fig.text(.5,.012,'First eligible block per seed; unavailable bins remain missing. This conditional comparison is not causal mediation.',ha='center');save(fig,out,'supplement_matched_'+phase+'_'+stratum)

def joint(out):
 f=read('meta_seed_outcomes.csv');fig,axs=plt.subplots(2,3,figsize=(16,8));fig.suptitle('Joint phenotype at every prescribed checkpoint; no checkpoint selection');norm=Normalize(0,1);cmap=plt.get_cmap('viridis')
 for ax,(phase,d) in zip(axs.flat,[(p,d) for p in ['R','F'] for d in [240,960,1920]]):
  for a,marker,dy in [('U','o',5),('P','^',-9)]:
   q=pick(f,a,phase,d);y=(q.D_A+q.D_B)/2;ax.scatter(q.gap,y,c=q.conflict_accuracy,cmap=cmap,norm=norm,marker=marker,s=40,alpha=.8,label=a)
   for r in q.itertuples():ax.annotate(str(r.seed)+('*' if r.strong_familiar else '')+('+' if r.Mixed_both_success else ''),(r.gap,(r.D_A+r.D_B)/2),xytext=(3,dy),textcoords='offset points',fontsize=7)
  ax.axvline(0,color='.6',lw=.6);ax.axhline(0,color='.6',lw=.6);ax.set(title=f'{phase} | +{d}',xlabel='C gap (nats)',ylabel='Mean first-A/B signed specificity');ax.legend(frameon=False)
 fig.text(.5,.014,'Seed IDs; * familiar-qualified; + Mixed-both success. Color: reserved conflict-position accuracy (0–1). Separate D_A/D_B determine comparisons.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,.92,.94));cax=fig.add_axes([.94,.20,.012,.60]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Conflict accuracy');fig.savefig(out/'figure_4_joint.png',metadata={'Software':'high_scale_constrained_meta_v1'});fig.savefig(out/'figure_4_joint.svg',metadata={'Date':None,'Creator':'high_scale_constrained_meta_v1'});plt.close(fig)

def mechanism(out):
 f=read('meta_seed_outcomes.csv');g=read('meta_geometry.csv');q=g[g['update'].eq(0)].merge(f[['seed','arm','dose','D_A','D_B','ce','NoContext_ce']],on=['seed','arm','dose'],validate='one_to_one');fig,axs=plt.subplots(1,3,figsize=(16,5));fig.suptitle('Raw context amplitude versus effective contextual computation')
 for phase,c in [('R','#327aad'),('F','#a35d3b')]:
  for arm,marker in [('U','o'),('P','^')]:
   z=q[q.phase.eq(phase)&q.condition.eq(arm)];label=phase+' '+arm;axs[0].scatter((z.A_norm+z.B_norm)/(2*np.sqrt(8)),z.centered_logit_separation,color=c,marker=marker,alpha=.55,label=label);axs[1].scatter(z.centered_logit_separation,(z.D_A+z.D_B)/2,color=c,marker=marker,alpha=.55,label=label);axs[2].scatter(z.NoContext_ce,z.ce,color=c,marker=marker,alpha=.55,label=label)
 axs[0].set(xlabel='Mean A/B column RMS at target0',ylabel='Mean ||center(K_t(wA−wB))|| at target0');axs[1].set(xlabel='Effective centered contrast at target0',ylabel='First-A/B mean signed specificity');axs[2].set(xlabel='NoContext CE at target330',ylabel='C CE at target330')
 for ax in axs:ax.legend(frameon=False,fontsize=7)
 fig.text(.5,.012,'Every fixed phase/dose/seed; no correlation implies mediation. All seven raw/centered components, K norms, margins and baselines are archived.',ha='center',fontsize=9);save(fig,out,'figure_5_effective_computation')

def supplements(out):
 f=read('meta_evaluations.csv');ci=read('meta_evaluations_intervals.csv')
 for phase in ['R','F']:
  fig,axs=plt.subplots(1,3,figsize=(15,4));fig.suptitle(phase+': signed C generalization gap throughout target learning')
  for ax,d in zip(axs,[240,960,1920]):
   for a,col in ARM.items():
    q=pick(f,a,phase,d);s=pick(ci,a,phase,d);band(ax,q[q.cue.eq('C')],s[s.cue.eq('C')],'update','gap',col,a)
   ax.axhline(0,color='.5',lw=.7);ax.set(title=f'+{d}',xlabel='Frozen target update',ylabel='C CE − familiar mean CE');ax.legend(frameon=False)
  fig.text(.5,.012,'Absolute C/familiar losses are shown separately in Figure3 and endpoint tables; negative gaps are preserved.',ha='center');save(fig,out,'supplement_C_gap_'+phase)
 raw=read('preflight_seed_outcomes.csv');summary=read('preflight_outcome_summary.csv');fig,axs=plt.subplots(1,3,figsize=(13,4));fig.suptitle('Preregistered High/Low preflight: functional learning, not weight norms')
 for ax,m in zip(axs,['D_A','D_B','cycle']):
  p=raw.pivot(index='seed',columns='arm',values=m)
  for _,r in p.iterrows():ax.plot([0,1],[r.High,r.Low],color='.6',alpha=.35,lw=.8)
  for x,a,col in [(0,'High','#a64d35'),(1,'Low','#557fa5')]:
   q=raw[raw.arm.eq(a)];ss=summary[summary.arm.eq(a)&summary.metric.eq(m)].iloc[0];ax.scatter(np.full(6,x),q[m],color=col,alpha=.5);ax.errorbar(x,ss['mean'],yerr=[[max(0,ss['mean']-ss.ci_low)],[max(0,ss.ci_high-ss['mean'])]],fmt='o',color=col,capsize=4)
  ax.axhline(0,color='.6',lw=.6);ax.set(xticks=[0,1],xticklabels=['High','Low'],title=m,ylabel='Signed nats')
 fig.text(.5,.012,'Six paired seeds, fixed firstA/firstB/first complete cycle; no meta-training before the frozen gate passed.',ha='center');save(fig,out,'preflight_functional_gate')

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args();out=destination(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 for fn in [scales,practiced,frozen,functional,joint,mechanism]:fn(out)
 practiced(out,pre=True);supplements(out)
if __name__=='__main__':main()
