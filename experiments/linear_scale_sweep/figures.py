"""Saved-table-only six figure families; raw coordinates preserved."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/linear_scale_sweep/v1';SCALES=[1,2,3,4,6,8,12,16];COL={'A':'#2166ac','B':'#ed8b23','C':'#ce3451','Mixed-average':'#27934c','Mixed-both':'#8a5bb4','NoContext':'#777777'}
plt.rcParams.update({'svg.hashsalt':'linear_scale_sweep_v1','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':130})
def read(n):return pd.read_csv(OUT/f'tables/{n}.csv',float_precision='round_trip',low_memory=False)
def save(fig,d,name):
 fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(d/f'{name}.png',metadata={'Software':'linear_scale_sweep_v1'});fig.savefig(d/f'{name}.svg',metadata={'Date':None,'Creator':'linear_scale_sweep_v1'});plt.close(fig)
def destination(p):
 p=Path(p) if p is not None else OUT/'figures'
 if (OUT/'manifest.json').exists() and p.resolve().is_relative_to(OUT.resolve()):raise ValueError('Sealed output requires external destination')
 return p

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','replicate'],default='screen');p.add_argument('--output-dir',type=Path);args=p.parse_args();stage=args.stage;dest=destination(args.output_dir);dest.mkdir(parents=True,exist_ok=True);o=read(stage+'_outcomes');su=read(stage+'_summary');t=read(stage+'_training');tc=read(stage+'_training_intervals');sw=read(stage+'_switch_costs');sc=read(stage+'_switch_intervals');balanced=read(stage+'_balanced_cycles');bc=read(stage+'_balanced_cycle_intervals');sources=o.arm.unique();scales=sorted(o.dose.unique());prefix=stage+'_'
 fig,axs=plt.subplots(1,len(sources),figsize=(6*len(sources),4),squeeze=False);fig.suptitle('First three cycles: signed response to nominal scale')
 for ax,src in zip(axs.flat,sources):
  for m,c in [('early_cost','#333333'),('A_to_B',COL['A']),('B_to_A',COL['B'])]:
   q=o[o.arm.eq(src)]
   for _,g in q.groupby('seed'):ax.plot(g.dose,g[m],color=c,alpha=.13,lw=.6)
   q=su[su.arm.eq(src)&su.metric.eq(m)].sort_values('dose');ax.plot(q.dose,q['mean'],color=c,label=m);ax.fill_between(q.dose,q.ci_low,q.ci_high,color=c,alpha=.12)
  ax.axhline(0,color='.6',lw=.7);ax.set(title=src,xlabel='Nominal scale s',ylabel='Adjusted switch cost (nats)');ax.legend(fontsize=7)
 save(fig,dest,prefix+'figure_1_scale_response')
 for src in sources:
  for stop,suffix in [(330,''),(100,'_first100'),(20,'_first20')]:
   fig,axs=plt.subplots((len(scales)+3)//4,min(4,len(scales)),figsize=(4.5*min(4,len(scales)),4*((len(scales)+3)//4)),squeeze=False);fig.suptitle(src+': actual practiced pre-update A10/B10 loss')
   q=t[t.arm.eq(src)&t['update'].le(stop)];ci=tc[tc.arm.eq(src)&tc['update'].le(stop)];common=max(1,float(q.ce_before.quantile(1)))
   for ax,s in zip(axs.flat,scales):
    a=q[q.dose.eq(s)];b=ci[ci.dose.eq(s)]
    for _,g in a.groupby('seed'):ax.plot(g['update'],g.ce_before,color='.6',alpha=.1,lw=.5)
    ax.plot(b['update'],b['mean'],color='.6',lw=.5)
    for start in range(1,stop+1,10):
     c='A' if ((start-1)//10)%2==0 else 'B';aa=a[a['update'].between(start,start+9)];bb=b[b['update'].between(start,start+9)]
     for _,g in aa.groupby('seed'):ax.plot(g['update'],g.ce_before,color=COL[c],alpha=.16,lw=.6)
     ax.plot(bb['update'],bb['mean'],color=COL[c]);ax.fill_between(bb['update'],bb.ci_low,bb.ci_high,color=COL[c],alpha=.12)
    ax.set(title=f's={s}',xlabel='Target repetition',ylabel='Practiced CE (nats)',ylim=(-.02,common*1.03))
   for ax in list(axs.flat)[len(scales):]:ax.set_visible(False)
   fig.text(.5,.01,'Common y range includes every seed. Separate close-up below preserves smaller effects; no smoothing across switches.',ha='center');save(fig,dest,prefix+'figure_2_'+src+suffix)
  # A second view retains all data via a labeled symlog close-up.
  fig,axs=plt.subplots((len(scales)+3)//4,min(4,len(scales)),figsize=(4.5*min(4,len(scales)),4*((len(scales)+3)//4)),squeeze=False);fig.suptitle(src+': first100 actual CE, symlog close-up (linear below0.1)')
  for ax,s in zip(axs.flat,scales):
   a=t[t.arm.eq(src)&t.dose.eq(s)&t['update'].le(100)]
   for start in range(1,101,10):
    c='A' if ((start-1)//10)%2==0 else 'B'
    for _,g in a[a['update'].between(start,start+9)].groupby('seed'):ax.plot(g['update'],g.ce_before,color=COL[c],alpha=.4,lw=.7)
   ax.set_yscale('symlog',linthresh=.1);ax.set(title=f's={s}',xlabel='Target repetition',ylabel='CE (symlog nats)')
  for ax in list(axs.flat)[len(scales):]:ax.set_visible(False)
  save(fig,dest,prefix+'figure_2_'+src+'_closeup')
  fig,axs=plt.subplots(2,3,figsize=(15,7));fig.suptitle(src+': first-block own/cross learning (signed)')
  for i,c in enumerate(['A','B']):
   for j,metric in enumerate(['own_'+c,'cross_'+c,'D_'+c]):
    ax=axs[i,j];a=o[o.arm.eq(src)]
    for _,g in a.groupby('seed'):ax.plot(g.dose,g[metric],color=COL[c],alpha=.2,lw=.6)
    if metric in set(su.metric):
     b=su[su.arm.eq(src)&su.metric.eq(metric)];ax.plot(b.dose,b['mean'],color=COL[c]);ax.fill_between(b.dose,b.ci_low,b.ci_high,color=COL[c],alpha=.15)
    ax.axhline(0,color='.5',lw=.6);ax.set(title=metric,xlabel='Nominal scale',ylabel='CE gain / own−cross')
  save(fig,dest,prefix+'figure_3_'+src)
  fig,axs=plt.subplots((len(scales)+3)//4,min(4,len(scales)),figsize=(4.5*min(4,len(scales)),4*((len(scales)+3)//4)),squeeze=False);fig.suptitle(src+': directional switch-cost persistence')
  for ax,s in zip(axs.flat,scales):
   for direction,c in [('A_to_B',COL['A']),('B_to_A',COL['B'])]:
    a=sw[sw.arm.eq(src)&sw.dose.eq(s)&sw.direction.eq(direction)];b=sc[sc.arm.eq(src)&sc.dose.eq(s)&sc.direction.eq(direction)&sc.metric.eq('adjusted_cost')]
    for _,g in a.groupby('seed'):ax.plot(g.boundary,g.adjusted_cost,color=c,alpha=.14,lw=.6)
    ax.plot(b.boundary,b['mean'],color=c,label=direction);ax.fill_between(b.boundary,b.ci_low,b.ci_high,color=c,alpha=.12)
   bb=balanced[balanced.arm.eq(src)&balanced.dose.eq(s)];cc=bc[bc.arm.eq(src)&bc.dose.eq(s)]
   for _,gg in bb.groupby('seed'):ax.plot(gg.boundary,gg.balanced_cost,color='#333333',alpha=.15,lw=.6)
   ax.plot(cc.boundary,cc['mean'],color='#333333',label='balanced cycle');ax.fill_between(cc.boundary,cc.ci_low,cc.ci_high,color='#333333',alpha=.1)
   ax.axvspan(10,60,color='.8',alpha=.1);ax.axhline(0,color='.5',lw=.6);ax.set(title=f's={s}',xlabel='Target boundary',ylabel='Signed adjusted cost');ax.legend(fontsize=6)
  for ax in list(axs.flat)[len(scales):]:ax.set_visible(False)
  save(fig,dest,prefix+'figure_4_'+src)
  m=read(stage+'_matched_pairs');m=m[m.arm.eq(src)&m.stratum.eq('all')];fig,axs=plt.subplots(1,4,figsize=(17,4));fig.suptitle(src+': matched specificity versus scale2 (paired eligible seeds)')
  for ax,b in zip(axs,['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5']):
   g=m[m.performance_range.eq(b)].sort_values('dose');ax.errorbar(g.dose,g['mean'],yerr=[g['mean']-g.ci_low,g.ci_high-g['mean']],fmt='o-',capsize=3)
   for r in g.itertuples():ax.annotate('n='+str(r.n),(r.dose,r.mean),fontsize=6)
   ax.axhline(0,color='.6',lw=.6);ax.set(title=b,xlabel='Scale',ylabel='Signed candidate−scale2 specificity')
  fig.text(.5,.01,'Missing bins remain missing; conditional matching is not causal mediation. n=1/2 is not population evidence.',ha='center');save(fig,dest,prefix+'figure_3_'+src+'_matched')
 if stage=='screen':
  fig,axs=plt.subplots(1,2,figsize=(13,5));fig.suptitle('Post-meta scaling: splitting versus C transfer')
  for ax,src in zip(axs,['U','P']):
   for s,c in zip(scales,plt.cm.viridis(np.linspace(0,1,len(scales)))):
    a=o[o.arm.eq(src)&o.dose.eq(s)];ax.scatter(a.gap,a.early_cost,color=c,label=str(s),marker='o' if s<=2 else '^')
    for r in a.itertuples():ax.annotate(str(r.seed)+('*' if r.strong_familiar==1 else '')+('+' if r.Mixed_both_success==1 else ''),(r.gap,r.early_cost),fontsize=5,xytext=(2,2),textcoords='offset points')
   ax.set_xscale('symlog',linthresh=.1);ax.set_yscale('symlog',linthresh=.1);ax.set(title=src,xlabel='C gap (symlog nats)',ylabel='Balanced early cost (symlog nats)');ax.legend(title='Scale',fontsize=6,ncol=2)
  fig.text(.5,.01,'Seed IDs; * familiar success; + Mixed-both success. Both axes retain signed values and zero.',ha='center');save(fig,dest,prefix+'figure_5_joint')
 init=pd.read_csv(OUT/f'registry/{stage}_initializations.csv');fig,axs=plt.subplots(1,3,figsize=(16,4));fig.suptitle('Actual magnitudes and effective context computation')
 for src,c in zip(sources,['#416f9a','#a55d37','#6f8d42']):
  a=init[init.source.eq(src)]
  for g in ['token','context_A','context_B','context_C','output']:q=a.groupby('scale')[g].mean();axs[0].plot(q.index,q,color=c,alpha=.6,label=src+' '+g)
  q=a.merge(o[o.arm.eq(src)],left_on=['seed','scale'],right_on=['seed','dose']);axs[1].scatter(q.scale,q.centered_contrast,color=c,alpha=.45,label=src);axs[2].scatter(q.centered_contrast,q.early_cost,color=c,alpha=.45,label=src)
 axs[0].set(xlabel='Nominal scale',ylabel='Actual group RMS');axs[1].set(xlabel='Nominal scale',ylabel='Centered effective A/B logit contrast');axs[2].set(xlabel='Effective centered contrast',ylabel='Balanced early switch cost')
 for ax in axs:ax.legend(fontsize=5,ncol=2)
 save(fig,dest,prefix+'figure_6_effective')
if __name__=='__main__':main()
