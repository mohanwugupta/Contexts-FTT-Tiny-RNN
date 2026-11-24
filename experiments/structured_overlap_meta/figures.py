"""Deterministic scientific figures from saved numeric tables only."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/structured_overlap_meta/v1'
CC={'A':'#2166ac','B':'#ed8b23','C':'#ce3451','Mixed-average':'#27934c','Mixed-both':'#8a5bb4','NoContext':'#777777'}
plt.rcParams.update({'svg.hashsalt':'structured_overlap_meta_v1','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':140})
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def save(fig,out,n):
 fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(out/f'{n}.png',metadata={'Software':'structured_overlap_meta_v1'});fig.savefig(out/f'{n}.svg',metadata={'Date':None,'Creator':'structured_overlap_meta_v1'});plt.close(fig)
def band(ax,raw,summary,t,m,col,label):
 for _,g in raw.groupby('seed'):ax.plot(g[t],g[m],color=col,alpha=.13,lw=.6)
 q=summary[summary.metric.eq(m)].sort_values(t);ax.plot(q[t],q['mean'],color=col,lw=1.6,label=label);ax.fill_between(q[t],q.ci_low,q.ci_high,color=col,alpha=.16,lw=0)
def pick(f,a,d):return f[f.arm.eq(a)&f.dose.eq(d)]
def place_labels(ax,labels):
 # Display-only deterministic spacing; plotted values never move.
 xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim();used=[]
 for label,x,y in sorted(labels,key=lambda v:(v[2],v[1],v[0])):
  xx=(x-xmin)/(xmax-xmin);yy=(y-ymin)/(ymax-ymin)
  if any(label==v[0] and abs(xx-v[1])<.02 and abs(yy-v[2])<.02 for v in used):continue
  candidates=sorted([(xx+dx,yy+dy) for dx in [0,.03,-.03,.06,-.06,.09,-.09] for dy in np.arange(-.30,.331,.033)],key=lambda p:(p[0]-xx)**2+(p[1]-yy)**2)
  for tx,ty in candidates:
   if not (.01<=tx<=.93 and .025<=ty<=.97):continue
   if all(abs(tx-v[3])>=.067 or abs(ty-v[4])>=.032 for v in used):break
  else:tx,ty=xx,yy
  ax.annotate(label,(x,y),xytext=(tx,ty),textcoords='axes fraction',fontsize=6,arrowprops=dict(arrowstyle='-',color='.65',lw=.4),ha='left',va='center');used.append((label,xx,yy,tx,ty))

def generate(stage,out):
 f=read(stage+'_evaluations');t=read(stage+'_training');tc=read(stage+'_training_intervals');fc=read(stage+'_evaluations_intervals');end=read(stage+'_seed_outcomes');arms=[a for a in end.arm.unique() if a!='Baseline'];doses=sorted(end[end.arm.ne('Baseline')].dose.unique());cases=[(a,d) for d in doses for a in arms];n=len(cases);cols=len(arms);rows=len(doses);unit='Exploratory six-seed screen' if stage=='screen' else 'Untouched 14-seed confirmation';prefix='' if stage=='confirm' else 'screen_'
 for stop,suffix in [(330,''),(100,'_first100'),(20,'_first20')]:
  fig,axs=plt.subplots(rows,cols,figsize=(5.2*cols,3.6*rows),squeeze=False);fig.suptitle(unit+': actual pre-update A10/B10 CE')
  for ax,(a,d) in zip(axs.flat,cases):
   x=pick(t,a,d);x=x[x['update'].le(stop)];ci=pick(tc,a,d);ci=ci[ci['update'].le(stop)]
   for _,g in x.groupby('seed'):ax.plot(g['update'],g.ce_before,color='.5',alpha=.1,lw=.5)
   ax.plot(ci['update'],ci['mean'],color='.5',lw=.6)
   for start in range(1,stop+1,10):
    c='A' if ((start-1)//10)%2==0 else 'B';xx=x[x['update'].between(start,start+9)];ss=ci[ci['update'].between(start,start+9)];band(ax,xx,ss,'update','ce_before',CC[c],c if start<12 else None)
   ax.set(title=f'Arm {a} | +{d} episodes',xlabel='Target repetition',ylabel='Practiced CE (nats)');ax.legend(frameon=False)
  fig.text(.5,.01,'Faint individual seeds; mean and 95% seed-bootstrap intervals. Switch jumps retained; no smoothing.',ha='center');save(fig,out,prefix+'figure_1'+suffix)
 for stop,suffix in [(330,''),(120,'_first120')]:
  fig,axs=plt.subplots(rows,cols,figsize=(5.2*cols,3.9*rows),squeeze=False);fig.suptitle(unit+': frozen six-cue sequence CE')
  for ax,(a,d) in zip(axs.flat,cases):
   for c,col in CC.items():
    x=pick(f,a,d);ci=pick(fc,a,d);band(ax,x[x.cue.eq(c)&x['update'].le(stop)],ci[ci.cue.eq(c)&ci['update'].le(stop)],'update','ce',col,c)
   ax.axhline(np.log(4),color='.5',ls=':',lw=.8);ax.set(title=f'Arm {a} | +{d}',xlabel='Frozen target update',ylabel='CE (nats)');ax.legend(frameon=False,ncol=2,fontsize=7)
  fig.text(.5,.01,'Identical weights at each checkpoint; no target C/Mixed practice. Dotted chance ln(4).',ha='center');save(fig,out,prefix+'figure_2'+suffix)
 fig,axs=plt.subplots(2,len(doses),figsize=(6*len(doses),8),squeeze=False);fig.suptitle(unit+': functional learning and transfer')
 colors=dict(zip(arms,['#286caa','#c67a29','#637d44']))
 for j,d in enumerate(doses):
  labels={0:[],1:[]}
  for a in arms:
   q=pick(end,a,d)
   for ax,y in [(axs[0,j],'early_specificity'),(axs[1,j],'conflict_accuracy')]:
    ax.scatter(q.gap,q[y],color=colors[a],label='Arm '+a,alpha=.6)
    for r in q.itertuples():
     label=f'{r.seed}'+('*' if r.strong_familiar else '')+('+' if r.Mixed_both_success else '');labels[0 if y=='early_specificity' else 1].append((label,r.gap,getattr(r,y)))
  for i,ax in enumerate(axs[:,j]):place_labels(ax,labels[i])
  for ax in axs[:,j]:ax.axvline(0,color='.6',lw=.6);ax.set(xlabel='C CE − mean familiar CE (nats)',title=f'+{d} episodes');ax.legend(frameon=False)
  axs[0,j].set(ylabel='Mean first-A/B signed specificity (nats)');axs[0,j].axhline(0,color='.6',lw=.6);axs[1,j].set(ylabel='Reserved conflict-position accuracy')
 fig.text(.5,.01,'Labels are seed IDs; * strong familiar learning; + Mixed-both success. Geometry is not a learning criterion.',ha='center');save(fig,out,prefix+'figure_3')
 blocks=read(stage+'_blocks');bc=read(stage+'_blocks_intervals');sw=read(stage+'_switch_costs');sc=read(stage+'_switch_intervals');diag=read(stage+'_learning_effects');dc=read(stage+'_diagnostic_intervals')
 for name,raw,ci,time,metrics in [('own_cross',blocks,bc,'end',['practiced_gain','unpracticed_gain','specificity']),('switch',sw,sc,'boundary',['raw_cost','adjusted_cost']),('restored_adam',diag,dc,'update',['D_A','D_B'])]:
  fig,axs=plt.subplots(rows,cols,figsize=(5.2*cols,3.8*rows),squeeze=False);fig.suptitle(unit+': '+name.replace('_',' '))
  for ax,(a,d) in zip(axs.flat,cases):
   for m,col in zip(metrics,['#2166ac','#ed8b23','#444444']):
    rr=pick(raw,a,d);ss=pick(ci,a,d)
    # Never join different block types or switch directions at one x value.
    grp='practiced' if name=='own_cross' else 'direction' if name=='switch' else None
    for level in rr[grp].unique() if grp else [None]:
     r=rr[rr[grp].eq(level)] if grp else rr;s=ss[ss[grp].eq(level)] if grp else ss;band(ax,r,s,time,m,col,m+(' '+str(level) if level else ''))
   ax.axhline(0,color='.6',lw=.6);ax.set(title=f'Arm {a} | +{d}',xlabel=time,ylabel='Signed CE change (nats)');ax.legend(fontsize=6,frameon=False)
  save(fig,out,prefix+'supplement_'+name)
 gm=read(stage+'_geometry');fig,axs=plt.subplots(rows,cols,figsize=(5.2*cols,3.4*rows),squeeze=False)
 names=[c for c in gm if 'separation' in c and 'normalized' not in c][:2]
 for ax,(a,d) in zip(axs.flat,cases):
  for m,col in zip(names,['#4776ac','#b95847']):
   q=pick(gm,a,d).groupby('update')[m].mean();ax.plot(q.index,q,color=col,label=m)
  ax.set(title=f'Arm {a} | +{d}',xlabel='Target update',ylabel='Descriptive separation');ax.legend(fontsize=6)
 fig.suptitle('Representational geometry does not establish separate learning');save(fig,out,prefix+'supplement_geometry')

def supplemental(stage,out):
 f=read(stage+'_seed_outcomes');arms=[a for a in f.arm.unique() if a!='Baseline'];doses=sorted(f[f.arm.ne('Baseline')].dose.unique());colors=dict(zip(arms,['#286caa','#c67a29','#637d44']));raw=read(stage+'_matched_acquisition');summary=read(stage+'_matched_summary');bins=['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5'];prefix='' if stage=='confirm' else 'screen_'
 for stratum in ['all','A','B']:
  fig,axs=plt.subplots(len(doses),4,figsize=(16,3.5*len(doses)),squeeze=False);fig.suptitle('Fixed matched-acquisition bins: '+stratum+' starting blocks')
  for i,d in enumerate(doses):
   for j,b in enumerate(bins):
    ax=axs[i,j];counts=[]
    for x,a in enumerate(arms):
     q=pick(raw,a,d);q=q[q.stratum.eq(stratum)&q.performance_range.eq(b)];v=q.specificity.astype(float);counts.append(str(int(v.notna().sum())));ax.scatter(x+np.linspace(-.07,.07,len(v)),v,color=colors[a],alpha=.35,s=12);ss=pick(summary,a,d);ss=ss[ss.stratum.eq(stratum)&ss.performance_range.eq(b)&ss.metric.eq('specificity')].iloc[0]
     if np.isfinite(ss['mean']):ax.errorbar(x,ss['mean'],yerr=[[max(0,ss['mean']-ss.ci_low)],[max(0,ss.ci_high-ss['mean'])]],fmt='o',color=colors[a],capsize=3)
    ax.set(xticks=range(len(arms)),xticklabels=arms,title=f'+{d} | {b} | n='+','.join(counts),ylabel='Signed own−cross CE gain');ax.axhline(0,color='.6',lw=.6)
  fig.text(.5,.01,'First eligible block per trained seed; unavailable bins remain missing, never zero. Conditional descriptive comparison.',ha='center');save(fig,out,prefix+'supplement_matched_'+stratum)
 gm=read(stage+'_geometry');metrics=['u_norm','v_norm','C_response_norm','C_minus_avg_centered_norm'];fig,axs=plt.subplots(len(doses),len(arms),figsize=(5.5*len(arms),3.8*len(doses)),squeeze=False);fig.suptitle('Exact Linear decomposition: centered four-logit magnitudes')
 for ax,(a,d) in zip(axs.flat,[(a,d) for d in doses for a in arms]):
  for m,col in zip(metrics,['#2e7b9b','#bb6a30','#82953f','#9c6298']):
   q=pick(gm,a,d).groupby('update')[m].mean();ax.plot(q.index,q,color=col,label=m)
  ax.set(title=f'Arm {a} | +{d}',xlabel='Target update',ylabel='Mean centered norm');ax.legend(fontsize=7,frameon=False)
 fig.text(.5,.01,'Norms describe components; they do not establish causal CE contributions. Full four-class/position identities are archived.',ha='center');save(fig,out,prefix+'supplement_decomposition')

def capacity(out):
 f=read('pilot_cases');fig,axs=plt.subplots(1,2,figsize=(11,4))
 for k,col in zip([0,1,2],['#699b68','#c18b39','#557fa2']):
  q=f[f.n_exceptions.eq(k)];axs[0].scatter(q.case_index if 'case_index' in q else np.arange(len(q)),q.first_pass,color=col,label=f'{k} exceptions',alpha=.6)
  axs[1].bar(k,q.reached.mean(),color=col)
 axs[0].set(xlabel='Registered case within type',ylabel='First simultaneous criterion update');axs[0].legend();axs[1].axhline(.8,color='.5',ls=':');axs[1].set(xticks=[0,1,2],xlabel='Exceptions',ylabel='Proportion reaching criterion',ylim=(0,1.08));fig.suptitle('Capacity pilot: 12 tasks/type × 2 fixed initializations');save(fig,out,'capacity_pilot')

def safe_destination(value):
 out=Path(value) if value is not None else OUT/'figures'
 if (OUT/'manifest.json').exists() and out.resolve().is_relative_to(OUT.resolve()):raise ValueError('Sealed figures require external output directory')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm','capacity'],required=True);p.add_argument('--output-dir',type=Path);a=p.parse_args();out=safe_destination(a.output_dir)
 if (OUT/'manifest.json').exists() and out.resolve().is_relative_to(OUT.resolve()):raise ValueError('Sealed figures require external output directory')
 out.mkdir(parents=True,exist_ok=True)
 if a.stage=='capacity':capacity(out)
 else:
  generate(a.stage,out);supplemental(a.stage,out)
if __name__=='__main__':main()
