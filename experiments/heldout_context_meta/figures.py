"""Table-only primary figures and prespecified full/early/diagnostic companions."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.meta_learning_scaling.figures import ci_line,save,chance
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/heldout_context_meta/v1';CONDS=['P1-I','P1-C','P2-I','P2-C'];PRIMARY=[f'figure_{i}' for i in range(1,6)];AC={'Historical':'#777777','Seen':'#3279b8','Transfer':'#cb334c'};STYLE={'Historical':':','Seen':'--','Transfer':'-'};CC={'A':'#2166ac','B':'#ef8a24','C':'#d03d54','Mixed-average':'#26954b','Mixed-both':'#8e63ad','NoContext':'#777777'}
plt.rcParams.update({'svg.hashsalt':'heldout_context_meta_v1','font.size':8,'savefig.dpi':140})
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def cases(dose):return [('Historical',0),('Seen',dose),('Transfer',dose)]
def pick(f,c,a,d):return f[(f.condition==c)&(f.arm==a)&(f.dose==d)]
def grid(title):
 fig,axs=plt.subplots(3,4,figsize=(17,10),squeeze=False);fig.suptitle(title);return fig,axs

def training(out):
 t=read('training');ci=read('training_intervals')
 for dose,suffix in [(960,''),(240,'_240')]:
  for stop,zoom in [(330,''),(100,'_early')]:
   fig,axs=grid(f'Figure 1 — Actual practiced-context loss before each update (1–{stop})')
   for i,(a,d) in enumerate(cases(dose)):
    for j,c in enumerate(CONDS):
     ax=axs[i,j];x=pick(t,c,a,d);x=x[x['update']<=stop];z=pick(ci,c,a,d);z=z[z['update']<=stop]
     for _,ss in x.groupby('seed'):ax.plot(ss['update'],ss.ce_before,color='.4',alpha=.10,lw=.5)
     ax.plot(z['update'],z['mean'],color='.4',lw=.5)
     for start in range(1,stop+1,10):
      cue='A' if (start-1)%20==0 else 'B';ci_line(ax,x[x['update'].between(start,start+9)],z[z['update'].between(start,start+9)],'update','ce_before',CC[cue],cue if start<=11 else None,style=STYLE[a])
     ax.set(title=f'{c} | {a} +{d}',xlabel='Target update',ylabel='Actual CE (nats)');ax.legend(frameon=False,ncol=2)
   save(fig,out,'figure_1'+suffix+zoom)

def learning_curves(out):
 f=read('evaluations');ci=read('evaluation_intervals')
 for dose,suffix in [(960,''),(240,'_240')]:
  for stop,zoom in [(330,''),(120,'_early')]:
   fig,axs=grid(f'Figure 2 — Frozen natural-cue CE (states0–{stop})')
   for i,(a,d) in enumerate(cases(dose)):
    for j,c in enumerate(CONDS):
     ax=axs[i,j]
     for cue,color in CC.items():
      x=pick(f,c,a,d);x=x[(x.cue==cue)&(x['update']<=stop)];z=pick(ci,c,a,d);z=z[(z.cue==cue)&(z['update']<=stop)];ci_line(ax,x,z,'update','ce',color,cue,style=STYLE[a])
     chance(ax);ax.set(title=f'{c} | {a} +{d}',xlabel='Frozen target state',ylabel='CE (nats)');ax.legend(frameon=False,ncol=3,fontsize=6)
   save(fig,out,'figure_2'+suffix+zoom)

def gaps(out):
 f=read('evaluations');ci=read('evaluation_intervals');post=read('post_acquisition_curve');pci=read('post_acquisition_curve_intervals');diff=read('gap_difference');dci=read('gap_difference_intervals')
 for dose,suffix in [(960,''),(240,'_240')]:
  for source,interval,metric,name,title in [(f[f.cue=='C'],ci[ci.cue=='C'],'gap','figure_3','Figure 3 — C minus mean familiar CE'),(post,pci,'gap_after_acquisition','figure_3_after_acquisition','C gap from first strong familiar acquisition onward')]:
   fig,axs=plt.subplots(2,2,figsize=(13,8));fig.suptitle(title)
   for ax,c in zip(axs.flat,CONDS):
    for a,d in cases(dose):ci_line(ax,pick(source,c,a,d),pick(interval,c,a,d),'update',metric,AC[a],a,style=STYLE[a])
    ax.axhline(0,color='black',lw=.7);ax.set(title=c,xlabel='Frozen target state',ylabel='Signed CE gap (nats)');ax.legend(frameon=False)
   fig.text(.5,.005,'All20 seeds; after-acquisition traces begin at each seed’s first familiar pass. Never-acquired is missing, not zero.',ha='center');save(fig,out,name+suffix)
  fig,axs=plt.subplots(2,2,figsize=(13,8));fig.suptitle('Paired C-gap difference — Transfer minus Seen')
  for ax,c in zip(axs.flat,CONDS):
   x=diff[(diff.condition==c)&(diff.dose==dose)];z=dci[(dci.condition==c)&(dci.dose==dose)];ci_line(ax,x,z,'update','gap_difference',AC['Transfer'],'Transfer − Seen');ax.axhline(0,color='black',lw=.7);ax.set(title=c,xlabel='Frozen target state',ylabel='Paired gap effect (nats)')
  save(fig,out,'figure_3_difference'+suffix)

def positions(out):
 f=read('C_positions');ci=read('C_position_summary')
 for dose,suffix in [(960,''),(240,'_240')]:
  for t in [40,80,120,170,330]:
   fig,axs=plt.subplots(2,4,figsize=(17,7));fig.suptitle(f'Figure 4 — Natural C prediction by position at target{t}')
   for j,c in enumerate(CONDS):
    for i,metric in enumerate(['ce','margin']):
     ax=axs[i,j]
     for a,d in cases(dose):
      x=pick(f,c,a,d);x=x[x['update']==t];z=pick(ci,c,a,d);z=z[z['update']==t];ci_line(ax,x,z,'position',metric,AC[a],a,style=STYLE[a])
     ax.axhline(0,color='.5',lw=.6);ax.set(title=c,xlabel='Sequence prediction position',ylabel='C CE (nats)' if metric=='ce' else 'Correct-class logit margin');ax.legend(frameon=False)
   save(fig,out,('figure_4' if t==330 else f'figure_4_t{t}')+suffix)

def specificity(out):
 b=read('block_transfer');ci=read('block_intervals');ms=read('matched_summary');m=read('matched_acquisition')
 for dose,suffix in [(960,''),(240,'_240')]:
  for metrics,name in [(['specificity'],'figure_5'),(['practiced_gain','unpracticed_gain'],'figure_5_gains')]:
   fig,axs=plt.subplots(2,4,figsize=(17,8));fig.suptitle('Figure 5 — Signed A/B-specific learning')
   for j,c in enumerate(CONDS):
    for i,cue in enumerate(['A','B']):
     ax=axs[i,j]
     for a,d in cases(dose):
      x=pick(b,c,a,d);x=x[x.practiced==cue];z=pick(ci,c,a,d);z=z[z.practiced==cue]
      for metric in metrics:ci_line(ax,x,z,'end',metric,AC[a],a+(' own' if metric=='practiced_gain' else ' cross' if metric=='unpracticed_gain' else ''),style='--' if metric=='unpracticed_gain' else STYLE[a])
      floor=x[x.floor_at_start];ax.scatter(floor.end,np.zeros(len(floor)),s=7,marker='|',color=AC[a],alpha=.1)
     ax.axhline(0,color='black',lw=.6);ax.set(title=c+' | '+cue+' practice',xlabel='End of10-update block',ylabel='CE gain / specificity');ax.legend(frameon=False,ncol=2,fontsize=6)
   fig.text(.5,.005,'Ticks mark both-context starting CE≤0.05. Lack of measurable learning there does not establish shared representations.',ha='center');save(fig,out,name+suffix)
 for stratum,suffix in [('all',''),('A','_A'),('B','_B')]:
  fig,axs=plt.subplots(4,4,figsize=(17,13));fig.suptitle('Matched familiar acquisition — first eligible block, '+stratum)
  for i,perf in enumerate(['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5']):
   for j,c in enumerate(CONDS):
    ax=axs[i,j]
    for k,(a,d) in enumerate(cases(960)):
     x=pick(m,c,a,d);x=x[(x.performance_range==perf)&(x.stratum==stratum)];z=pick(ms,c,a,d);z=z[(z.performance_range==perf)&(z.stratum==stratum)&(z.metric=='specificity')].iloc[0];xx=x.sort_values('seed');jitter=np.linspace(-.15,.15,len(xx));ax.scatter(k+jitter,xx.specificity,color=AC[a],s=12,alpha=.4);ax.errorbar(k,z['mean'],yerr=np.array([[z['mean']-z.ci_low],[z.ci_high-z['mean']]]),fmt='ko',capsize=3);ax.text(k,.97,f'n={z.n}',ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=7)
    ax.axhline(0,color='.5',lw=.6);ax.set_xticks(range(3),['Historical','Seen+960','Transfer+960'],rotation=25);ax.set(title=c+' | '+perf,ylabel='Own − cross CE gain')
  save(fig,out,'figure_5_matched'+suffix)
 s=read('switch_costs');si=read('switch_intervals');fig,axs=plt.subplots(2,4,figsize=(17,8));fig.suptitle('Same-weight switch costs: solid raw, dotted baseline adjusted')
 for j,c in enumerate(CONDS):
  for i,direction in enumerate(['A_to_B','B_to_A']):
   ax=axs[i,j]
   for a,d in cases(960):
    x=pick(s,c,a,d);x=x[x.direction==direction];z=pick(si,c,a,d);z=z[z.direction==direction]
    for metric in ['raw_cost','adjusted_cost']:ci_line(ax,x,z,'boundary',metric,AC[a],a if metric=='raw_cost' else None,style='-' if metric=='raw_cost' else ':')
   ax.axhline(0,color='.5',lw=.6);ax.set(title=c+' | '+direction,xlabel='Target switch boundary',ylabel='Signed cost (nats)');ax.legend(frameon=False)
 save(fig,out,'figure_5_switches');d=read('learning_effects');di=read('diagnostic_intervals');fig,axs=plt.subplots(3,4,figsize=(17,10));fig.suptitle('Restored-Adam one-update effects (clones only)')
 for j,c in enumerate(CONDS):
  for i,metric in enumerate(['mean_own_improvement','mean_cross_improvement','mean_specificity']):
   ax=axs[i,j]
   for a,dd in cases(960):ci_line(ax,pick(d,c,a,dd),pick(di,c,a,dd),'update',metric,AC[a],a,style=STYLE[a])
   ax.axhline(0,color='.5',lw=.6);ax.set(title=c,xlabel='Frozen target state',ylabel=metric.replace('_',' '));ax.legend(frameon=False)
 save(fig,out,'figure_5_restored_adam')

def mechanisms(out):
 g=read('geometry')
 for metric in ['baseline_ce','C_ce','C_minus_avg_centered_norm','C_minus_avg_CE','C_minus_baseline_CE','u_norm','v_norm','C_response_norm','C_norm','C_common_cosine','K_centered_norm','recurrent_spectral_norm','hidden_separation','centered_logit_separation']:
  fig,axs=plt.subplots(1,4,figsize=(17,4));fig.suptitle('Frozen mechanism: '+metric.replace('_',' '))
  for ax,c in zip(axs,CONDS):
   for a,d in cases(960):
    x=pick(g,c,a,d)
    for _,ss in x.groupby('seed'):ax.plot(ss['update'],ss[metric],color=AC[a],alpha=.07,lw=.5)
    z=x.groupby('update')[metric].mean();ax.plot(z.index,z,color=AC[a],ls=STYLE[a],label=a)
   ax.set(title=c,xlabel='Frozen target state',ylabel=metric.replace('_',' '));ax.legend(frameon=False)
  save(fig,out,'mechanism_'+metric)
 if (OUT/'tables/control_summary.csv').exists():
  co=read('control_summary');fig,axs=plt.subplots(3,4,figsize=(17,10));fig.suptitle('Frozen target330 C-drive controls (no optimization)');labels=['Natural-C','NoContext','C-removed','C-half','C-onehalf','Random']
  for i,(a,d) in enumerate(cases(960)):
   for j,c in enumerate(CONDS):
    ax=axs[i,j];x=pick(co,c,a,d);x=x[(x['update']==330)&(x.metric=='ce')].set_index('control').reindex(labels);ax.errorbar(range(6),x['mean'],yerr=np.array([x['mean']-x.ci_low,x.ci_high-x['mean']]),fmt='o',color=AC[a],capsize=3);ax.set_xticks(range(6),labels,rotation=35,fontsize=6);ax.set(title=c+' | '+a,ylabel='CE (nats)');chance(ax)
  save(fig,out,'mechanism_controls')
 if (OUT/'tables/heldout_summary.csv').exists():
  f=read('heldout_summary');fig,axs=plt.subplots(2,3,figsize=(15,9));fig.suptitle('Unseen conflict tasks after20 adaptation updates (diagnostic)')
  for ax,metric in zip(axs.flat,['ce','accuracy','conflict_accuracy','appropriate_context_change','prefix_centered_separation']):
   for j,c in enumerate(CONDS):
    xx=[];yy=[];lo=[];hi=[]
    for i,(a,d) in enumerate(cases(960)):
     z=pick(f,c,a,d);z=z[z.metric==metric].iloc[0];xx.append(j*4+i);yy.append(z['mean']);lo.append(z['mean']-z.ci_low);hi.append(z.ci_high-z['mean'])
    ax.errorbar(xx,yy,yerr=[lo,hi],fmt='o-',capsize=3,label=c)
   ax.set_xticks([1,5,9,13],CONDS);ax.set_ylabel(metric.replace('_',' '));ax.legend(frameon=False,fontsize=6)
  axs.flat[-1].axis('off');axs.flat[-1].text(.05,.8,'Each condition, left to right:\nHistorical (+0), Seen (+960), Transfer (+960)\n\nMeans and 95% seed-bootstrap intervals.\n12 tasks averaged within each of 20 seeds.\nCorrected v2: individually unseen full sequences.',va='top',transform=axs.flat[-1].transAxes,fontsize=11);save(fig,out,'mechanism_heldout_conflicts')

def draw_all(out=None):
 out=Path(out or OUT/'figures');out.mkdir(parents=True,exist_ok=True)
 for fn in [training,learning_curves,gaps,positions,specificity,mechanisms]:fn(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args();draw_all(a.output_dir)
