"""Deterministic, table-only scientific figures. Never loads model states."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/context_experience_prior/v1'
CONDS=['P0-I','P0-C','P1-I','P1-C','P2-I','P2-C'];OUT_NAMES=[f'figure_{i}' for i in range(1,6)]
COLORS={'A':'#2166ac','B':'#ef8a24','Mixed-average':'#26954b','Mixed-both':'#8e63ad','C':'#d03d54','NoContext':'#777777'};PC={'P0':'#777777','P1':'#238b8d','P2':'#9d4eaa'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'context_experience_prior_v1','savefig.dpi':160,'axes.titlesize':11})
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip')
def save(fig,dest,name):
 fig.tight_layout(pad=1.3,rect=(0,0,1,.97));fig.savefig(dest/f'{name}.png',metadata={'Software':'context_experience_prior_v1'});fig.savefig(dest/f'{name}.svg',metadata={'Date':None,'Creator':'context_experience_prior_v1'});plt.close(fig)
def panels(title):
 fig,axs=plt.subplots(3,2,figsize=(12,9),squeeze=False);fig.suptitle(title)
 for ax,c in zip(axs.flat,CONDS):ax.set_title(c+' — '+('Independent' if c.endswith('I') else 'Initially centered'))
 return fig,list(axs.flat)
def lines(ax,raw,ci,time,value,color,label,metric=None):
 for _,x in raw.groupby('seed'):
  x=x.sort_values(time);ax.plot(x[time],x[value],color=color,alpha=.09,lw=.5)
 z=ci[ci.metric.eq(metric or value)].sort_values(time);ax.fill_between(z[time],z.ci_low,z.ci_high,color=color,alpha=.13,lw=0);ax.plot(z[time],z['mean'],color=color,lw=1.5,label=label)
def zero(ax):ax.axhline(0,color='.5',lw=.6)
def chance(ax):ax.axhline(np.log(4),color='.5',ls=':',lw=1,label='ln(4)')
def training(dest):
 t=read('training');ci=read('training_intervals')
 for stop,suffix in [(330,''),(100,'_early')]:
  fig,axs=panels(f'Figure 1 — Actual pre-update training loss (updates 1–{stop})')
  for ax,c in zip(axs,CONDS):
   x=t[(t.condition==c)&(t['update']<=stop)];z=ci[(ci.condition==c)&(ci['update']<=stop)]
   for _,seed in x.groupby('seed'):ax.plot(seed['update'],seed.ce_before,color='.4',alpha=.10,lw=.5)
   ax.plot(z['update'],z['mean'],color='.4',lw=.6)
   for start in range(1,stop+1,10):
    cue='A' if (start-1)%20==0 else 'B';ff=x[x['update'].between(start,start+9)];ii=z[z['update'].between(start,start+9)];lines(ax,ff,ii,'update','ce_before',COLORS[cue],cue if start<=11 else None)
   ax.set(xlabel='Sequence repetition / update',ylabel='Practiced-context CE (nats)',xlim=(1,stop));ax.legend(frameon=False,ncol=2)
  save(fig,dest,'figure_1'+suffix)
def generalization(dest):
 f=read('evaluations');ci=read('evaluation_intervals')
 for stop,suffix in [(330,''),(120,'_early')]:
  fig,axs=panels(f'Figure 2 — Frozen natural-cue predictions (states 0–{stop})')
  for ax,c in zip(axs,CONDS):
   for cue,color in COLORS.items():
    x=f[(f.condition==c)&(f.cue==cue)&(f['update']<=stop)];z=ci[(ci.condition==c)&(ci.cue==cue)&(ci['update']<=stop)];lines(ax,x,z,'update','ce',color,cue)
   chance(ax);ax.set(xlabel='Frozen target state',ylabel='CE (nats)',xlim=(0,stop));ax.legend(frameon=False,ncol=3,fontsize=7)
  save(fig,dest,'figure_2'+suffix)
def learning(dest):
 b=read('block_transfer');ci=read('block_intervals')
 for metrics,name,title in [(['specificity'],'figure_3','Figure 3 — Signed context-specific block learning'),(['practiced_gain','unpracticed_gain'],'figure_3_gains','Block learning: own (solid) and cross-context (dashed)')]:
  fig,axs=panels(title)
  for ax,c in zip(axs,CONDS):
   for cue in ['A','B']:
    for metric in metrics:
     x=b[(b.condition==c)&(b.practiced==cue)];z=ci[(ci.condition==c)&(ci.practiced==cue)];lines(ax,x,z,'end',metric,COLORS[cue],cue+': '+metric.replace('_',' '))
     if metric=='unpracticed_gain':ax.lines[-1].set_linestyle('--')
    floor=x[x.floor_at_start];ax.scatter(floor.end,np.zeros(len(floor)),color=COLORS[cue],marker='|',alpha=.15,s=12)
   zero(ax);ax.set(xlabel='End of 10-update block',ylabel='CE gain / specificity (nats)');ax.legend(frameon=False,fontsize=7)
  fig.text(.5,.005,'Ticks at zero mark floor-limited blocks (both starting CE ≤ 0.05); these do not establish shared learning.',ha='center',fontsize=8);save(fig,dest,name)
 s=read('switch_costs');si=read('switch_intervals');gap=read('performance_gaps');gi=read('gap_intervals');fig,axs=panels('Switch costs and signed same-weight A/B gaps')
 for ax,c in zip(axs,CONDS):
  for direction in ['A_to_B','B_to_A']:
   x=s[(s.condition==c)&(s.direction==direction)];z=si[(si.condition==c)&(si.direction==direction)];color=COLORS['A' if direction=='A_to_B' else 'B'];lines(ax,x,z,'boundary','raw_cost',color,direction+' raw');lines(ax,x,z,'boundary','adjusted_cost',color,direction+' adjusted');ax.lines[-1].set_linestyle(':')
  lines(ax,gap[gap.condition==c],gi[gi.condition==c],'update','signed_A_minus_B','#333333','CE A − B');zero(ax);ax.legend(frameon=False,fontsize=6,ncol=2);ax.set(xlabel='Frozen state / switch boundary',ylabel='Signed CE difference (nats)')
 save(fig,dest,'figure_3_switches')
 d=read('learning_effects');di=read('diagnostic_intervals');fig,axs=panels('Restored-Adam one-update diagnostics (clones only)')
 for ax,c in zip(axs,CONDS):
  for metric,color in [('mean_own_improvement','#2166ac'),('mean_cross_improvement','#ef8a24'),('mean_specificity','#333333')]:lines(ax,d[d.condition==c],di[di.condition==c],'update',metric,color,metric.replace('_',' '))
  zero(ax);ax.legend(frameon=False,fontsize=7);ax.set(xlabel='Frozen state',ylabel='CE gain (nats)')
 save(fig,dest,'figure_3_diagnostics')
def associations(dest):
 f=read('transfer_associations');metrics=['early_specificity','hidden_separation','centered_logit_separation','early_switch_cost','mean_specificity','mean_cross_improvement']
 for k,suffix in [(330,''),(170,'_170')]:
  fig,axs=plt.subplots(2,3,figsize=(13,8));fig.suptitle(f'Figure 4 — Transfer versus learning/geometry at state {k} (descriptive)')
  for ax,metric in zip(axs.flat,metrics):
   for c in CONDS:
    x=f[(f.condition==c)&(f['update']==k)];ax.scatter(x[metric],x.ce,c=PC[c[:2]],marker='o' if c.endswith('I') else 's',s=26,alpha=.65,label=c,edgecolors=np.where(x.strong_familiar,'black','none'),linewidths=.7)
   chance(ax);ax.set(xlabel=metric.replace('_',' '),ylabel='C CE (nats)');ax.legend(frameon=False,fontsize=7,ncol=2)
  fig.text(.5,.005,'Each point is one network; black border = strong familiar acquisition. Geometry is not a splitting criterion.',ha='center',fontsize=8);save(fig,dest,'figure_4'+suffix)
def mechanisms(dest):
 g=read('geometry');f=read('endpoints');gate=read('functional_splitting');fig,axs=plt.subplots(2,3,figsize=(13,8));fig.suptitle('Figure 5 — What prior experience changes (all seeds)')
 specs=[('baseline_ce',0,'Initial context-free CE'),('C_ce',0,'Initial C CE'),('C_ce',330,'C CE after 330 updates'),('functional_split',330,'Original functional splitting gate (330)'),('recurrent_spectral_norm',330,'Recurrent spectral norm'),('K_centered_norm',330,'Class-centered recurrent sensitivity')]
 for ax,(metric,k,title) in zip(axs.flat,specs):
  for j,c in enumerate(CONDS):
   x=(gate if metric=='functional_split' else g);x=x[(x.condition==c)&(x['update']==k)].sort_values('seed');v=x[metric].astype(float).to_numpy();jitter=np.linspace(-.15,.15,len(v));ax.scatter(j+jitter,v,color=PC[c[:2]],s=13,alpha=.5,marker='o' if c.endswith('I') else 's');ax.plot([j-.2,j+.2],[v.mean(),v.mean()],color='black',lw=2)
   if metric=='C_ce' and k==330:
    q=f[(f.condition==c)&(f['update']==330)&(f.cue=='C')].sort_values('seed');good=q.strong_familiar.to_numpy(bool);ax.scatter(j+jitter[good],v[good],facecolors='none',edgecolors='black',s=38);ax.text(j,.98,f'n={good.sum()}',transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=7)
  ax.set_title(title);ax.set_xticks(range(6),CONDS,rotation=30);ax.set_ylabel('Proportion / 0–1 seed outcome' if metric=='functional_split' else 'Nats / norm');
  if metric=='functional_split':ax.set_ylim(-.05,1.05);ax.set_yticks([0,.5,1])
  if 'CE' in title:chance(ax)
 save(fig,dest,'figure_5')
 ci=read('geometry_intervals');metrics=['baseline_ce','C_ce','baseline_margin','C_margin','u_norm','v_norm','C_response_norm','recurrent_spectral_norm','K_centered_norm','g_norm','d_norm','C_norm']
 for n,mm in enumerate([metrics[:6],metrics[6:]]):
  fig,axs=plt.subplots(2,3,figsize=(13,8));fig.suptitle('Linear mechanisms throughout target learning — '+('predictions' if n==0 else 'weights and sensitivity'))
  for ax,metric in zip(axs.flat,mm):
   for c in CONDS:
    lines(ax,g[g.condition==c],ci[ci.condition==c],'update',metric,PC[c[:2]],c);ax.lines[-1].set_linestyle('-' if c.endswith('I') else '--')
   ax.set(xlabel='Frozen target state',ylabel=metric.replace('_',' '));ax.legend(frameon=False,ncol=2,fontsize=7)
  save(fig,dest,f'figure_5_mechanisms_{n+1}')
def draw_all(dest=None):
 dest=Path(dest or OUT/'figures');dest.mkdir(parents=True,exist_ok=True)
 for fun in [training,generalization,learning,associations,mechanisms]:fun(dest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);args=p.parse_args();draw_all(args.output_dir)
