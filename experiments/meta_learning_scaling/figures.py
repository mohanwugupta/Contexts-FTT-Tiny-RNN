"""Deterministic table-only figures: no checkpoint, model evaluation or training."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/meta_learning_scaling/v1';PRIMARY=[f'figure_{i}' for i in range(1,6)];CONDITIONS=['P1-I','P1-C','P2-I','P2-C'];CUE_COLORS={'A':'#2166ac','B':'#ef8a24','C':'#d03d54','Mixed-average':'#26954b','Mixed-both':'#8e63ad','NoContext':'#777777'};CC={'P1-I':'#238b8d','P1-C':'#5b9dd0','P2-I':'#9d4eaa','P2-C':'#c7824b'}
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'meta_learning_scaling_v1','savefig.dpi':140,'axes.titlesize':9})
def read(dest,n):return pd.read_csv(dest/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def save(fig,out,name):
 fig.tight_layout(pad=1.0,rect=(0,.025,1,.965));fig.savefig(out/f'{name}.png',metadata={'Software':'meta_learning_scaling_v1'});fig.savefig(out/f'{name}.svg',metadata={'Date':None,'Creator':'meta_learning_scaling_v1'});plt.close(fig)
def log_dose_axis(ax,doses):
 ax.set_xscale('symlog',linthresh=20);ax.set_xticks(doses,[str(d) for d in doses],rotation=35,fontsize=7);ax.set_xlabel('Meta episodes (positive doses log spaced; 0 baseline)')
def chance(ax):ax.axhline(np.log(4),color='.5',ls=':',lw=.8)
def ci_line(ax,raw,ci,time,metric,color,label=None,alpha=.1,style='-'):
 for _,s in raw.groupby('seed'):
  s=s.sort_values(time);ax.plot(s[time],s[metric],color=color,alpha=alpha,lw=.5)
 z=ci[ci.metric==metric].sort_values(time);ax.plot(z[time],z['mean'],color=color,label=label,lw=1.4,ls=style);ax.fill_between(z[time],z.ci_low,z.ci_high,color=color,alpha=.13,lw=0)
def dose_grid(doses,title):
 fig,axs=plt.subplots(len(doses),4,figsize=(17,3*len(doses)),squeeze=False);fig.suptitle(title)
 for i,d in enumerate(doses):
  for j,c in enumerate(CONDITIONS):axs[i,j].set_title(f'{c} | {d} meta episodes')
 return fig,axs

def training(dest,out,doses):
 t=read(dest,'training');ci=read(dest,'training_intervals');main=[d for d in [0,240,960,1920] if d in doses];extra=[d for d in doses if d not in main];pages=[(main,'')]+[(extra[i:i+3],f'_supp_{i//3+1}') for i in range(0,len(extra),3)]
 for dd,suffix in pages:
  for stop,zoom in [(330,''),(100,'_early')]:
   fig,axs=dose_grid(dd,f'Figure 1 — Actual practiced-context pre-update loss (1–{stop})')
   for i,dose in enumerate(dd):
    for j,c in enumerate(CONDITIONS):
     ax=axs[i,j];x=t[(t.condition==c)&(t.dose==dose)&(t['update']<=stop)];z=ci[(ci.condition==c)&(ci.dose==dose)&(ci['update']<=stop)]
     for _,ss in x.groupby('seed'):ax.plot(ss['update'],ss.ce_before,color='.4',alpha=.1,lw=.4)
     ax.plot(z['update'],z['mean'],color='.4',lw=.5)
     for start in range(1,stop+1,10):
      cue='A' if (start-1)%20==0 else 'B';ci_line(ax,x[x['update'].between(start,start+9)],z[z['update'].between(start,start+9)],'update','ce_before',CUE_COLORS[cue],cue if start<=11 else None)
     ax.set(xlabel='Target update',ylabel='Actual CE (nats)',xlim=(1,stop));ax.legend(frameon=False,ncol=2,fontsize=7)
   fig.text(.5,.007,'20 seed traces; mean and 95% whole-seed bootstrap. Dose0 is the same physical P0 history in P1/P2 panels.',ha='center');save(fig,out,'figure_1'+suffix+zoom)

def generalization(dest,out,doses):
 f=read(dest,'evaluations');ci=read(dest,'evaluation_intervals');main=[d for d in [0,240,960,1920] if d in doses];extra=[d for d in doses if d not in main];pages=[(main,'')]+[(extra[i:i+3],f'_supp_{i//3+1}') for i in range(0,len(extra),3)]
 for dd,suffix in pages:
  for stop,zoom in [(330,''),(120,'_early')]:
   fig,axs=dose_grid(dd,f'Figure 2 — Frozen natural-cue CE (states 0–{stop})')
   for i,dose in enumerate(dd):
    for j,c in enumerate(CONDITIONS):
     ax=axs[i,j]
     for cue,color in CUE_COLORS.items():
      x=f[(f.condition==c)&(f.dose==dose)&(f.cue==cue)&(f['update']<=stop)];z=ci[(ci.condition==c)&(ci.dose==dose)&(ci.cue==cue)&(ci['update']<=stop)];ci_line(ax,x,z,'update','ce',color,cue)
     chance(ax);ax.set(xlabel='Frozen target state',ylabel='CE (nats)',xlim=(0,stop));ax.legend(frameon=False,ncol=3,fontsize=6)
   save(fig,out,'figure_2'+suffix+zoom)

def scaling(dest,out,doses):
 ep=read(dest,'endpoints');su=read(dest,'endpoint_summary');ac=read(dest,'acquisition_timing');acs=read(dest,'acquisition_summary');ga=read(dest,'functional_splitting');js=read(dest,'joint_summary');fig,axs=plt.subplots(2,3,figsize=(16,9));fig.suptitle('Figure 3 — Registered pretraining dose response')
 specs=[('C',170,'ce','C CE at target170'),('C',330,'ce','C CE at target330'),('C',330,'success','C success at330 (%)'),('Mixed-both',330,'success','Mixed-both success at330 (%)'),('acq',330,'first_familiar','First strong familiar acquisition'),('gate',330,'functional_split','Original functional split at330 (%)')]
 for ax,(cue,k,metric,title) in zip(axs.flat,specs):
  for c in CONDITIONS:
   if cue=='acq':raw=ac[ac.condition==c];z=acs[(acs.condition==c)&(acs.metric==metric)].copy()
   elif cue=='gate':raw=ga[(ga.condition==c)&(ga['update']==k)];z=js[(js.condition==c)&(js['update']==k)&(js.metric==metric)].copy()
   else:raw=ep[(ep.condition==c)&(ep['update']==k)&(ep.cue==cue)];z=su[(su.condition==c)&(su['update']==k)&(su.cue==cue)&(su.metric==metric)].copy()
   mult=100 if metric in ['success','functional_split'] else 1;raw=raw.copy();raw[metric]=raw[metric].astype(float)*mult
   for col in ['mean','ci_low','ci_high']:z[col]*=mult
   ci_line(ax,raw,z,'dose',metric,CC[c],c,style='-' if c.endswith('I') else '--')
  log_dose_axis(ax,doses);ax.set_title(title);ax.legend(frameon=False,ncol=2)
  if metric in ['success','functional_split']:ax.set_ylim(-5,105)
  elif metric=='ce':chance(ax)
 fig.text(.5,.005,'Lines connect paired seeds; never-acquired timing is missing/censored. Percentages retain all20 primary seeds.',ha='center');save(fig,out,'figure_3')

def specificity_plots(dest,out,doses):
 b=read(dest,'block_transfer');ci=read(dest,'block_intervals');palette={d:plt.cm.viridis(i/max(1,len(doses)-1)) for i,d in enumerate(doses)}
 for metrics,name in [(['specificity'],'figure_4'),(['practiced_gain','unpracticed_gain'],'figure_4_gains')]:
  fig,axs=plt.subplots(2,4,figsize=(17,8));fig.suptitle('Figure 4 — Signed block learning across pretraining doses')
  for j,c in enumerate(CONDITIONS):
   for i,cue in enumerate(['A','B']):
    ax=axs[i,j]
    for dose in doses:
     x=b[(b.condition==c)&(b.dose==dose)&(b.practiced==cue)];z=ci[(ci.condition==c)&(ci.dose==dose)&(ci.practiced==cue)]
     for metric in metrics:ci_line(ax,x,z,'end',metric,palette[dose],str(dose)+(' own' if metric=='practiced_gain' else ' cross' if metric=='unpracticed_gain' else ''),alpha=.04,style='--' if metric=='unpracticed_gain' else '-')
     floor=x[x.floor_at_start];ax.scatter(floor.end,np.zeros(len(floor)),s=5,marker='|',alpha=.1,color=palette[dose])
    ax.axhline(0,color='.5',lw=.5);ax.set(title=c+' | '+cue+' practice',xlabel='End of target block',ylabel='Signed CE gain (nats)');ax.legend(frameon=False,ncol=3,fontsize=6)
  fig.text(.5,.005,'Ticks mark both-context floor (starting CE≤0.05); no-learning there is not evidence of shared representations.',ha='center');save(fig,out,name)
 m=read(dest,'matched_acquisition');ms=read(dest,'matched_summary');ranges=list(m.performance_range.drop_duplicates());fig,axs=plt.subplots(4,4,figsize=(17,13));fig.suptitle('Matched familiar performance — first eligible block (conditional description)')
 for i,perf in enumerate(ranges):
  for j,c in enumerate(CONDITIONS):
   ax=axs[i,j];x=m[(m.condition==c)&(m.performance_range==perf)&(m.stratum=='all')];z=ms[(ms.condition==c)&(ms.performance_range==perf)&(ms.stratum=='all')&(ms.metric=='specificity')]
   for cue in ['A','B']:
    xx=x[x.practiced==cue];ax.scatter(xx.dose,xx.specificity,color=CUE_COLORS[cue],alpha=.25,s=10,label=cue+' practiced')
   ax.errorbar(z.dose,z['mean'],yerr=[z['mean']-z.ci_low,z.ci_high-z['mean']],fmt='k.-',lw=1,capsize=2);ax.axhline(0,color='.5',lw=.5);log_dose_axis(ax,doses);ax.set_title(c+' | '+perf);ax.set_ylabel('Own − cross CE gain');ax.legend(frameon=False,fontsize=6)
   for row in z.itertuples():
    if row.n<20:ax.text(row.dose,.98,str(row.n),transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=6)
 fig.text(.5,.005,'Blue/orange identify the practiced cue of the first eligible block; numbers show eligible n when <20. Missing bins stay missing.',ha='center');save(fig,out,'figure_4_matched')
 s=read(dest,'switch_costs');si=read(dest,'switch_intervals');fig,axs=plt.subplots(2,4,figsize=(17,8));fig.suptitle('Same-weight switch costs (solid raw, dotted initial-bias adjusted)')
 for j,c in enumerate(CONDITIONS):
  for i,direction in enumerate(['A_to_B','B_to_A']):
   ax=axs[i,j]
   for dose in doses:
    x=s[(s.condition==c)&(s.dose==dose)&(s.direction==direction)];z=si[(si.condition==c)&(si.dose==dose)&(si.direction==direction)]
    for metric in ['raw_cost','adjusted_cost']:ci_line(ax,x,z,'boundary',metric,palette[dose],str(dose) if metric=='raw_cost' else None,alpha=.04,style='-' if metric=='raw_cost' else ':')
   ax.axhline(0,color='.5',lw=.5);ax.set(title=c+' | '+direction,xlabel='Target boundary',ylabel='Signed CE cost');ax.legend(frameon=False,ncol=3,fontsize=6)
 save(fig,out,'figure_4_switches')
 d=read(dest,'learning_effects');di=read(dest,'diagnostic_intervals');fig,axs=plt.subplots(3,4,figsize=(17,10));fig.suptitle('Restored-Adam one-update diagnostics (isolated clones)')
 for j,c in enumerate(CONDITIONS):
  for i,metric in enumerate(['mean_own_improvement','mean_cross_improvement','mean_specificity']):
   ax=axs[i,j]
   for dose in doses:ci_line(ax,d[(d.condition==c)&(d.dose==dose)],di[(di.condition==c)&(di.dose==dose)],'update',metric,palette[dose],str(dose),alpha=.05)
   ax.axhline(0,color='.5',lw=.5);ax.set(title=c,xlabel='Frozen target state',ylabel=metric.replace('_',' '));ax.legend(frameon=False,ncol=3,fontsize=6)
 save(fig,out,'figure_4_diagnostics')

def coexistence(dest,out,doses):
 j=read(dest,'joint_outcomes');e=read(dest,'endpoints');j=j[j['update']==330].merge(e[(e['update']==330)&(e.cue=='C')][['seed','condition','dose','ce']],on=['seed','condition','dose']);fig,axs=plt.subplots(int(np.ceil(len(doses)/3)),3,figsize=(15,4*int(np.ceil(len(doses)/3))),squeeze=False);fig.suptitle('Figure 5 — Natural C transfer versus early functional learning')
 for ax,dose in zip(axs.flat,doses):
  for c in CONDITIONS:
   x=j[(j.condition==c)&(j.dose==dose)];ax.scatter(x.early_specificity,x.ce,facecolors=np.where(x.strong_familiar.eq(1),CC[c],'none'),edgecolors=CC[c],marker='o' if c.endswith('I') else 's',s=24,alpha=.65,label=c);good=x[x.functional_split==True];ax.scatter(good.early_specificity,good.ce,s=75,facecolors='none',edgecolors='black');yes=x[x.nonfloor_coexistence==True];ax.scatter(yes.early_specificity,yes.ce,s=13,marker='+',color='black')
  chance(ax);ax.set(title=f'{dose} meta episodes',xlabel='Mean signed specificity (first100 updates)',ylabel='C CE at330');ax.legend(frameon=False,ncol=2,fontsize=6)
 for ax in list(axs.flat)[len(doses):]:ax.set_visible(False)
 fig.text(.5,.005,'Hollow=familiar failure; ring=original gate; +=C+Both+familiar success with earlier nonfloor bidirectional positive learning (descriptive, not original joint gate).',ha='center');save(fig,out,'figure_5')
 m=read(dest,'matched_acquisition');m=m[m.stratum=='all'].merge(j[['seed','condition','dose','ce','strong_familiar','functional_split']],on=['seed','condition','dose']);ranges=list(m.performance_range.drop_duplicates())
 for page in range(0,len(doses),3):
  dd=doses[page:page+3];fig,axs=plt.subplots(len(dd),4,figsize=(17,3.5*len(dd)),squeeze=False);fig.suptitle('C transfer versus matched-acquisition specificity — conditional, missing bins omitted')
  for i,dose in enumerate(dd):
   for k,perf in enumerate(ranges):
    ax=axs[i,k]
    for c in CONDITIONS:
     x=m[(m.condition==c)&(m.dose==dose)&(m.performance_range==perf)];ax.scatter(x.specificity,x.ce,facecolors=np.where(x.strong_familiar.eq(1),CC[c],'none'),edgecolors=CC[c],marker='o' if c.endswith('I') else 's',s=20,alpha=.6,label=c)
    chance(ax);ax.set(title=f'{dose} episodes | {perf}',xlabel='Matched own − cross gain',ylabel='C CE at330');ax.legend(frameon=False,ncol=2,fontsize=6)
  save(fig,out,f'figure_5_matched_{page//3+1}')
 gs=read(dest,'geometry_summary');geo=read(dest,'geometry');metrics=['baseline_ce','C_ce','u_norm','v_norm','C_response_norm','K_centered_norm','recurrent_spectral_norm','C_norm','hidden_separation','centered_logit_separation','baseline_margin','C_margin']
 for page in range(0,len(metrics),6):
  mm=metrics[page:page+6];fig,axs=plt.subplots(2,3,figsize=(16,9));fig.suptitle('Mechanistic dose response at target330 — descriptive')
  for ax,metric in zip(axs.flat,mm):
   for c in CONDITIONS:ci_line(ax,geo[(geo.condition==c)&(geo['update']==330)],gs[(gs.condition==c)&(gs['update']==330)],'dose',metric,CC[c],c,style='-' if c.endswith('I') else '--')
   log_dose_axis(ax,doses);ax.set_ylabel(metric.replace('_',' '));ax.legend(frameon=False,ncol=2,fontsize=7)
  save(fig,out,f'figure_5_mechanisms_{page//6+1}')
 # Full recurrent mechanism trajectories are kept, with allseeds and cohort means.
 for metric in ['baseline_ce','C_ce','K_centered_norm','recurrent_spectral_norm','u_norm','v_norm','C_response_norm','A_norm','B_norm','C_norm','baseline_margin','C_margin']:
  fig,axs=plt.subplots(1,4,figsize=(17,3.7));fig.suptitle('Full target trajectory: '+metric.replace('_',' '))
  for ax,c in zip(axs,CONDITIONS):
   for i,dose in enumerate(doses):
    x=geo[(geo.condition==c)&(geo.dose==dose)];color=plt.cm.viridis(i/max(1,len(doses)-1))
    for _,ss in x.groupby('seed'):ax.plot(ss['update'],ss[metric],color=color,alpha=.035,lw=.4)
    z=x.groupby('update')[metric].mean();ax.plot(z.index,z,color=color,label=str(dose),lw=1)
   ax.set(title=c,xlabel='Frozen target state',ylabel=metric.replace('_',' '));ax.legend(frameon=False,ncol=3,fontsize=6)
  save(fig,out,'figure_5_trajectory_'+metric)

def draw_all(dest,out=None):
 out=Path(out or dest/'figures');out.mkdir(parents=True,exist_ok=True);doses=sorted(read(dest,'physical_run_registry').dose.unique().tolist())
 for fun in [training,generalization,scaling,specificity_plots,coexistence]:fun(dest,out,doses)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['a','full'],required=True);p.add_argument('--output-dir',type=Path);a=p.parse_args();draw_all(OUT/'stage_a' if a.stage=='a' else OUT,a.output_dir)
