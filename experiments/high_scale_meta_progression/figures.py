"""Deterministic table-only PNG/SVG figures; no model imports/evaluation."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/high_scale_meta_progression/v1'
FIGURE_FAMILIES=['extended_fresh','fresh_six_contexts','Reptile','FOMAML','joint','effective_computation']
COLORS={'A':'#1f77b4','B':'#ff7f0e','C':'#2ca02c','Mixed-average':'#9467bd','Mixed-both':'#d62728','NoContext':'#777777','R':'#111111','T':'#1f77b4','O1':'#9467bd','O2':'#2ca02c'}
ARMCOLORS={'R':'#111111','T':'#1f77b4','B':'#ff7f0e','O1':'#9467bd','O2':'#2ca02c'}
plt.rcParams.update({'svg.hashsalt':'high_scale_meta_progression_v1','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def destination(root,target):
 root=Path(root);p=Path(target) if target is not None else root/'figures'
 if (root/'manifest.json').exists() and p.resolve().is_relative_to(root.resolve()):raise ValueError('Sealed figures require an external destination')
 p.mkdir(parents=True,exist_ok=True);return p

def main(source=OUT,output_dir=None):
 source=Path(source);dest=destination(source,output_dir);read=lambda n:pd.read_csv(source/f'tables/{n}.csv',float_precision='round_trip');save=lambda fig,n:_save(fig,dest,n)
 out=read('outcomes');summary=read('outcome_summary');t=read('training');ti=read('fresh_training_intervals');f=read('evaluations');fi=read('fresh_evaluations_intervals');held=read('conflict_seed');hs=read('conflict_summary');geom=read('meta_geometry')
 for stop,view,symlog in [(990,'full990',False),(100,'first100',False),(20,'first20',False),(990,'closeup',True)]:
  fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True);fresh=t[t.phase.eq('Fresh')&t['update'].le(stop)];top=fresh.ce_before.max()*1.03
  for ax,scale in zip(axes,[4,5,6]):
   g=fresh[fresh.scale.eq(scale)];ci=ti[ti.scale.eq(scale)&ti['update'].le(stop)].sort_values('update')
   for start in range(1,stop+1,10):
    end=min(start+9,stop);c='A' if ((start-1)//10)%2==0 else 'B';q=g[g['update'].between(start,end)]
    for _,seed in q.groupby('seed'):ax.plot(seed['update'],seed.ce_before,color=COLORS[c],alpha=.14,lw=.6)
    z=ci[ci['update'].between(start,end)];ax.plot(z['update'],z['mean'],color=COLORS[c],lw=1.7,label=c if start<=20 else None);ax.fill_between(z['update'],z.ci_low,z.ci_high,color=COLORS[c],alpha=.18)
    if start>1:ax.axvline(start-.5,color='k',alpha=.08,lw=.4)
   ax.set(title=f'Nominal scale {scale}',xlabel='Target sequence repetition',xlim=(1,stop),ylim=(0,top));ax.legend(loc='upper right')
   if symlog:ax.set_yscale('symlog',linthresh=.1)
  axes[0].set_ylabel('Actual pre-update CE (nats)'+(' — symlog' if symlog else ''));fig.suptitle('Fresh A10/B10 acquisition: six seeds and 95% seed-bootstrap intervals');fig.tight_layout();save(fig,'figure_1_'+view)
 fig,axes=plt.subplots(3,6,figsize=(22,10),sharex=True,sharey=True)
 for i,scale in enumerate([4,5,6]):
  for j,cue in enumerate(['A','B','C','Mixed-average','Mixed-both','NoContext']):
   ax=axes[i,j];g=f[f.phase.eq('Fresh')&f.scale.eq(scale)&f.cue.eq(cue)];ci=fi[fi.scale.eq(scale)&fi.cue.eq(cue)&fi.metric.eq('ce')].sort_values('update')
   for _,z in g.groupby('seed'):ax.plot(z['update'],z.ce,color=COLORS[cue],alpha=.17,lw=.65)
   ax.plot(ci['update'],ci['mean'],color=COLORS[cue]);ax.fill_between(ci['update'],ci.ci_low,ci.ci_high,color=COLORS[cue],alpha=.18);ax.axhline(np.log(4),color='k',ls=':',lw=.8)
   for k in [330,660,990]:ax.axvline(k,color='k',alpha=.12,lw=.6)
   ax.set_yscale('symlog',linthresh=.01);ax.set_title(f's={scale}: {cue}');ax.set_xlabel('Target update')
   if j==0:ax.set_ylabel('CE nats — symlog')
 fig.suptitle('Fresh physical-cue generalization; dotted horizontal line = ln(4)');fig.tight_layout();save(fig,'figure_2_six_contexts')
 metrics=['D_A','D_B','early_cost','A_to_B','B_to_A','sustained_cost','gap','Mixed_both_ce'];fig,axes=plt.subplots(len(metrics),3,figsize=(15,20),squeeze=False)
 q=out[out['update'].eq(330)&out.phase.isin(['Fresh','R'])].copy();q['episode']=np.where(q.phase.eq('Fresh'),0,q.dose)
 for i,m in enumerate(metrics):
  for j,scale in enumerate([4,5,6]):
   ax=axes[i,j];z=q[q.scale.eq(scale)].sort_values('episode')
   for _,seed in z.groupby('seed'):ax.plot(seed.episode,seed[m],color='#555555',alpha=.25,lw=.7,marker='.',ms=3)
   _means(ax,z,'episode',m,color='k');ax.axhline(0,color='k',lw=.5);ax.set_title(f's={scale}: {m}');ax.set_xlabel('Reptile episodes');ax.set_ylabel('Nats')
 fig.suptitle('Reptile effects at identical target update330; all six paired seeds');fig.tight_layout();save(fig,'figure_3_reptile')
 for y,name in [('early_cost','balanced'),('D_A','first_A'),('D_B','first_B')]:
  fig,axes=plt.subplots(1,3,figsize=(15,5.1))
  for ax,scale in zip(axes,[4,5,6]):
   parent=out[out.scale.eq(scale)&out.phase.eq('R')&out.dose.eq(1920)&out['update'].eq(330)]
   for arm in ['T','B','O1','O2']:
    g=out[out.scale.eq(scale)&out.phase.eq(arm)&out['update'].eq(330)];z=pd.concat([parent.assign(dose=0),g]).sort_values('dose');color=ARMCOLORS[arm]
    for _,seed in z.groupby('seed'):ax.plot(seed.gap,seed[y],color=color,alpha=.18,lw=.7,marker='.',ms=3)
    means=z.groupby('dose')[['gap',y]].mean();ax.plot(means.gap,means[y],color=color,lw=2,label=arm)
    for k,row in means.iterrows():
     from experiments.meta_objective_balance.statistics import interval
     zz=z[z.dose.eq(k)].set_index('seed');cx=interval(zz.gap.reindex(range(6)).to_numpy(float),list(range(6)));cy=interval(zz[y].reindex(range(6)).to_numpy(float),list(range(6)));ax.errorbar(row.gap,row[y],xerr=[[max(0,row.gap-cx['ci_low'])],[max(0,cx['ci_high']-row.gap)]],yerr=[[max(0,row[y]-cy['ci_low'])],[max(0,cy['ci_high']-row[y])]],color=color,alpha=.45,lw=.65,fmt='none');ax.scatter(row.gap,row[y],marker={0:'s',240:'o',960:'^',1920:'D'}[int(k)],color=color,s=32,zorder=5)
   ax.axhline(0,color='k',lw=.5);ax.axvline(0,color='k',lw=.5);ax.set_xscale('symlog',linthresh=.1);ax.set(title=f'scale {scale}',xlabel='C gap at target330 (nats; symlog, linear ±0.1)',ylabel={'early_cost':'Balanced early switch cost (nats)','D_A':'First-A specificity (nats)','D_B':'First-B specificity (nats)'}[y]);ax.legend()
  fig.suptitle('Four FOMAML procedures from one Reptile parent');fig.text(.5,.015,'Mean markers: square = parent, circle = +240, triangle = +960, diamond = +1920; pointwise 95% seed-bootstrap intervals',ha='center',fontsize=9);fig.tight_layout(rect=(0,.06,1,.93));save(fig,'figure_4_fomaml_'+name)
 joint_metrics=['gap','D_A','D_B','sustained_cost','Mixed_both_ce','familiar_ce','conflict_accuracy'];fig,axes=plt.subplots(len(joint_metrics),3,figsize=(15,18),squeeze=False)
 for i,m in enumerate(joint_metrics):
  for j,scale in enumerate([4,5,6]):
   ax=axes[i,j];z=(held if m=='conflict_accuracy' else out[out['update'].eq(990)]);z=z[z.scale.eq(scale)&z.dose.eq(1920)&z.phase.isin(ARMCOLORS)]
   for x,arm in enumerate(['R','T','B','O1','O2']):
    a=z[z.phase.eq(arm)].sort_values('seed');values=a[m].to_numpy(float);ax.scatter(x+(a.seed-2.5)*.035,values,color=ARMCOLORS[arm],s=15,alpha=.65)
    from experiments.meta_objective_balance.statistics import interval
    ci=interval(a.set_index('seed')[m].reindex(range(6)).to_numpy(float),list(range(6)));ax.errorbar(x,ci['mean'],yerr=[[ci['mean']-ci['ci_low']],[ci['ci_high']-ci['mean']]],fmt='o',color='k',capsize=3,ms=3)
   ax.axhline(0,color='k',lw=.4);ax.set_xticks(range(5),['R','T','B','O1','O2']);ax.set_title(f's={scale}: {m}')
 fig.suptitle('Final procedures: target990 and separately reserved conflict adaptation; all six seeds');fig.tight_layout(rect=(0,0,1,.97));save(fig,'figure_5_joint_components')
 for metrics,name in [(['token','context_A','context_B','context_C','output'],'RMS'),(['recurrent_spectral_norm','centered_contrast','AB_column_difference','NoContext_ce'],'computation')]:
  fig,axes=plt.subplots(len(metrics),3,figsize=(15,3*len(metrics)),squeeze=False)
  for i,m in enumerate(metrics):
   for j,scale in enumerate([4,5,6]):
    ax=axes[i,j];g=geom[geom.scale.eq(scale)].copy()
    if m=='NoContext_ce':g=out[out.scale.eq(scale)&out['update'].eq(330)].copy()
    parent=g[g.phase.isin(['Fresh','R'])].copy();parent['stage']=np.where(parent.phase.eq('Fresh'),0,parent.dose.map({240:1,960:2,1920:3}));_trace(ax,parent,m,'k','Fresh → Reptile')
    for arm in ['T','B','O1','O2']:
     z=g[g.phase.eq(arm)].copy();z['stage']=z.dose.map({240:4,960:5,1920:6});z=pd.concat([parent[parent.stage.eq(3)],z]).sort_values('stage');_trace(ax,z,m,ARMCOLORS[arm],arm)
    ax.set_xticks(range(7),['Fresh','R240','R960','R1920','F240','F960','F1920'],rotation=40);ax.set_title(f's={scale}: {m}')
    if i==0 and j==0:ax.legend(fontsize=8)
  fig.suptitle('Weight scale and effective computation; descriptive, not causal mediation');fig.tight_layout();save(fig,'figure_6_'+name)

def _means(ax,z,x,y,color):
 from experiments.meta_objective_balance.statistics import interval
 rows=[]
 for k,g in z.groupby(x):
  ci=interval(g.set_index('seed')[y].reindex(range(6)).to_numpy(float),list(range(6)));rows.append(dict(x=k,**ci))
 q=pd.DataFrame(rows).sort_values('x');ax.plot(q.x,q['mean'],color=color,lw=1.7);ax.fill_between(q.x,q.ci_low,q.ci_high,color=color,alpha=.15)

def _trace(ax,z,m,color,label):
 for _,g in z.groupby('seed'):ax.plot(g.stage,g[m],color=color,alpha=.15,lw=.6)
 _means(ax,z,'stage',m,color);ax.lines[-1].set_label(label)

def _save(fig,dest,n):
 fig.savefig(dest/(n+'.png'),dpi=160,metadata={'Software':'high_scale_meta_progression/v1'});fig.savefig(dest/(n+'.svg'),metadata={'Date':None});plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=OUT);p.add_argument('--output-dir',type=Path);a=p.parse_args();main(a.source,a.output_dir)
