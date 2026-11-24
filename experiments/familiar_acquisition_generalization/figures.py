"""Three saved-table-only figures, with seeds paired across checkpoints."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from .audit import CHANCE,bootstrap
COLORS={'Original':'#333333','B':'#2166ac','I':'#1b7837','S':'#b35806'}
MARKERS={'Original':'o','B':'s','I':'^','S':'D'}


def validate_inputs(q):
 expected={(s,a) for s in range(20) for a in COLORS}
 if q.duplicated(['seed','arm']).any() or set(q[['seed','arm']].itertuples(index=False,name=None))!=expected:raise ValueError('Figure needs80 unique models, not duplicated originals')
 return True


def generate(out,destination=None):
 out=Path(out);dest=Path(destination) if destination else out/'figures';dest.mkdir(exist_ok=True)
 q=pd.read_csv(out/'tables/seed_level.csv',float_precision='round_trip');d=pd.read_csv(out/'tables/paired_changes.csv',float_precision='round_trip');validate_inputs(q);orig=q[q.arm=='Original'].sort_values('seed')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'srn-familiar-acquisition-audit-v1'})
 def save(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext)
   if p.exists():raise FileExistsError(p)
   fig.savefig(p,dpi=180,metadata={'Software':'srn-familiar-audit-v1'} if ext=='png' else {'Date':None,'Creator':'srn-familiar-audit-v1'})
  plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(12,8))
 for col,arm in enumerate(['B','I','S']):
  final=q[q.arm==arm].sort_values('seed')
  for row,metric in enumerate(['familiar_ce','ce_Mixed']):
   ax=axes[row,col]
   for i,(a,b) in enumerate(zip(orig[metric],final[metric])):
    jitter=(i-9.5)*.008;ax.plot([jitter,1+jitter],[a,b],'-',color=COLORS[arm],alpha=.35,lw=.8);ax.scatter([jitter,1+jitter],[a,b],s=13,facecolors='none',edgecolors=COLORS[arm],lw=.7)
   for i,frame in enumerate([orig,final]):
    s=bootstrap(frame,metric);ax.errorbar(i,s['mean'],yerr=[[s['mean']-s['ci_low']],[s['ci_high']-s['mean']]],fmt='o',color='black',capsize=4,lw=2,ms=5)
   ax.axhline(CHANCE,color='.4',ls='--',lw=1,label='Uniform ln(4)=1.386');ax.set(xticks=[0,1],xticklabels=['Original170','Continued330'],ylabel=('Familiar A/B CE' if row==0 else 'Mixed CE')+' (nats)');ax.grid(axis='y',alpha=.15)
   if row==0:ax.set_title({'B':'B: blocked','I':'I: whole-sequence interleaved','S':'S: within-sequence switch'}[arm])
 axes[0,0].legend(fontsize=8);fig.suptitle('Familiar acquisition and absolute Mixed prediction: paired seeds',fontsize=14);fig.text(.5,.015,'20 paired seeds per schedule. Open circles/lines: individuals; black marks: cohort means and seed-bootstrap95% intervals.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.95));save(fig,'fig_1_before_after')
 fig,axes=plt.subplots(1,2,figsize=(12,5.6))
 for arm in ['B','I','S']:
  final=q[q.arm==arm].sort_values('seed')
  for (_,a),(_,b) in zip(orig.iterrows(),final.iterrows()):axes[0].plot([a.familiar_ce,b.familiar_ce],[a.ce_Mixed,b.ce_Mixed],color=COLORS[arm],alpha=.12,lw=.8)
 for arm,g in q.groupby('arm'):
  axes[0].scatter(g.familiar_ce,g.ce_Mixed,c=COLORS[arm],marker=MARKERS[arm],s=32,label=arm,alpha=.85)
 flagged=q[q.strong_retained_poor];axes[0].scatter(flagged.familiar_ce,flagged.ce_Mixed,s=100,facecolors='none',edgecolors='black',lw=1.2,label='Strong familiar + retained separation + poor Mixed')
 success=q[q.strong_retained_success];axes[0].scatter(success.familiar_ce,success.ce_Mixed,s=110,marker='+',color='#762a83',lw=2,label='Strong familiar + retained + successful Mixed')
 for arm,g in flagged.groupby('arm'):
  for r in g.sort_values('seed').head(2).itertuples():axes[0].annotate(f'{arm}{r.seed}',(r.familiar_ce,r.ce_Mixed),xytext=(4,4),textcoords='offset points',fontsize=8)
 axes[0].axhline(CHANCE,color='.5',ls='--');axes[0].set(xlabel='Familiar mean CE (nats)',ylabel='Absolute Mixed CE (nats)',title='All80 saved networks; originals shown once');axes[0].legend(fontsize=7,loc='upper right')
 for arm,g in d.groupby('arm'):axes[1].scatter(g.delta_familiar_ce,g.delta_ce_Mixed,marker=MARKERS[arm],color=COLORS[arm],label=arm,s=35)
 axes[1].axhline(0,color='.5',lw=.8);axes[1].axvline(0,color='.5',lw=.8);axes[1].set(xlabel='Change in familiar CE (330−170)',ylabel='Change in Mixed CE (330−170)',title='Paired changes: descriptive association');axes[1].legend()
 fig.suptitle('Does better familiar learning accompany better transfer?',fontsize=14);fig.text(.5,.015,'Lower CE is better. Lines connect each original to its matched endpoints; correlations do not identify causal mediation.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.94));save(fig,'fig_2_familiar_vs_mixed')
 fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
 for ax,metric,label in zip(axes,['delta_hidden_separation','delta_logit_separation'],['A/B hidden separation','A/B centered-logit separation']):
  for arm,g in d.groupby('arm'):ax.scatter(g[metric],g.delta_ce_Mixed,c=COLORS[arm],marker=MARKERS[arm],s=36,label=arm,alpha=.8)
  ax.axhline(0,color='.5',lw=.8);ax.axvline(0,color='.5',lw=.8);ax.set(xlabel='Change in '+label+' (330−170)',title=label);ax.legend()
 axes[0].set_ylabel('Change in Mixed CE (330−170)');fig.suptitle('Mixed improvement and A/B separation change',fontsize=14);fig.text(.5,.015,'All60 matched continued endpoints. Separation is mean per-position Euclidean distance; no cohort is pooled before pairing.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.94));save(fig,'fig_3_transfer_vs_separation')
