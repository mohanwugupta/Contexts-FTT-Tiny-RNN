"""Simple unsmoothed SRN curves; table-only deterministic exports."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .analysis import OUT,ARCHS
COLORS={'A':'#1f77b4','B':'#ff7f0e'}
NAMES=['figure_1_actual_training_loss','figure_2_absolute_AB_gap','figure_3_A_to_B_pooling']

def validate_training(t):
 expected={(s,a,r) for s in range(20) for a in ARCHS for r in range(1,331)}
 if t.duplicated(['seed','architecture','repetition']).any() or set(t[['seed','architecture','repetition']].itertuples(index=False,name=None))!=expected:raise ValueError('All20 seeds and330 actual repetitions required')
 if not np.array_equal(t.context.to_numpy(),np.where(((t.repetition-1)//10)%2==0,'A','B')):raise ValueError('Wrong context schedule')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 if any((dest/(n+'.'+e)).exists() for n in NAMES for e in ['png','svg']):raise FileExistsError('Use a fresh figure destination')
 read=lambda n:pd.read_csv(out/'tables'/n);t=read('figure_1_training_values.csv');validate_training(t);ts=read('figure_1_training_summary.csv');g=read('figure_2_gap_values.csv');gs=read('figure_2_gap_summary.csv');p=read('figure_3_pooling_values.csv');ps=read('figure_3_pooling_summary.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'original-learning-curves-v1'})
 paths=[]
 def export(fig,name):
  for ext in ['png','svg']:
   q=dest/(name+'.'+ext);fig.savefig(q,dpi=180,metadata={'Software':'original-learning-curves-v1'} if ext=='png' else {'Date':None,'Creator':'original-learning-curves-v1'});paths.append(q.name)
  plt.close(fig)
 def frame(ax,xlabel,ylabel):ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.15)
 fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for col,a in enumerate(ARCHS):
  for row,limit in [(0,330),(1,100)]:
   ax=axes[row,col];f=t[(t.architecture==a)&(t.repetition<=limit)];s=ts[(ts.architecture==a)&(ts.repetition<=limit)].sort_values('repetition')
   # Every direct switch jump remains a line segment, without cross-boundary smoothing.
   for _,seed in f.groupby('seed'):
    seed=seed.sort_values('repetition');ax.plot(seed.repetition,seed.ce_before,color='#555555',alpha=.11,lw=.6,zorder=1)
   for context,color in COLORS.items():
    observations=f[f.context==context];ax.scatter(observations.repetition,observations.ce_before,color=color,s=3,alpha=.10,lw=0,zorder=1)
   ax.plot(s.repetition,s['mean'],color='#555555',lw=1.2,zorder=3)
   for end in range(10,limit+1,10):
    start=end-9;context='A' if (end//10)%2 else 'B';part=s[s.repetition.between(start,end)];color=COLORS[context]
    ax.fill_between(part.repetition,part.ci_low,part.ci_high,color=color,alpha=.18,lw=0,zorder=2);ax.plot(part.repetition,part['mean'],color=color,lw=1.8,marker='o',ms=2.6,zorder=4)
   ax.set_xlim(1,limit);ax.set_ylim(bottom=0);ax.set_title(a+(' — full training' if row==0 else ' — first100 repetitions'));frame(ax,'Sequence repetition / update','Pre-update CE (mean of 7 targets)')
 fig.legend(handles=[Line2D([0],[0],color=c,marker='o',lw=2,label='Context '+k) for k,c in COLORS.items()],loc='outside upper center',ncol=2,frameon=False)
 fig.get_layout_engine().set(rect=(0,.045,1,.96));fig.text(.5,.01,'Faint lines: all20 seeds. Colored observations: practiced context only. Mean and pointwise95% seed-bootstrap intervals.\nA1–10 → B11–20 → A21–30; direct switch jumps are retained. Loss is measured before each update.',ha='center',fontsize=9);export(fig,NAMES[0])
 fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for col,a in enumerate(ARCHS):
  for row,metric,title in [(0,'gap_AB','Absolute A/B gap'),(1,'signed_B_minus_A','Signed diagnostic: CE_B − CE_A')]:
   ax=axes[row,col];f=g[g.architecture==a];s=gs[(gs.architecture==a)&(gs.metric==metric)].sort_values('state')
   for _,seed in f.groupby('seed'):ax.plot(seed.state,seed[metric],color='#555555',alpha=.14,lw=.65)
   ax.fill_between(s.state,s.ci_low,s.ci_high,color='#555555',alpha=.16,lw=0);ax.plot(s.state,s['mean'],color='#333333',lw=1.8)
   at=s[s.executed_A_to_B_switch];ax.scatter(at.state,at['mean'],s=22,color=COLORS['B'],zorder=4,label='A→B boundary (10,30,…,310)')
   last=s[s.end_censored];ax.scatter(last.state,last['mean'],s=30,facecolors='none',edgecolors=COLORS['B'],zorder=4,label='330: no next B block')
   ax.axhline(0,color='#777777',lw=.65);ax.set_xlim(0,334);ax.set_title(a+' — '+title);frame(ax,'Frozen state: updates completed','CE difference (nats)')
   if row==0:ax.set_ylim(bottom=0)
   if col==0 and row==0:ax.legend(fontsize=8,frameon=False)
 fig.get_layout_engine().set(rect=(0,.045,1,.94));fig.text(.5,.01,'A and B are evaluated at identical weights at every state. Absolute differences are computed within each seed before averaging.\nPositive signed values indicate a same-weight A→B cost. Faint lines:20 seeds; bands: pointwise95% seed-bootstrap intervals.',ha='center',fontsize=9);export(fig,NAMES[1])
 fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
 for ax,a in zip(axes,ARCHS):
  f=p[p.architecture==a]
  for metric,context,label in [('gain_A','A','A improvement (practiced)'),('gain_B','B','B improvement (transfer)')]:
   color=COLORS[context];s=ps[(ps.architecture==a)&(ps.metric==metric)].sort_values('end')
   for _,seed in f.groupby('seed'):ax.plot(seed.end,seed[metric],color=color,alpha=.14,lw=.65)
   ax.fill_between(s.end,s.ci_low,s.ci_high,color=color,alpha=.15,lw=0);ax.plot(s.end,s['mean'],color=color,lw=1.8,marker='o',ms=4,label=label)
  ax.axhline(0,color='#777777',lw=.7);ax.set_xlim(0,334);ax.set_xticks([10,50,100,150,200,250,300,330]);ax.set_title(a);frame(ax,'End of A block (10 repetitions each)','CE decrease over the A block (nats)');ax.legend(fontsize=9,frameon=False)
 fig.get_layout_engine().set(rect=(0,.14,1,.75));fig.suptitle('How much A-block learning also benefits B?',fontsize=15);fig.text(.5,.015,'Raw gains: loss at block start minus loss at block end, using paired frozen A/B evaluations. No ratios.\nB receives no training within these A blocks. Negative gains are retained; faint lines:20 seeds; bands:95% seed-bootstrap intervals.',ha='center',fontsize=9);export(fig,NAMES[2]);return paths

if __name__=='__main__':print('\n'.join(generate()))
