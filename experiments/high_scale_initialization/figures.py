"""Five deterministic table-only figures, retaining every registered seed."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from .core import OUT,ARCHS,PROBES,CHANCE
NAMES=['figure_1_alternating_learning','figure_2_AB_gap','figure_3_cross_context_pooling','figure_4_Mixed_learning_stages','figure_5_initialization_generalization'];BLUE='#1f77b4';ORANGE='#ff7f0e';GREEN='#2ca02c'

def validate_endpoints(q):
 if q.duplicated(['seed','architecture','checkpoint']).any() or set(q[['seed','architecture','checkpoint']].itertuples(index=False,name=None))!={(s,a,k) for s in range(20) for a in ARCHS for k in PROBES}:raise ValueError('All40 networks at all5 fixed checkpoints required, including missing cells')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 if any((dest/(n+'.'+e)).exists() for n in NAMES for e in ['png','svg']):raise FileExistsError('Fresh destination required')
 read=lambda n:pd.read_csv(out/n);t=read('training_measurements.csv');ti=read('tables/training_intervals.csv');gap=read('tables/AB_gap.csv');gi=read('tables/gap_intervals.csv');b=read('splitting.csv');bi=read('tables/block_intervals.csv');q=read('tables/endpoint_seed_summary.csv');validate_endpoints(q);ci=read('tables/endpoint_intervals.csv');comp=read('tables/initialization_comparison.csv');cc=read('tables/comparison_intervals.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'high-scale-initialization-v1'});paths=[]
 def export(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext);fig.savefig(p,dpi=170,metadata={'Software':'high-scale-initialization-v1'} if ext=='png' else {'Date':None,'Creator':'high-scale-initialization-v1'});paths.append(p.name)
  plt.close(fig)
 def line(ax,f,c,x,y,color,label,style='-'):
  for _,seed in f.groupby('seed'):
   seed=seed.sort_values(x);ax.plot(seed[x],seed[y],color=color,alpha=.13,lw=.65,ls=style)
  c=c.sort_values(x);ax.fill_between(c[x],c.ci_low,c.ci_high,color=color,alpha=.16,lw=0);ax.plot(c[x],c['mean'],color=color,lw=1.8,marker='o' if len(c)<40 else None,ms=3,label=label,ls=style)
 def axis(ax,x,y):ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(axis='y',alpha=.13)
 fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for j,a in enumerate(ARCHS):
  for row,limit in [(0,330),(1,100)]:
   ax=axs[row,j];f=t[(t.architecture==a)&(t['update']<=limit)];c=ti[(ti.architecture==a)&(ti['update']<=limit)].sort_values('update')
   for _,seed in f.groupby('seed'):ax.plot(seed['update'],seed.ce_before,color='#555555',alpha=.13,lw=.6)
   ax.plot(c['update'],c['mean'],color='#555555',lw=1)
   for end in range(10,limit+1,10):
    part=c[c['update'].between(end-9,end)];color=BLUE if end%20==10 else ORANGE;ax.fill_between(part['update'],part.ci_low,part.ci_high,color=color,alpha=.18,lw=0);ax.plot(part['update'],part['mean'],color=color,lw=1.8,marker='o',ms=2)
   ax.set_xlim(1,limit);ax.set_ylim(bottom=0);ax.set_title(a+(' — full330' if row==0 else ' — first100'));axis(ax,'Sequence repetition / update','Actual pre-update CE (raw nats)')
 fig.legend(handles=[Line2D([0],[0],color=BLUE,label='A'),Line2D([0],[0],color=ORANGE,label='B')],loc='outside upper center',ncol=2,frameon=False);fig.get_layout_engine().set(rect=(0,.05,1,.93));fig.text(.5,.012,'All20 seeds faint; colored mean/95% seed intervals. Direct switch jumps retained. Linear raw losses span a much larger range.\nOriginal unnormalized Normal(0,2) recurrent weights; no rescaling or rescue.',ha='center',fontsize=9);export(fig,NAMES[0])
 fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for j,a in enumerate(ARCHS):
  for row,metric,label in [(0,'absolute_gap','|CE_B − CE_A|'),(1,'signed_B_minus_A','CE_B − CE_A (signed)')]:
   ax=axs[row,j];f=gap[gap.architecture==a];c=gi[(gi.architecture==a)&(gi.metric==metric)];line(ax,f,c,'update',metric,'#555555',label)
   ab=c[(c['update']%20==10)&(c['update']<330)];ax.scatter(ab['update'],ab['mean'],color=ORANGE,s=15,zorder=5);last=c[c['update']==330];ax.scatter(last['update'],last['mean'],facecolors='none',edgecolors=ORANGE,s=25);ax.axhline(0,color='#777777',lw=.65);ax.set_title(a+' — '+label);axis(ax,'Frozen state: updates completed','CE difference (nats)')
   if row==0:ax.set_ylim(bottom=0)
 fig.get_layout_engine().set(rect=(0,.05,1,.84));fig.text(.5,.012,'Paired same-weight A/B evaluation at every state; absolute gaps computed within seed before averaging.\nOrange: A→B boundaries10,30,…310; hollow330 has no following B block. All20 seeds and95% intervals.',ha='center',fontsize=9);export(fig,NAMES[1])
 fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for j,a in enumerate(ARCHS):
  for row,practiced in [(0,'A'),(1,'B')]:
   ax=axs[row,j];f=b[(b.architecture==a)&(b.practiced==practiced)];ends=set(f.end)
   for metric,color,label in [('own',BLUE if practiced=='A' else ORANGE,'Practiced '+practiced),('cross',ORANGE if practiced=='A' else BLUE,'Other '+('B' if practiced=='A' else 'A'))]:line(ax,f,bi[(bi.architecture==a)&(bi.end.isin(ends))&(bi.metric==metric)],'end',metric,color,label)
   ax.axhline(0,color='#777777',lw=.7);ax.set_title(a+' — '+practiced+' blocks');axis(ax,'Block end: updates completed','CE decrease over10 updates (nats)');ax.legend(fontsize=8,frameon=False)
   if a=='RNN-Linear':ax.set_yscale('symlog',linthresh=.01);ax.set_ylabel('CE decrease (symlog; signed nats)')
 fig.get_layout_engine().set(rect=(0,.05,1,.84));fig.text(.5,.012,'Raw gains, not ratios: loss at block start minus block end. Negative transfer retained. All20 seeds/pointwise95% intervals.\nLinear uses a signed log scale to retain very large gains and losses without clipping.',ha='center',fontsize=9);export(fig,NAMES[2])
 fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
 for ax,a in zip(axs,ARCHS):
  for metric,color,label in [('A_ce',BLUE,'A'),('B_ce',ORANGE,'B'),('Mixed_ce',GREEN,'Mixed')]:line(ax,q[q.architecture==a],ci[(ci.architecture==a)&(ci.metric==metric)],'checkpoint',metric,color,label)
  ax.axhline(CHANCE,color='#555555',ls='--',lw=1,label='Chance1.386');ax.set_xticks(PROBES);ax.set_title(a);axis(ax,'Training updates completed','CE (mean over7 targets)');ax.legend(fontsize=8,frameon=False)
  if a=='RNN-Linear':ax.set_yscale('symlog',linthresh=.01);ax.set_ylabel('CE (symlog; zero retained)')
  ax.set_ylim(bottom=0)
 fig.get_layout_engine().set(rect=(0,.06,1,.92));fig.text(.5,.012,'All20 seeds; cohort means and95% seed intervals. Primary170, predetermined secondary330; earlier states descriptive.\nAllcheckpoints fixed before training; no selection on Mixed. Linear uses symlog for its very large dynamic range.',ha='center',fontsize=9);export(fig,NAMES[3])
 groups=['Normalized RNN-Linear','Unnormalized RNN-Linear','Normalized RNN-Tanh','Unnormalized RNN-Tanh','Canonical Tanh reference'];labels=['Norm\nLinear','Unnorm\nLinear','Norm\nTanh','Unnorm\nTanh','Canonical\nreference'];xx=[0,1,2,3,4.8];fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
 for ax,metric,title in zip(axs,['familiar_ce','Mixed_ce'],['Familiar A/B CE','Mixed CE']):
  for x,group,label in zip(xx,groups,labels):
   f=comp[(comp.group==group)&(comp.checkpoint==170)].sort_values('seed');c=cc[(cc.group==group)&(cc.checkpoint==170)&(cc.metric==metric)].iloc[0];color=BLUE if 'Linear' in group else GREEN if 'reference' in group else ORANGE
   ax.scatter(x+(f.seed-9.5)*.025,f[metric],s=20,facecolors='none',edgecolors=color,alpha=.65);ax.errorbar(x,c['mean'],yerr=[[c['mean']-c.ci_low],[c.ci_high-c['mean']]],fmt='o',color='black',capsize=4,ms=4,lw=1.6)
  ax.axvline(3.9,color='grey',ls=':',lw=1);ax.axhline(CHANCE,color='grey',ls='--',lw=1);ax.set_xticks(xx,labels);ax.set_yscale('symlog',linthresh=.0001);ax.set_ylim(bottom=0);ax.set_title(title+' — primary170');axis(ax,'Initialization/model group','CE (symlog; zero retained)')
 fig.get_layout_engine().set(rect=(0,.10,1,.84));fig.text(.5,.015,'All20 seed values per group; means/95% intervals. Canonical reference is separated and NOT an independent arm.\nNew Tanh170 exactly reproduces canonical tensors/training; slight evaluator rounding may differ. Normalized/new comparisons are unpaired descriptive.',ha='center',fontsize=9);export(fig,NAMES[4]);return paths

if __name__=='__main__':print('\n'.join(generate()))
