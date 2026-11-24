"""One simple frozen-checkpoint A/B/Mixed figure, regenerated only from CSVs."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .evaluate import OUT,ARCHS,STEPS,CHANCE
NAME='A_B_Mixed_frozen_80_120_170';COLORS={'A':'#1f77b4','B':'#ff7f0e','Mixed':'#2ca02c'}

def validate(values):
 expected={(s,a,k,c) for s in range(20) for a in ARCHS for k in STEPS for c in COLORS}
 if values.duplicated(['seed','architecture','checkpoint','cue']).any() or set(values[['seed','architecture','checkpoint','cue']].itertuples(index=False,name=None))!=expected:raise ValueError('All20 seeds,3 frozen states,3 cues in both architectures required')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 if any((dest/(NAME+'.'+e)).exists() for e in ['png','svg']):raise FileExistsError('Use an empty destination')
 v=pd.read_csv(out/'tables/seed_values.csv');s=pd.read_csv(out/'tables/summary.csv');validate(v)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'mixed-checkpoint-curves-v1'})
 fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
 for ax,a in zip(axs,ARCHS):
  for cue,color in COLORS.items():
   f=v[(v.architecture==a)&(v.cue==cue)]
   for _,seed in f.groupby('seed'):
    seed=seed.sort_values('checkpoint');ax.plot(seed.checkpoint,seed.ce,color=color,lw=.65,alpha=.15)
   q=s[s.architecture==a].sort_values('checkpoint');mean=q[cue+'_mean'];ax.fill_between(q.checkpoint,q[cue+'_ci_low'],q[cue+'_ci_high'],color=color,alpha=.15,lw=0);ax.plot(q.checkpoint,mean,color=color,lw=2,marker='o',ms=5,label=cue)
  ax.axhline(CHANCE,color='#555555',ls='--',lw=1,label='Chance (1.386 nats)');ax.set_title(a);ax.set_xticks(STEPS);ax.set_xlim(75,175);ax.set_ylim(bottom=0);ax.set_xlabel('Training updates completed');ax.set_ylabel('Cross-entropy (mean over 7 targets)');ax.grid(axis='y',alpha=.13)
 handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4,frameon=False)
 fig.get_layout_engine().set(rect=(0,.06,1,.93));fig.text(.5,.012,'Faint lines: all20 seeds. Bold lines: cohort means; shaded bands: pointwise95% seed-bootstrap intervals.\nPhysical Mixed=(0.5,0.5,0). Frozen weights, identical tokens, no training; checkpoint170 reuses archived evaluations.',ha='center',fontsize=9)
 names=[]
 for ext in ['png','svg']:
  p=dest/(NAME+'.'+ext);fig.savefig(p,dpi=180,metadata={'Software':'mixed-checkpoint-curves-v1'} if ext=='png' else {'Date':None,'Creator':'mixed-checkpoint-curves-v1'});names.append(p.name)
 plt.close(fig);return names

if __name__=='__main__':print('\n'.join(generate()))
