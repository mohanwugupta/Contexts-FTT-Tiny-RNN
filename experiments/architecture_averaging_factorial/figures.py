"""Four deterministic figures, generated from saved tables only."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from .protocol import OUT,ARCHS,STEPS
from .statistics import bootstrap
COLORS={'FF-Linear':'#2166ac','FF-Tanh':'#b35806','RNN-Linear':'#1b7837','RNN-Tanh':'#762a83'}
NAMES=['fig_1_architecture_design','fig_2_learning_transfer','fig_3_midpoint_residuals','fig_4_recurrence_gap']


def validate_tables(q):
 expected={(s,a,k) for s in range(20) for a in ARCHS for k in STEPS}
 if q.duplicated(['seed','architecture','step']).any() or set(q[['seed','architecture','step']].itertuples(index=False,name=None))!=expected:raise ValueError('Four fixed checkpoints/all80 paired models required')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 for n in NAMES:
  for ext in ['png','svg']:
   if (dest/(n+'.'+ext)).exists():raise FileExistsError('Immutable figure destination')
 q=pd.read_csv(out/'seed_level.csv');validate_tables(q);g=pd.read_csv(out/'tables/geometry_by_position.csv');intervals=pd.read_csv(out/'tables/architecture_intervals.csv');paired=pd.read_csv(out/'tables/paired_contrasts.csv');pis=pd.read_csv(out/'tables/paired_intervals.csv');primary=q[q.step==1000]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'architecture-averaging-factorial-v1'})
 paths=[]
 def export(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext);fig.savefig(p,dpi=170,metadata={'Software':'architecture-averaging-factorial-v1'} if ext=='png' else {'Date':None,'Creator':'architecture-averaging-factorial-v1'});paths.append(str(p.relative_to(dest)))
  plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(13,7))
 for ax,arch in zip(axs.flat,ARCHS):
  ax.set_xlim(0,10);ax.set_ylim(0,5);ax.axis('off');color=COLORS[arch];ax.text(5,4.65,arch,ha='center',weight='bold',fontsize=15,color=color)
  boxes=[(.15,1.6,2.5,1.3,'History window\n3 × one-hot5 + cue3\n15 + 3 shared inputs'),(3.5,1.6,2.4,1.3,'8 hidden units\n'+('tanh' if arch.endswith('Tanh') else 'identity')),(7,1.6,2.7,1.3,'4 linear logits\nthen softmax')]
  for x,y,w,h,text in boxes:ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=1.5));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
  for x1,x2 in [(2.65,3.5),(5.9,7)]:ax.add_patch(FancyArrowPatch((x1,2.25),(x2,2.25),arrowstyle='-|>',mutation_scale=13,color=color))
  if arch.startswith('RNN'):
   ax.add_patch(FancyArrowPatch((5.6,2.95),(3.8,2.95),connectionstyle='arc3,rad=.7',arrowstyle='-|>',mutation_scale=13,color=color));ax.text(4.7,3.85,'W_h h_(t−1); reset h0=0',ha='center',fontsize=9)
  else:ax.text(4.7,3.45,'No recurrent matrix or feedback',ha='center',fontsize=9)
  ax.text(5,.8,'Train: whole A/B sequences alternate for1000 updates',ha='center',fontsize=10);ax.text(5,.3,'Evaluate: A, B, physical Mixed; NoContext/Novel secondary',ha='center',fontsize=9)
 fig.suptitle('2 × 2: recurrence × activation, with identical causal history inputs',fontsize=15);fig.text(.5,.015,'20 paired seeds × 4 architectures. Shared High-family initialization; recurrent spectral norm .9 initially only. New cohort, not canonical ablation.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[0])
 fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
 for ax,arch in zip(axs.flat,ARCHS):
  f=q[q.architecture==arch]
  for metric,color,style,label in [('familiar_ce','#2166ac','o-','Familiar A/B'),('Mixed_ce','#b35806','s--','Mixed')]:
   for _,trace in f.groupby('seed'):trace=trace.sort_values('step');ax.plot(trace.step,trace[metric],color=color,lw=.6,alpha=.13)
   s=intervals[(intervals.architecture==arch)&(intervals.qualification=='all')&(intervals.metric==metric)].sort_values('step');ax.plot(s.step,s['mean'],style,color=color,lw=2,ms=5,label=label);ax.fill_between(s.step,s.ci_low,s.ci_high,color=color,alpha=.15)
  ax.axhline(np.log(4),color='.4',ls=':',lw=1,label='Uniform1.386');ax.set_yscale('symlog',linthresh=1e-4);ax.set_xticks(STEPS);ax.set_title(arch,color=COLORS[arch],weight='bold');ax.grid(axis='y',alpha=.15)
 axs[0,0].legend(fontsize=9);axs[1,0].set_xlabel('Predetermined update');axs[1,1].set_xlabel('Predetermined update');axs[0,0].set_ylabel('Mean CE (nats; symmetric-log scale)');axs[1,0].set_ylabel('Mean CE (nats; symmetric-log scale)')
 fig.suptitle('Familiar learning and Mixed prediction across all fixed checkpoints',fontsize=15);fig.text(.5,.015,'Thin lines: every seed. Bands:95% whole-seed bootstrap intervals. Scale is linear within±1e−4; no loss values are clipped.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[1])
 fig,axs=plt.subplots(2,4,figsize=(15,7),sharex=True,sharey='row')
 for col,arch in enumerate(ARCHS):
  f=g[(g.architecture==arch)&(g.step==1000)]
  for row,metric,label in [(0,'hidden_midpoint_residual','Hidden midpoint residual norm'),(1,'centered_logit_midpoint_residual','Centered-logit midpoint residual norm')]:
   ax=axs[row,col]
   for _,s in f.groupby('seed'):ax.plot(s.transition,s[metric],color=COLORS[arch],lw=.6,alpha=.15)
   mean=[];lo=[];hi=[]
   for t in range(1,8):
    s=bootstrap(f[f.transition==t],metric);mean.append(s['mean']);lo.append(s['ci_low']);hi.append(s['ci_high'])
   ax.plot(range(1,8),mean,'o-',color=COLORS[arch],lw=2,ms=4);ax.fill_between(range(1,8),lo,hi,color=COLORS[arch],alpha=.18);ax.axhline(0,color='.5',lw=.7);ax.set_xticks(range(1,8));ax.grid(axis='y',alpha=.15)
   if col==0:ax.set_ylabel(label)
   if row==1:ax.set_xlabel('Sequence position')
  axs[0,col].set_title(arch,color=COLORS[arch],weight='bold')
  if arch.endswith('Linear'):axs[0,col].text(.5,.88,'Affine identity validated\nin double arithmetic',transform=axs[0,col].transAxes,ha='center',va='top',fontsize=9)
 fig.suptitle('Where averaging fails: update1000, identical tokens and constant contexts',fontsize=15);fig.text(.5,.015,'Residuals use actual float32 model outputs. Linear roundoff is near zero on these axes; no residual is interpreted as a fraction of CE caused.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[2])
 fig,axs=plt.subplots(1,3,figsize=(15,5.5))
 for j,arch in enumerate(ARCHS):
  f=primary[primary.architecture==arch].sort_values('seed');jitter=(np.arange(20)-9.5)*.012
  for ax,metric in [(axs[0],'Mixed_gap'),(axs[1],'familiar_ce')]:
   s=bootstrap(f,metric);ax.scatter(j+jitter,f[metric],s=25,facecolors='none',edgecolors=COLORS[arch],lw=1);ax.errorbar(j,s['mean'],yerr=[[s['mean']-s['ci_low']],[s['ci_high']-s['mean']]],fmt='o',color='black',capsize=4,lw=2,ms=5)
 for ax in axs[:2]:ax.set_xticks(range(4),ARCHS,rotation=20,ha='right');ax.grid(axis='y',alpha=.15)
 axs[0].axhline(0,color='.5',lw=.8);axs[0].set(ylabel='Mixed CE − familiar CE (nats)',title='Primary gap: all20 paired seeds')
 axs[1].set_yscale('symlog',linthresh=1e-5);axs[1].axhline(.5,color='.4',ls=':',lw=1);axs[1].set(ylabel='Familiar CE (nats; symmetric-log scale)',title='Acquisition alongside the gap')
 p=paired[paired.step==1000];s=pis[(pis.step==1000)&(pis.qualification=='all')]
 for j,metric in enumerate(['recurrence_tanh','interaction']):
  vals=p.sort_values('seed')[metric];ss=s[s.metric==metric].iloc[0];axs[2].scatter(j+(np.arange(20)-9.5)*.012,vals,facecolors='none',edgecolors='#333333',s=25);axs[2].errorbar(j,ss['mean'],yerr=[[ss['mean']-ss.ci_low],[ss.ci_high-ss['mean']]],fmt='o',color='black',capsize=4,lw=2)
 axs[2].axhline(0,color='.5',lw=.8);axs[2].set(xticks=[0,1],xticklabels=['RNN-Tanh\n− FF-Tanh','Recurrence × activation\ninteraction'],ylabel='Paired gap difference (nats)',title='Positive = recurrence worsens gap')
 fig.suptitle('Does recurrence worsen nonlinear generalization? Primary endpoint1000',fontsize=15);fig.text(.5,.015,'Dots: trained seeds. Black marks: means and paired seed-bootstrap95% intervals. All80 networks have perfect A/B accuracy at this endpoint.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.06,1,.94));export(fig,NAMES[3])
 return paths
