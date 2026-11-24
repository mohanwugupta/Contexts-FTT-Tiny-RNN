"""Five deterministic, saved-table-only figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .design import OUT,ARCHS,DIAG,ENDPOINTS
from .measures import bootstrap
COLORS={'RNN-Linear':'#2166ac','RNN-Tanh':'#b35806'}
NAMES=['fig_1_learning_switches','fig_2_learning_transfer','fig_3_geometry_and_transfer','fig_4_mixed_generalization','fig_5_midpoint_breakdown']


def validate_endpoints(q):
 expected={(s,a,k) for s in range(20) for a in ARCHS for k in ENDPOINTS}
 if q.duplicated(['seed','architecture','endpoint']).any() or set(q[['seed','architecture','endpoint']].itertuples(index=False,name=None))!=expected:raise ValueError('All40 models at both fixed endpoints required')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 for name in NAMES:
  for ext in ['png','svg']:
   if (dest/(name+'.'+ext)).exists():raise FileExistsError('Immutable figure destination')
 q=pd.read_csv(out/'tables/endpoint_summary.csv');validate_endpoints(q);l=pd.read_csv(out/'learning_trajectories.csv');w=pd.read_csv(out/'tables/direct_specificity.csv');di=pd.read_csv(out/'tables/diagnostic_intervals.csv');ci=pd.read_csv(out/'tables/endpoint_intervals.csv');pca=pd.read_csv(out/'tables/pca_trajectories.csv');res=pd.read_csv(out/'tables/midpoint_residuals.csv');d1=pd.read_csv(out/'tables/tanh_D1_decomposition.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'recurrent-learning-transfer-v1'})
 paths=[]
 def export(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext);fig.savefig(p,dpi=170,metadata={'Software':'recurrent-learning-transfer-v1'} if ext=='png' else {'Date':None,'Creator':'recurrent-learning-transfer-v1'});paths.append(str(p.relative_to(dest)))
  plt.close(fig)
 def dots(ax,g,metric,x,color):
  g=g.sort_values('seed');ax.scatter(x+(g.seed.to_numpy()-9.5)*.014,g[metric],s=22,facecolors='none',edgecolors=color,lw=.8);s=bootstrap(g,metric);ax.errorbar(x,s['mean'],yerr=[[s['mean']-s['ci_low']],[s['ci_high']-s['mean']]],fmt='o',color='black',capsize=4,lw=2,ms=5)
 fig,axs=plt.subplots(2,1,figsize=(13,8),sharex=True)
 for ax,arch in zip(axs,ARCHS):
  f=l[l.architecture==arch]
  for k in range(0,330,20):ax.axvspan(k,k+10,color='#2166ac',alpha=.035)
  for k in range(10,330,10):ax.axvline(k,color='.65',lw=.5,ls=':')
  for cue,color,style in [('A','#2166ac','-'),('B','#b35806','--')]:
   ff=f[f.cue==cue];pivot=ff.pivot(index='seed',columns='update',values='ce').sort_index();values=pivot.to_numpy();draw=np.random.default_rng(20260925).integers(0,20,(10000,20));weights=np.stack([np.bincount(row,minlength=20) for row in draw]);means=(weights@values)/20;lo,hi=np.quantile(means,[.025,.975],axis=0)
   for row in values:ax.plot(pivot.columns,row,color=color,alpha=.08,lw=.5)
   ax.plot(pivot.columns,values.mean(0),style,color=color,lw=2,label='Frozen '+cue+' CE');ax.fill_between(pivot.columns,lo,hi,color=color,alpha=.14)
  ax.axvline(170,color='black',lw=1.2,ls='--',label='Primary170');ax.set_yscale('symlog',linthresh=.001);ax.set(ylabel='CE (nats; symmetric-log)',title=arch);ax.legend(fontsize=8,ncol=3);ax.grid(axis='y',alpha=.15)
 axs[1].set(xlabel='Scientific updates completed; A/B evaluated before the next update',xlim=(0,330));fig.suptitle('Learning and every A10/B10 context switch',fontsize=15);fig.text(.5,.015,'Thin lines:20 seeds per architecture. Bands:95% seed-bootstrap intervals. Dotted lines mark all32 executed switches; final330 is end-censored.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.95));export(fig,NAMES[0])
 fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey='row')
 for col,arch in enumerate(ARCHS):
  for row,(train,within,cross) in enumerate([('A','delta_A_A','delta_B_A'),('B','delta_B_B','delta_A_B')]):
   ax=axs[row,col];f=w[w.architecture==arch]
   for metric,color,style,label in [(within,'#2166ac','o-','Within-context gain'),(cross,'#b35806','s--','Cross-context gain')]:
    for _,t in f.groupby('seed'):ax.plot(t.checkpoint,t[metric],color=color,alpha=.1,lw=.6)
    s=di[(di.architecture==arch)&(di.metric==metric)].sort_values('checkpoint');ax.plot(s.checkpoint,s['mean'],style,color=color,lw=2,ms=4,label=label);ax.fill_between(s.checkpoint,s.ci_low,s.ci_high,color=color,alpha=.15)
   ax.axhline(0,color='.5',lw=.8);ax.axvline(170,color='black',ls=':',lw=1);ax.set_xscale('symlog',linthresh=20);ax.set_yscale('symlog',linthresh=.0001);ax.set_xticks(DIAG,[str(k) for k in DIAG],fontsize=8);ax.set(title=arch+': diagnostic '+train+' update',ylabel='Signed CE improvement (nats)');ax.grid(axis='y',alpha=.15)
   if row==1:ax.set_xlabel('Frozen scientific checkpoint (170 primary)')
 axs[0,0].legend(fontsize=9);fig.suptitle('Actual one-update learning transfer with restored Adam moments',fontsize=15);fig.text(.5,.015,'Every branch starts from the same model+policy+optimizer state.20 seed curves/bands; clones never modify the scientific trajectory.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[1])
 fig,axs=plt.subplots(2,2,figsize=(12,9))
 for ax,arch in zip(axs[0],ARCHS):
  f=pca[(pca.architecture==arch)&(pca.endpoint==170)&(pca.seed==0)]
  for cue,color,marker,style in [('A','#2166ac','o','-'),('B','#b35806','s','--')]:
   t=f[f.cue==cue].sort_values('transition');ax.plot(t.PC1,t.PC2,style,marker=marker,color=color,label=cue)
   for r in t.itertuples():ax.annotate(str(r.transition),(r.PC1,r.PC2),xytext=(4,3),textcoords='offset points',fontsize=8)
  ax.set(xlabel='Familiar-only PC1',ylabel='Familiar-only PC2',title=arch+': seed0, update170 (fixed illustration)');ax.legend();ax.axis('equal')
 primary=q[q.endpoint==170]
 for ax,metric,label in [(axs[1,0],'hidden_centroid_distance','A/B hidden-centroid distance'),(axs[1,1],'centered_logit_separation','A/B centered-logit separation')]:
  for arch,f in primary.groupby('architecture'):ax.scatter(f[metric],f.mean_specificity,color=COLORS[arch],marker='o' if arch=='RNN-Linear' else 's',s=32,label=arch,alpha=.85)
  ax.axhline(0,color='.5',lw=.8);ax.set_yscale('symlog',linthresh=.0001);ax.set(xlabel=label,ylabel='Mean own−cross one-update gain (nats)',title='All20 seeds per architecture; geometry is secondary');ax.legend(fontsize=8)
 fig.suptitle('Separated states are not evidence of separate learning',fontsize=15);fig.text(.5,.015,'PCA is fit independently within each model using14 familiar states only. No geometry enters the functional-splitting classifier.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[2])
 fig,axs=plt.subplots(2,3,figsize=(14,8),sharey='col')
 for row,k in enumerate(ENDPOINTS):
  for col,metric,title in [(0,'familiar_ce','Familiar A/B CE'),(1,'Mixed_ce','Absolute Mixed CE'),(2,'Mixed_gap','Mixed−familiar gap')]:
   ax=axs[row,col]
   for j,arch in enumerate(ARCHS):dots(ax,q[(q.endpoint==k)&(q.architecture==arch)],metric,j,COLORS[arch])
   ax.set_xticks([0,1],ARCHS);ax.set(ylabel='Nats',title=('PRIMARY170: ' if k==170 else 'Secondary330: ')+title);ax.grid(axis='y',alpha=.15)
   if col<2:ax.axhline(np.log(4),color='.5',ls=':',lw=.8)
   else:ax.axhline(0,color='.5',lw=.8)
   if col==0:ax.set_yscale('symlog',linthresh=.001)
 fig.suptitle('Generalization after freezing every training trajectory and checkpoint',fontsize=15);fig.text(.5,.015,'Open dots: every trained seed; black marks: means and paired-seed95% intervals. Uniform reference1.386 shown for absolute losses.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[3])
 fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey='row')
 for col,arch in enumerate(ARCHS):
  for row,metric,label in [(0,'hidden_midpoint_residual','Hidden residual norm'),(1,'centered_logit_midpoint_residual','Centered-logit residual norm')]:
   ax=axs[row,col]
   for k,color,style in [(170,'#2166ac','o-'),(330,'#b35806','s--')]:
    f=res[(res.architecture==arch)&(res.endpoint==k)];mean=[];lo=[];hi=[]
    for _,t in f.groupby('seed'):ax.plot(t.transition,t[metric],color=color,alpha=.08,lw=.5)
    for t in range(1,8):
     s=bootstrap(f[f.transition==t],metric);mean.append(s['mean']);lo.append(s['ci_low']);hi.append(s['ci_high'])
    ax.plot(range(1,8),mean,style,color=color,lw=2,ms=4,label=str(k));ax.fill_between(range(1,8),lo,hi,color=color,alpha=.12)
   ax.axhline(0,color='.6',lw=.7);ax.set_xticks(range(1,8));ax.set_title(arch);ax.set_ylabel(label);ax.legend(title='Endpoint',fontsize=8)
 for row,(metrics,label) in enumerate([(['j_hidden_norm','r_hidden_norm'],'D1 hidden component norms'),(['j_task_norm','r_task_norm'],'D1 output component norms')]):
  ax=axs[row,2];f=d1[d1.endpoint==170]
  if not f.Torch_source_linked.eq(True).all():ax.text(.5,.5,'Adapter/source linkage blocked\nSee full audit table',ha='center',transform=ax.transAxes)
  else:
   for metric,color,style,lab in [(metrics[0],'#2166ac','o-','Instant mismatch j'),(metrics[1],'#b35806','s--','Prior propagation r')]:
    mean=[];lo=[];hi=[]
    for t in range(1,8):
     s=bootstrap(f[f.transition==t],metric);mean.append(s['mean']);lo.append(s['ci_low']);hi.append(s['ci_high'])
    ax.plot(range(1,8),mean,style,color=color,lw=2,ms=4,label=lab);ax.fill_between(range(1,8),lo,hi,color=color,alpha=.14)
  ax.set_title('Imported D1: Tanh, primary170');ax.set_ylabel(label);ax.set_xticks(range(1,8));ax.legend(fontsize=8)
 for ax in axs[1]:ax.set_xlabel('Sequence position')
 fig.suptitle('Where averaging breaks: source residuals and audited D1 components',fontsize=15);fig.text(.5,.015,'Linear identities are validated in double arithmetic; float32 traces include bounded roundoff. Component norms are not additive or percentages of CE.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,NAMES[4])
 return paths
