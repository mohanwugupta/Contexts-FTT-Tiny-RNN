"""Six figure families from saved tables only, whole-trained-seed intervals."""
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
CUCOL={'A':'#287cb8','B':'#ee882f','Mixed-average':'#3a9950','Mixed-both':'#009e9e','C-zero':'#8c54a1','C-random':'#c43a48'}
ARMCOL={'HI':'#777777','HC':'#b86f2c','NI':'#377eb8','NC':'#2b995d'}
STYLE={'font.size':9,'svg.hashsalt':'nonzero-context-v1','axes.spines.top':False,'axes.spines.right':False}
plt.rcParams.update(STYLE)

def validate_primary(f):
 keys=['seed','arm','update'];a=pd.MultiIndex.from_frame(f[keys]);expected=pd.MultiIndex.from_product([range(20),ARMS,range(331)],names=keys)
 if a.has_duplicates or len(a)!=len(expected) or set(a)!=set(expected):raise ValueError('All20 trained seeds andall331 states required')

def band(ax,f,color,label=None,x='update',ls='-'):
 f=f.sort_values(x);ax.plot(f[x],f['mean'],color=color,label=label,lw=1.65,ls=ls);ax.fill_between(f[x].to_numpy(),f.ci_low.to_numpy(),f.ci_high.to_numpy(),color=color,alpha=.14,lw=0)
def individual(ax,f,x,y,color):
 for _,s in f.groupby('seed'):ax.plot(s[x],s[y],color=color,lw=.55,alpha=.12)
def save(fig,dest,name):
 fig.tight_layout();names=[]
 for ext in ['png','svg']:
  n=name+'.'+ext;fig.savefig(dest/n,dpi=165,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {'Software':'nonzero-context-generalization'});names.append(n)
 plt.close(fig);return names

def generate(dest=None,regenerate_training=False):
 dest=Path(dest) if dest else OUT/'figures';dest.mkdir(parents=True,exist_ok=True);files=[]
 if regenerate_training:
  from experiments.recurrent_context_factorial.figures import primary
  with matplotlib.rc_context({'font.size':9,'svg.hashsalt':'recurrent-context-factorial-v1','axes.spines.top':False,'axes.spines.right':False}):files+=primary(dest)
  plt.rcParams.update(STYLE)
 else:
  for n in ['01_training_loss','01_training_loss_scale_companion']:
   for ext in ['png','svg']:
    name=n+'.'+ext
    if (dest/name).resolve()!=(BASE/'figures'/name).resolve():shutil.copyfile(BASE/'figures'/name,dest/name)
    files.append(name)
 read=lambda n:pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip');allseed=read('seed_trajectories');seed=read('primary_seed_trajectories');validate_primary(seed);curves=read('learning_curves');ci=read('learning_intervals');ep=read('seed_endpoints');summary=read('endpoint_summary');primaryci=read('primary_intervals');mechanism=read('primary_mechanism');mc=read('mechanism_intervals')
 for last,num,title in [(330,2,'Novel nonzero C over full training'),(120,3,'Novel nonzero C during early acquisition')]:
  fig,axs=plt.subplots(2,2,figsize=(13,8))
  for ax,arm in zip(axs.flat,ARMS):
   for cue,color in CUCOL.items():
    f=curves[(curves.arm==arm)&(curves.cue==cue)&(curves['update']<=last)];individual(ax,f,'update','ce',color);band(ax,ci[(ci.arm==arm)&(ci.cue==cue)&(ci['update']<=last)],color,cue)
   ax.axhline(CHANCE,color='gray',ls=':',lw=.9,label='ln(4)');ax.set(title=LABELS[arm],xlabel='Frozen training state',ylabel='CE (nats; symlog)',xlim=(0,last));ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.legend(frameon=False,ncol=3,fontsize=7)
  fig.suptitle(f'Figure {num} · {title} · C-random averaged over 20 probes within each seed',y=1.01);files+=save(fig,dest,f'0{num}_novel_C_'+('full' if last==330 else 'early'))
 fig,axs=plt.subplots(2,2,figsize=(12,8))
 for ax,arm in zip(axs.flat,ARMS):
  f=ep[(ep.arm==arm)&(ep.alpha==1)];individual(ax,f,'update','R',ARMCOL[arm]);ss=summary[(summary.arm==arm)&(summary.alpha==1)&(summary.subgroup=='all')&(summary.metric=='R')];band(ax,ss,ARMCOL[arm],'Mean ±95% seed CI');ax.scatter(f['update'],f.R,s=13,facecolor=np.where(f.strong_familiar,ARMCOL[arm],'white'),edgecolor=ARMCOL[arm],lw=.7,alpha=.65);ax.set(title=LABELS[arm],xlabel='Training checkpoint',ylabel='R:successful C vectors /20',ylim=(-.025,1.025));ax.set_xticks(PROBES);ax.legend(frameon=False,fontsize=8)
 fig.suptitle('Figure 4 · primary-scale novel-context robustness · filled points:familiar-qualified seeds',y=1.01);files+=save(fig,dest,'04_novel_robustness')
 vectors=pd.read_csv(OUT/'tables/primary_vector_trajectories.csv',float_precision='round_trip');vectors=vectors[vectors['update'].isin(PROBES)];fig,axs=plt.subplots(4,5,figsize=(17,10))
 for row,arm in enumerate(ARMS):
  for col,step in enumerate(PROBES):
   ax=axs[row,col];f=vectors[(vectors.arm==arm)&(vectors['update']==step)];jitter=(f.vector.to_numpy()-9.5)*.025;ax.scatter(f.seed+jitter,f.ce,s=5,c=np.where(f.strong_familiar,ARMCOL[arm],'#bcbcbc'),alpha=.45);med=f.groupby('seed').ce.median();ax.scatter(med.index,med,s=9,marker='_',c='black');n=int(f.drop_duplicates('seed').strong_familiar.sum());ax.set(title=f'{arm} · {step} · qualified {n}/20',xlabel='Trained seed',ylabel='C CE (nats; symlog)');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.set_xticks([0,5,10,15,19]);ax.axhline(CHANCE,color='gray',ls=':',lw=.6)
 fig.suptitle('Figure 4 companion · every probe within every seed · black ticks:within-seed median · not 400 independent networks',y=1.005);files+=save(fig,dest,'04_within_seed_probe_distributions')
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,step in zip(axs,[100,170,330]):
  f=ep[(ep['update']==step)&(ep.alpha==1)];wide=f.pivot(index='seed',columns='arm',values='R').reindex(columns=ARMS)
  for _,r in wide.iterrows():ax.plot(range(4),r,color='gray',lw=.7,alpha=.25)
  for j,arm in enumerate(ARMS):
   x=f[f.arm==arm].sort_values('seed');ss=summary[(summary.arm==arm)&(summary['update']==step)&(summary.alpha==1)&(summary.subgroup=='all')&(summary.metric=='R')].iloc[0];ax.scatter(j+np.linspace(-.045,.045,20),x.R,s=12,facecolors=np.where(x.strong_familiar,ARMCOL[arm],'white'),edgecolors=ARMCOL[arm],lw=.6);ax.errorbar(j,ss['mean'],yerr=[[max(0,ss['mean']-ss.ci_low)],[max(0,ss.ci_high-ss['mean'])]],fmt='s',color='black',capsize=3,markersize=4)
  ax.set_xticks(range(4),ARMS);ax.set(title=f'Checkpoint {step}',ylabel='Primary C robustness R',ylim=(-.025,1.025))
 fig.suptitle('Figure 4 paired companion · lines connect the same trained seed and same 20 novel vectors',y=1.02);files+=save(fig,dest,'04_paired_seed_robustness')
 fig,axs=plt.subplots(5,4,figsize=(17,13))
 for col,arm in enumerate(ARMS):
  ax=axs[0,col]
  for cue in ['C-zero','C-random']:
   f=curves[(curves.arm==arm)&(curves.cue==cue)];individual(ax,f,'update','ce',CUCOL[cue]);band(ax,ci[(ci.arm==arm)&(ci.cue==cue)],CUCOL[cue],cue)
  ax.set(title=LABELS[arm],ylabel='Baseline / novel CE');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=7)
  for row,metric,label,color in [(1,'K_centered_norm','Centered ||K_t||₂ (position mean)','#444444'),(2,'novel_contribution_norm','Mean centered ||K_t wC||','#c43a48'),(3,'margin_change','Correct-margin change','#c43a48'),(4,'recurrent_spectral_norm','Current ||Wh||₂','#555555')]:
   ax=axs[row,col];f=mechanism[mechanism.arm==arm];individual(ax,f,'update',metric,color);ss=(primaryci if metric in ['novel_contribution_norm','margin_change'] else mc);band(ax,ss[(ss.arm==arm)&(ss.metric==metric)],color);ax.set(ylabel=label)
   if row==3:ax.set_yscale('symlog',linthresh=.01)
   if row in [1,2]:ax.set_ylim(bottom=0)
   if row==3:ax.axhline(0,color='black',lw=.6)
  for ax in axs[:,col]:ax.set_xlabel('Training state')
 fig.suptitle('Figure 5 · baseline access and novel-input sensitivity · norms alone do not establish prediction benefit',y=1.003);files+=save(fig,dest,'05_robustness_mechanism')
 fig,axs=plt.subplots(4,2,figsize=(12,11));tc={100:'#377eb8',170:'#e2852d',330:'#299358'}
 for row,arm in enumerate(ARMS):
  for col,metric in enumerate(['mean_ce','R']):
   ax=axs[row,col]
   for step,color in tc.items():
    f=allseed[(allseed.arm==arm)&(allseed['update']==step)];individual(ax,f,'alpha',metric,color);ss=summary[(summary.arm==arm)&(summary['update']==step)&(summary.subgroup=='all')&(summary.metric==metric)];band(ax,ss,color,str(step),x='alpha')
   ax.axvline(1,color='gray',ls=':',lw=.8);ax.set(title=LABELS[arm],xlabel='Input multiplier α (primary=1)',ylabel='C CE (nats)' if col==0 else 'Successful contexts /20');ax.set_xticks(SCALES);ax.legend(title='Checkpoint',frameon=False,fontsize=8)
   if col==0:ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0)
   else:ax.set_ylim(-.025,1.025)
 fig.suptitle('Figure 6 · prespecified magnitude response · identical directions at every α',y=1.005);files+=save(fig,dest,'06_magnitude_response');return files

if __name__=='__main__':generate()
