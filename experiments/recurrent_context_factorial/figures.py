"""Table-only figure families. Primary learning figure precedes endpoint analysis."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
from .analysis import time_intervals
COL={'A':'#287cb8','B':'#ee882f','Mixed-average':'#3a9950','Mixed-both':'#009e9e','C':'#8c54a1','NoContext':'#777777','Familiar':'#333333'}
plt.rcParams.update({'font.size':9,'svg.hashsalt':'recurrent-context-factorial-v1','axes.spines.top':False,'axes.spines.right':False})

def validate_training(f):
 keys=['seed','arm','update'];expected=pd.MultiIndex.from_product([range(20),ARMS,range(1,331)],names=keys);a=pd.MultiIndex.from_frame(f[keys])
 if a.has_duplicates or set(a)!=set(expected):raise ValueError('All seed/update rows required')
 if any(r.training_cue!=training_cue(r.update) for r in f.itertuples()):raise ValueError('Wrong training cue schedule')

def band(ax,s,color,label=None,x='update',ls='-'):
 s=s.sort_values(x);ax.plot(s[x],s['mean'],color=color,lw=1.6,label=label,ls=ls);ax.fill_between(s[x].to_numpy(),s.ci_low.to_numpy(),s.ci_high.to_numpy(),color=color,alpha=.12,lw=0)
def individual(ax,f,x,y,color):
 for _,s in f.groupby('seed'):ax.plot(s[x],s[y],color=color,lw=.5,alpha=.10)
def finish(fig,dest,name):
 fig.tight_layout();dest.mkdir(parents=True,exist_ok=True);names=[]
 for ext in ['png','svg']:
  n=name+'.'+ext;fig.savefig(dest/n,dpi=160,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {'Software':'recurrent-context-factorial'});names.append(n)
 plt.close(fig);return names

def figure_a(dest=None):
 base=OUT/'part_a';dest=Path(dest) if dest else base/'figures';f=pd.read_csv(base/'tables/evaluations.csv');ci=pd.read_csv(base/'tables/evaluation_intervals.csv');fig,axs=plt.subplots(1,2,figsize=(13,4.5))
 for ax,arm in zip(axs,['Standard','Centered']):
  for cue in list(CUES)[:-1]:
   x=f[(f.arm==arm)&(f.cue==cue)];individual(ax,x,'update','ce',COL[cue]);band(ax,ci[(ci.arm==arm)&(ci.cue==cue)&(ci.metric=='ce')],COL[cue],cue,ls='--' if cue=='Mixed-both' else '-')
  ax.axhline(CHANCE,color='gray',ls=':',lw=.9);ax.set(title='Existing '+arm,xlabel='Frozen training state',ylabel='CE (nats; symlog)');ax.set_yscale('symlog',linthresh=.001);ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=8,ncol=2)
 fig.suptitle('Part A · two Mixed encodings · all 20 seeds, means and 95% seed CIs',y=1.01);return finish(fig,dest,'A_existing_mixed_encodings')

def primary(dest=None):
 base=OUT/'part_b';dest=Path(dest) if dest else base/'figures';f=pd.read_csv(base/'tables/training.csv');validate_training(f);ci=time_intervals(f,['arm'],'update',['ce_before']);path=base/'tables/training_intervals.csv'
 if not path.exists():ci.to_csv(path,index=False)
 files=[]
 for adjusted in [False,True]:
  fig,axs=plt.subplots(4,2,figsize=(13,12))
  for row,arm in enumerate(ARMS):
   for j,last in enumerate([330,100]):
    ax=axs[row,j]
    for start in range(1,last+1,10):
     end=min(start+9,last);cue=training_cue(start);x=f[(f.arm==arm)&f['update'].between(max(1,start-1),end)];individual(ax,x,'update','ce_before',COL[cue]);band(ax,ci[(ci.arm==arm)&ci['update'].between(max(1,start-1),end)],COL[cue],cue if start in [1,11] else None)
    ax.set(title=LABELS[arm]+f' · first {last}',xlabel='Sequence repetition / update',ylabel='Actual pre-update CE (nats)',xlim=(1,last));ax.legend(frameon=False,ncol=2)
    if adjusted:ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0)
  fig.suptitle('Figure 1 · A10/B10 actual loss · '+('symlog companion; all finite extremes retained' if adjusted else 'linear scale; all 20 seeds and 95% seed CIs'),y=1.003)
  files+=finish(fig,dest,'01_training_loss'+('_scale_companion' if adjusted else ''))
 return files

def remaining(dest=None):
 base=OUT/'part_b';dest=Path(dest) if dest else base/'figures';read=lambda n:pd.read_csv(base/'tables'/f'{n}.csv');f=read('evaluations');ci=read('evaluation_intervals');files=[]
 fig,axs=plt.subplots(4,2,figsize=(13,12))
 for row,arm in enumerate(ARMS):
  for j,last in enumerate([330,120]):
   ax=axs[row,j]
   for cue in CUES:
    x=f[(f.arm==arm)&(f.cue==cue)&(f['update']<=last)];individual(ax,x,'update','ce',COL[cue]);band(ax,ci[(ci.arm==arm)&(ci.cue==cue)&(ci.metric=='ce')&(ci['update']<=last)],COL[cue],cue,ls='--' if cue in ['NoContext','Mixed-both'] else '-')
   ax.axhline(CHANCE,color='gray',ls=':',lw=.9);ax.set(title=LABELS[arm]+f' · first {last}',xlabel='Frozen training state',ylabel='CE (nats; symlog)');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.legend(frameon=False,ncol=3,fontsize=7)
 fig.suptitle('Figure 2 · all six cue conditions · full and early time courses',y=1.003);files+=finish(fig,dest,'02_all_context_learning')
 b=read('block_transfer');bc=read('block_intervals');s=read('switch_costs');sc=read('switch_intervals');fig,axs=plt.subplots(3,4,figsize=(17,9))
 for col,arm in enumerate(ARMS):
  ax=axs[0,col]
  for direction,cue in [('A→B','B'),('B→A','A')]:
   individual(ax,s[(s.arm==arm)&(s.direction==direction)],'boundary','switch_cost',COL[cue]);band(ax,sc[(sc.arm==arm)&(sc.direction==direction)&(sc.metric=='switch_cost')],COL[cue],direction,x='boundary')
  ax.set_title(LABELS[arm]);ax.set_ylabel('Incoming − outgoing CE');ax.legend(frameon=False,fontsize=7)
  for row,cue in enumerate(['A','B'],1):
   ax=axs[row,col];ff=b[(b.arm==arm)&(b.practiced==cue)]
   for metric,color,label in [('practiced_gain',COL[cue],cue+' own gain'),('unpracticed_gain',COL['B' if cue=='A' else 'A'],'Cross gain')]:
    individual(ax,ff,'end',metric,color);band(ax,bc[(bc.arm==arm)&(bc.practiced==cue)&(bc.metric==metric)],color,label,x='end')
   floor=ff.groupby('end').floor_at_start.mean();ax.scatter(floor.index,np.zeros(len(floor)),s=3+12*floor,color='gray',alpha=.5);ax.set_ylabel('Block CE decrease');ax.legend(frameon=False,fontsize=7)
  for ax in axs[:,col]:ax.axhline(0,color='black',lw=.6);ax.set_yscale('symlog',linthresh=.01);ax.set_xlabel('State / block end')
 fig.suptitle('Figure 3 · signed switch costs and raw within/cross gains · gray zero dots: floor fraction',y=1.01);files+=finish(fig,dest,'03_switch_costs_and_transfer')
 g=read('geometry');gc=read('geometry_intervals');fig,axs=plt.subplots(5,4,figsize=(17,13))
 for col,arm in enumerate(ARMS):
  for row,metric,label,color in [(0,'ce','Baseline C CE',COL['C']),(1,'u_norm','||centered u||','#009e9e'),(2,'actual_avg_margin_change','Change in correct margin','#009e9e'),(3,'v_norm','||centered v||','#c47326'),(4,'recurrent_spectral_norm','Current ||Wh||₂','#444444')]:
   ax=axs[row,col]
   if row==0:
    x=f[(f.arm==arm)&(f.cue=='C')];ss=ci[(ci.arm==arm)&(ci.cue=='C')&(ci.metric=='ce')]
   else:x=g[g.arm==arm];ss=gc[(gc.arm==arm)&(gc.metric==metric)]
   individual(ax,x,'update',metric,color);band(ax,ss,color);ax.set(ylabel=label,xlabel='Training state')
   if row<4:ax.set_yscale('symlog',linthresh=.01)
   if row==2:
    band(ax,gc[(gc.arm==arm)&(gc.metric=='u_fixed_competitor_margin')],'gray','Fixed C competitor',ls='--');ax.axhline(0,color='black',lw=.6);ax.legend(frameon=True,framealpha=.8,edgecolor='none',fontsize=7,loc='upper left')
   if row==0:ax.set_title(LABELS[arm]);ax.set_ylim(bottom=0)
 fig.suptitle('Figure 4 · baseline learning, common/contrast response and recurrent dynamics · descriptive associations',y=1.003);files+=finish(fig,dest,'04_learned_computation')
 fig,axs=plt.subplots(4,4,figsize=(17,10))
 for col,arm in enumerate(ARMS):
  for row,metric,label in [(0,'g_norm','||g||'),(1,'d_norm','||d||'),(2,'hidden_separation','Hidden trajectory separation'),(3,'centered_logit_separation','Centered-logit separation')]:
   ax=axs[row,col];individual(ax,g[g.arm==arm],'update',metric,'#555555');band(ax,gc[(gc.arm==arm)&(gc.metric==metric)],'#555555');ax.set(ylabel=label,xlabel='Training state')
   if row>=2:ax.set_yscale('symlog',linthresh=.01)
   else:ax.set_ylim(bottom=0)
   if row==0:ax.set_title(LABELS[arm])
 fig.suptitle('Figure 4 companion · input contributions and initial versus retained geometry',y=1.005);files+=finish(fig,dest,'04_geometry_companion')
 ep=read('endpoints');summary=read('endpoint_summary');fig,axs=plt.subplots(2,2,figsize=(13,8))
 for ax,arm in zip(axs.flat,ARMS):
  for offset,cue in zip([-.24,-.08,.08,.24],['Familiar','Mixed-average','Mixed-both','C']):
   for j,k in enumerate([100,170,330]):
    if cue=='Familiar':values=ep[(ep.arm==arm)&(ep['update']==k)&(ep.cue=='C')].familiar_ce;metric='familiar_ce';c='C'
    else:values=ep[(ep.arm==arm)&(ep['update']==k)&(ep.cue==cue)].ce;metric='ce';c=cue
    ss=summary[(summary.arm==arm)&(summary['update']==k)&(summary.cue==c)&(summary.subgroup=='all')&(summary.metric==metric)].iloc[0];xs=j+offset;ax.scatter(xs+np.linspace(-.025,.025,len(values)),values,color=COL[cue],s=8,alpha=.2);ax.errorbar(xs,ss['mean'],yerr=[[max(0,ss['mean']-ss.ci_low)],[max(0,ss.ci_high-ss['mean'])]],fmt='o',color=COL[cue],markersize=4,capsize=2,label=cue if j==0 else None)
  ax.axhline(CHANCE,color='gray',ls=':',lw=.9);ax.set_xticks([0,1,2],[100,170,330]);ax.set(title=LABELS[arm],xlabel='Training update',ylabel='CE (nats; symlog)');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.legend(frameon=False,fontsize=8,ncol=2)
 fig.suptitle('Figure 5 · paired factorial arms · individual seeds, means and 95% seed CIs',y=1.01);files+=finish(fig,dest,'05_factorial_endpoints');return files

if __name__=='__main__':
 import sys
 if '--part-a' in sys.argv:figure_a()
 elif '--primary' in sys.argv:primary()
 else:remaining()
