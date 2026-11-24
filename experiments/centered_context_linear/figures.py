"""Five deterministic table-only figures; no fits or model evaluations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import OUT,ARMS,np,pd,Path
from .analysis import validate_grid
COL={'A':'#2677ba','B':'#ee8531','Mixed':'#389b59','C':'#8d53a2'}
plt.rcParams.update({'font.size':10,'svg.hashsalt':'centered-linear-v1','axes.spines.top':False,'axes.spines.right':False})
def validate_inputs(f):validate_grid(f)
def band(ax,s,color,label=None,x='update',style='-'):
 s=s.sort_values(x);ax.plot(s[x],s['mean'],color=color,lw=1.8,label=label,ls=style);ax.fill_between(s[x].to_numpy(),s.ci_low.to_numpy(),s.ci_high.to_numpy(),color=color,alpha=.17,lw=0)
def lines(ax,f,x,y,color):
 for _,s in f.groupby('seed'):ax.plot(s[x],s[y],color=color,alpha=.13,lw=.65)
def generate(source=OUT,destination=None):
 source=Path(source);dest=Path(destination) if destination else source/'figures';dest.mkdir(parents=True,exist_ok=True)
 read=lambda n:pd.read_csv(source/'tables'/f'{n}.csv');f=read('frozen_evaluations');validate_inputs(f);t=read('training_trajectories');ti=read('training_intervals');ei=read('evaluation_intervals');s=read('switch_costs');si=read('switch_intervals');b=read('block_transfer');bi=read('block_intervals');g=read('geometry');gi=read('geometry_intervals');names=[]
 def save(fig,n):
  fig.tight_layout()
  for ext in ['png','svg']:
   name=n+'.'+ext;fig.savefig(dest/name,dpi=170,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else {'Software':'centered-context-linear'});names.append(name)
  plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(12,7),sharey='row')
 for j,arm in enumerate(ARMS):
  for row,last in enumerate([330,100]):
   ax=axs[row,j]
   for start in range(1,last+1,10):
    end=min(start+9,last);cue='A' if (start-1)%20==0 else 'B';x=t[(t.arm==arm)&t['update'].between(max(1,start-1),end)];ss=ti[(ti.arm==arm)&ti['update'].between(max(1,start-1),end)];lines(ax,x,'update','ce_before',COL[cue]);band(ax,ss,COL[cue],cue if start in [1,11] else None)
   ax.set(title=f'{arm} — {last} updates',xlabel='Sequence repetition / update',ylabel='Practiced pre-update CE (nats)',xlim=(1,last));ax.legend(frameon=False)
 fig.suptitle('Actual A10/B10 training loss · all 20 seeds, mean and 95% seed-bootstrap CI',fontsize=12,y=1.01);save(fig,'01_actual_training_loss')
 fig,axs=plt.subplots(1,2,figsize=(12,4.4))
 for ax,arm in zip(axs,ARMS):
  for cue in COL:
   ff=f[(f.arm==arm)&(f.cue==cue)];ss=ei[(ei.arm==arm)&(ei.cue==cue)&(ei.metric=='ce')];lines(ax,ff,'update','ce',COL[cue]);band(ax,ss,COL[cue],cue,style='--' if cue=='C' else '-')
  ax.axhline(np.log(4),color='gray',ls=':',lw=1,label='ln(4)');ax.set(title=arm,xlabel='Frozen training state',ylabel='CE (nats)',yscale='symlog');ax.set_yscale('symlog',linthresh=.001);ax.set_ylim(bottom=0);ax.legend(frameon=False,ncol=3,fontsize=9)
 fig.suptitle('Frozen A/B/Mixed/C · Centered C and Mixed coincide · symlog CE axis',y=1.02);save(fig,'02_frozen_cue_learning')
 fig,axs=plt.subplots(2,2,figsize=(12,7))
 for j,arm in enumerate(ARMS):
  for row,direction in enumerate(['A→B','B→A']):
   ax=axs[row,j];ff=s[(s.arm==arm)&(s.direction==direction)];ss=si[(si.arm==arm)&(si.direction==direction)&(si.metric=='switch_cost')];lines(ax,ff,'boundary','switch_cost',COL['B' if direction=='A→B' else 'A']);band(ax,ss,COL['B' if direction=='A→B' else 'A'],'Raw same-weight cost',x='boundary');adj=si[(si.arm==arm)&(si.direction==direction)&(si.metric=='baseline_adjusted_cost')];band(ax,adj,'#666666','Initial-bias adjusted',x='boundary',style='--');ax.axhline(0,color='black',lw=.6);ax.set(title=f'{arm} · {direction}',xlabel='Boundary state (before incoming update)',ylabel='Incoming − outgoing CE');ax.set_yscale('symlog',linthresh=.01);ax.legend(frameon=False,fontsize=8)
 save(fig,'03_signed_switch_costs')
 fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
 for j,arm in enumerate(ARMS):
  for row,cue in enumerate(['A','B']):
   ax=axs[row,j];ff=b[(b.arm==arm)&(b.practiced==cue)]
   for metric,color,label in [('practiced_gain',COL[cue],'Practiced gain'),('unpracticed_gain',COL['B' if cue=='A' else 'A'],'Unpracticed gain')]:
    lines(ax,ff,'end',metric,color);band(ax,bi[(bi.arm==arm)&(bi.practiced==cue)&(bi.metric==metric)],color,label,x='end')
   inset=ax.inset_axes([.65,.53,.32,.33]);inset.scatter(ff.practiced_gain,ff.unpracticed_gain,s=4,alpha=.12,color=COL[cue]);lim=max(abs(ff.practiced_gain).max(),abs(ff.unpracticed_gain).max());inset.plot([-lim,lim],[-lim,lim],color='gray',lw=.6,ls=':');inset.set(title='Gain pairs',xlabel='Practiced',ylabel='Unpracticed');inset.tick_params(labelsize=6);inset.xaxis.label.set_size(6);inset.yaxis.label.set_size(6);inset.title.set_size(7)
   ax.axhline(0,color='black',lw=.6);ax.set(title=f'{arm} · {cue} blocks',xlabel='Block end update',ylabel='Start CE − end CE (nats)');ax.set_yscale('symlog',linthresh=.005);ax.legend(frameon=False,loc='lower left')
 fig.suptitle('Signed learning gains · negative cross-context transfer retained · no ratios',y=1.01);save(fig,'04_within_cross_context_gains')
 fig,axs=plt.subplots(3,2,figsize=(12,9),sharex=True)
 for j,arm in enumerate(ARMS):
  for row,metric,label in [(0,'hidden_separation','Mean positionwise hidden distance'),(1,'centered_logit_separation','Centered-logit distance')]:
   ax=axs[row,j];lines(ax,g[g.arm==arm],'update',metric,'#555555');band(ax,gi[(gi.arm==arm)&(gi.metric==metric)],'#555555');ax.set(title=arm if row==0 else '',ylabel=label)
  ax=axs[2,j]
  for cue in ['A','B']:
   ff=b[(b.arm==arm)&(b.practiced==cue)];lines(ax,ff,'end','specificity',COL[cue]);band(ax,bi[(bi.arm==arm)&(bi.practiced==cue)&(bi.metric=='specificity')],COL[cue],cue+' practice',x='end');floor=ff[ff.floor_at_start].groupby('end').size()/20;ax.scatter(floor.index,np.zeros(len(floor)),s=4+18*floor,color='gray',alpha=.6)
  ax.axhline(0,color='black',lw=.6);ax.set(ylabel='Practiced − unpracticed gain',xlabel='Training state / block end');ax.set_yscale('symlog',linthresh=.005);ax.legend(frameon=False,loc='lower left')
 fig.suptitle('Representation ≠ independent learning · gray zero markers: loss-floor blocks',y=1.01);save(fig,'05_representation_and_learning')
 return names
if __name__=='__main__':generate()
