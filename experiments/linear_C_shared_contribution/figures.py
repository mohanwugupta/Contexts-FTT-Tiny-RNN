"""Three archived-data figures, no independent rescue arm invented."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .core import *
NAMES=['figure_1_C_shared_by_position','figure_2_logit_decomposition','figure_3_individual_rescue'];COLORS={'C':'#777777','C_plus_u':'#2ca02c','z0':'#777777','u':'#2ca02c','v':'#9467bd'}

def validate(p):
 if p.duplicated(['seed','checkpoint','position']).any() or set(p[['seed','checkpoint','position']].itertuples(index=False,name=None))!={(s,k,t) for s in range(20) for k in STEPS for t in range(1,8)}:raise ValueError('All20 seeds,7 positions,170/330 required')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 if any((dest/(n+'.'+e)).exists() for n in NAMES for e in ['png','svg']):raise FileExistsError('Fresh destination required')
 p=pd.read_csv(out/'tables/position_decomposition.csv');validate(p);s=pd.read_csv(out/'tables/seed_values.csv');ci=pd.read_csv(out/'tables/intervals.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'linear-C-shared-contribution-v1'});files=[]
 def export(fig,name):
  for ext in ['png','svg']:
   path=dest/(name+'.'+ext);fig.savefig(path,dpi=170,metadata={'Software':'linear-C-shared-contribution-v1'} if ext=='png' else {'Date':None,'Creator':'linear-C-shared-contribution-v1'});files.append(path.name)
  plt.close(fig)
 def curve(ax,k,metric,color,label):
  f=p[p.checkpoint==k];x=ci[(ci.level=='position')&(ci.checkpoint==k)&(ci.qualification=='all')&(ci.metric==metric)].sort_values('position')
  for _,g in f.groupby('seed'):ax.plot(g.position,g[metric],color=color,alpha=.11,lw=.65)
  ax.fill_between(x.position,x.ci_low,x.ci_high,color=color,alpha=.16,lw=0);ax.plot(x.position,x['mean'],color=color,lw=2,marker='o',ms=4,label=label)
 def axis(ax,y):ax.set_xlabel('Sequence position');ax.set_ylabel(y);ax.set_xticks(range(1,8));ax.grid(axis='y',alpha=.13)
 fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for row,k in enumerate(STEPS):
  for col,metrics,y in [(0,[('ce_C','C'),('ce_restored','C_plus_u')],'CE (symlog; zero retained)'),(1,[('C_accuracy','C'),('restored_accuracy','C_plus_u')],'Next-token accuracy')]:
   ax=axs[row,col]
   for metric,c in metrics:curve(ax,k,metric,COLORS[c],'Natural C' if c=='C' else 'C + u (= A/B logit mean)')
   ax.set_title(f'Update{k} — '+('cross-entropy' if col==0 else 'accuracy'));axis(ax,y)
   if col==0:
    ax.set_yscale('symlog',linthresh=.01);panel=p[p.checkpoint==k];ax.set_ylim(0,max(panel.ce_C.max(),panel.ce_restored.max())*1.7)
   else:ax.set_ylim(-.03,1.05)
   ax.legend(loc='upper left' if col==0 else 'lower left',fontsize=8,frameon=False)
 fig.get_layout_engine().set(rect=(0,.06,1,.87));fig.text(.5,.012,'All20 seeds faint; cohort means and pointwise95% seed-bootstrap intervals. Conditions are external output constructions.\nC uses archived NoContext logits after explicit zero-third-channel proof. Qualified summaries are reported separately.',ha='center',fontsize=9);export(fig,NAMES[0])
 fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for col,k in enumerate(STEPS):
  for row,suffix,title,y in [(0,'_centered_norm','Centered vector magnitude','Centered logit norm (log scale)'),(1,'_fixed_C_competitor_margin','Signed contributions to a fixed margin','Correct minus fixed-wrong logit (symlog)')]:
   ax=axs[row,col]
   for name in ['z0','u','v']:curve(ax,k,name+suffix,COLORS[name],{'z0':'Baseline z0','u':'Common u','v':'A/B-specific v'}[name])
   panel=p[p.checkpoint==k];vals=np.r_[tuple(panel[name+suffix].to_numpy() for name in ['z0','u','v'])];ax.set_yscale('log' if row==0 else 'symlog',**({} if row==0 else {'linthresh':.01}));ax.set_ylim(float(vals.min())/1.7 if row==0 else min(0,float(vals.min()))*1.7,float(vals.max())*1.7);
   if row==1:ax.axhline(0,color='#777777',lw=.6)
   ax.set_title(f'Update{k} — '+title);axis(ax,y);ax.legend(fontsize=8,frameon=False)
 fig.get_layout_engine().set(rect=(0,.075,1,.86));fig.text(.5,.018,'Norms remove class-common offsets and do not establish CE causality. All20 seeds/95% intervals.\nMargin reference is C’s strongest incorrect class at that position, held fixed for z0/u/v; +v applies to A, −v to B.\nThe actual strongest competitor can switch after restoration; that switch penalty is retained in the exact tables.',ha='center',fontsize=9);export(fig,NAMES[1])
 fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax,k in zip(axs,STEPS):
  f=s[s.checkpoint==k];good=f[f.strong_familiar];weak=f[~f.strong_familiar];maximum=max(float(f.ce_C.max()),float(f.ce_restored.max()))*1.7;xx=np.r_[0,np.geomspace(.00001,maximum,200)];ax.plot(xx,xx,color='#555555',ls='--',lw=1)
  ax.scatter(weak.ce_C,weak.ce_restored,facecolors='none',edgecolors='#777777',s=35,label=f'Other seeds (n={len(weak)})');ax.scatter(good.ce_C,good.ce_restored,color='#2ca02c',edgecolors='black',lw=.6,s=42,label=f'Strong A/B (n={len(good)})')
  for index,r in enumerate(f.sort_values('ce_C').itertuples()):ax.annotate(str(r.seed),(r.ce_C,r.ce_restored),xytext=(3,5+9*(index%3)),textcoords='offset points',fontsize=6.5,color='#444444')
  ax.set_xscale('symlog',linthresh=.0001);ax.set_yscale('symlog',linthresh=.0001);ax.set_xlim(0,maximum);ax.set_ylim(0,maximum);ax.set_title(f'Update{k}: all20 seeds');ax.set_xlabel('Natural C CE (symlog)');ax.set_ylabel('C + u CE (symlog)');ax.grid(alpha=.12);ax.legend(fontsize=8,frameon=False)
 fig.get_layout_engine().set(rect=(0,.09,1,.85));fig.text(.5,.018,'One point per frozen network; labels are seed IDs. Strong familiar criterion uses only A/B CE and accuracy.\nBelow diagonal: lower loss after external u addition. This reproduces the A/B midpoint, not natural retrieval under C.',ha='center',fontsize=9);export(fig,NAMES[2]);return files

if __name__=='__main__':print('\n'.join(generate()))
