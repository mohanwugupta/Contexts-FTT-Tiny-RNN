"""Three saved-table figures, allseed and actual common-strong n explicit."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .core import *
NAMES_F=['figure_1_novel_context_generalization','figure_2_novel_gap_paired','figure_3_novel_vs_familiar'];COLORS={'A':'#1f77b4','B':'#ff7f0e','Mixed':'#2ca02c','NoContext':'#777777','Novel30_A':'#9467bd','Novel30_B':'#5a3d8f','Novel60_A':'#e377c2','Novel60_B':'#b5164c'}
SHORT=['A','B','Mixed','NoCtx','30 A','30 B','60 A','60 B']

def validate(v):
 expected={(s,a,k,c) for s in range(20) for a in ARCHS for k in STEPS for c in NAMES}
 if v.duplicated(['seed','architecture','checkpoint','cue']).any() or set(v[['seed','architecture','checkpoint','cue']].itertuples(index=False,name=None))!=expected:raise ValueError('All640 registered cells required')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 if any((dest/(n+'.'+ext)).exists() for n in NAMES_F for ext in ['png','svg']):raise FileExistsError('Fresh figure directory required')
 v=pd.read_csv(out/'tables/seed_values.csv');validate(v);ci=pd.read_csv(out/'tables/intervals.csv');config=json.loads((out/'audit/config.json').read_text());common=config['common_strong330_seeds'];ncommon=len(common)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'high-scale-novel-contexts-v1'});files=[]
 def export(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext);fig.savefig(p,dpi=170,metadata={'Software':'high-scale-novel-contexts-v1'} if ext=='png' else {'Date':None,'Creator':'high-scale-novel-contexts-v1'});files.append(p.name)
  plt.close(fig)
 def point(ax,x,r,color):
  if np.isfinite(r['mean']):ax.errorbar(x,r['mean'],yerr=[[max(0,r['mean']-r.ci_low)],[max(0,r.ci_high-r['mean'])]],fmt='o',color=color,ms=4,capsize=3,lw=1.5,zorder=5)
 fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
 for row,k in enumerate(STEPS):
  for col,a in enumerate(ARCHS):
   ax=axs[row,col]
   for x,c in enumerate(NAMES):
    f=v[(v.architecture==a)&(v.checkpoint==k)&(v.cue==c)].sort_values('seed');r=ci[(ci.architecture==a)&(ci.checkpoint==k)&(ci.cue==c)&(ci.qualification=='all')&(ci.metric=='ce')].iloc[0];ax.scatter(x+(f.seed-9.5)*.027,f.ce,color=COLORS[c],s=15,alpha=.42);point(ax,x,r,'black')
   ax.axhline(CHANCE,color='#555555',ls='--',lw=1);ax.set_xticks(range(8),SHORT);ax.set_title(f'{a} — update{k}');ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(0,max(v[(v.architecture==a)&(v.checkpoint==k)].ce.max(),ci[(ci.architecture==a)&(ci.checkpoint==k)&(ci.qualification=='all')&(ci.metric=='ce')&ci.cue.isin(NAMES)].ci_high.max())*1.7);ax.set_ylabel('CE (symlog; zero retained)');ax.grid(axis='y',alpha=.13)
 fig.get_layout_engine().set(rect=(0,.055,1,.87));fig.text(.5,.013,'All20 seeds per cell; black means/pointwise95% seed-bootstrap intervals. Dashed:chance ln4=1.386.\n30/60 A/B are the four registered Novel vectors; no cue pooling in these panels. All losses retained without clipping.',ha='center',fontsize=9);export(fig,NAMES_F[0])
 fig,axs=plt.subplots(3,1,figsize=(11,10),layout='constrained')
 for ax,k,qual,title in [(axs[0],170,'all','Update170 — all20 paired seeds'),(axs[1],330,'all','Update330 — all20 paired seeds'),(axs[2],330,'common_strong330',f'Update330 — both architectures strong familiar: n={ncommon}')]:
  seeds=list(range(20)) if qual=='all' else common
  for x,c in enumerate(NOVEL):
   both=v[(v.checkpoint==k)&(v.cue==c)&v.seed.isin(seeds)].pivot(index='seed',columns='architecture',values='gap')
   for seed in seeds:
    jitter=(seed-9.5)*.013
    if seed in both.index:ax.plot([x-.15+jitter,x+.15+jitter],[both.loc[seed,'RNN-Linear'],both.loc[seed,'RNN-Tanh']],color='#777777',alpha=.15,lw=.6)
   for a,offset,color in [('RNN-Linear',-.15,'#1f77b4'),('RNN-Tanh',.15,'#ff7f0e')]:
    f=v[(v.architecture==a)&(v.checkpoint==k)&(v.cue==c)&v.seed.isin(seeds)].sort_values('seed');r=ci[(ci.architecture==a)&(ci.checkpoint==k)&(ci.cue==c)&(ci.qualification==qual)&(ci.metric=='gap')].iloc[0];ax.scatter(x+offset+(f.seed-9.5)*.013,f.gap,color=color,s=22,alpha=.6);point(ax,x+offset,r,color)
  ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(4),['Novel30_A','Novel30_B','Novel60_A','Novel60_B']);ax.set_ylabel('Novel−familiar CE (symlog)');ax.set_yscale('symlog',linthresh=.01);plotted=v[(v.checkpoint==k)&v.cue.isin(NOVEL)&v.seed.isin(seeds)].gap;selected_ci=ci[(ci.checkpoint==k)&ci.cue.isin(NOVEL)&(ci.qualification==qual)&(ci.metric=='gap')];lo=min(0,float(plotted.min()),float(selected_ci.ci_low.min()));hi=max(float(plotted.max()),float(selected_ci.ci_high.max()));ax.set_ylim(lo*1.7,hi*1.7);ax.set_title(title);ax.grid(axis='y',alpha=.13)
 fig.legend(handles=[Line2D([0],[0],color='#1f77b4',marker='o',ls='',label='Linear'),Line2D([0],[0],color='#ff7f0e',marker='o',ls='',label='Tanh')],loc='outside upper center',ncol=2,frameon=False);fig.get_layout_engine().set(rect=(0,.045,1,.90));fig.text(.5,.012,'Every seed remains visible; thin grey links connect the same seed across architectures. Means/95% seed intervals.\nCommon subgroup uses only the fixed A/B-learning criterion; paired Tanh−Linear intervals are provided in tables.',ha='center',fontsize=9);export(fig,NAMES_F[1])
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for row,k in enumerate(STEPS):
  for col,a in enumerate(ARCHS):
   ax=axs[row,col]
   for c in NOVEL:
    f=v[(v.architecture==a)&(v.checkpoint==k)&(v.cue==c)];weak=f[~f.strong_familiar];good=f[f.strong_familiar];ax.scatter(weak.familiar_ce,weak.ce,facecolors='none',edgecolors=COLORS[c],s=24,alpha=.45);ax.scatter(good.familiar_ce,good.ce,color=COLORS[c],edgecolors='black',lw=.55,s=30,alpha=.85)
   ax.axhline(CHANCE,color='#555555',ls='--',lw=.9);ax.set_xscale('symlog',linthresh=.0001);ax.set_yscale('symlog',linthresh=.01);panel=v[(v.architecture==a)&(v.checkpoint==k)&v.cue.isin(NOVEL)];ax.set_xlim(0,panel.familiar_ce.max()*1.7);ax.set_ylim(0,panel.ce.max()*1.7);ax.set_title(f'{a} — update{k}');ax.set_xlabel('Familiar mean A/B CE (symlog)');ax.set_ylabel('Novel CE (symlog)');ax.grid(alpha=.1)
 handles=[Line2D([0],[0],color=COLORS[c],marker='o',ls='',label=c) for c in NOVEL];fig.legend(handles=handles,loc='outside upper center',ncol=4,frameon=False,fontsize=9);fig.get_layout_engine().set(rect=(0,.055,1,.90));fig.text(.5,.013,'Allfour Novel cues per seed. Filled points with black edges meet the original strong A/B criterion; hollow points do not.\nBoth familiar contexts must meet CE/accuracy criteria separately (meanCE alone does not define qualification). Dashed:chance.',ha='center',fontsize=9);export(fig,NAMES_F[2]);return files

if __name__=='__main__':print('\n'.join(generate()))
