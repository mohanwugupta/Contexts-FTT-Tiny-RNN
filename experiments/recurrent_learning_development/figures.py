"""Deterministic three-figure saved-table regeneration; geometry is never interpolated."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analysis import OUT,ARCHS
COLORS=['#2166ac','#b35806'];NAMES=['fig_1_context_specific_blocks','fig_2_within_cross_learning','fig_3_learning_and_separation']

def validate(b):
 expected={(s,a,e) for s in range(20) for a in ARCHS for e in range(10,331,10)}
 if b.duplicated(['seed','architecture','end']).any() or set(b[['seed','architecture','end']].itertuples(index=False,name=None))!=expected:raise ValueError('All20 seeds and33 blocks required in both architectures')
 return True

def generate(out=OUT,destination=None):
 out=Path(out);dest=out/'figures' if destination is None else Path(destination);dest.mkdir(parents=True,exist_ok=True)
 if any((dest/(n+'.'+e)).exists() for n in NAMES for e in ['png','svg']):raise FileExistsError('Figure products cannot be overwritten')
 read=lambda n:pd.read_csv(out/'tables'/n)
 b=read('block_learning.csv');validate(b);bi=read('block_intervals.csv');l=read('original_learning_trajectories.csv');li=read('loss_intervals.csv');sw=read('original_switch_costs.csv');si=read('switch_intervals.csv');d=read('original_learning_transfer.csv');di=read('diagnostic_intervals.csv');w=read('original_direct_specificity.csv');wi=read('specificity_intervals.csv');g=read('geometry_learning.csv');gi=read('geometry_intervals.csv');floors=read('floor_counts.csv')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'recurrent-learning-development-v1'})
 paths=[]
 def export(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext);fig.savefig(p,dpi=170,metadata={'Software':'recurrent-learning-development-v1'} if ext=='png' else {'Date':None,'Creator':'recurrent-learning-development-v1'});paths.append(p.name)
  plt.close(fig)
 def curve(ax,raw,ci,x,y,color,label,style='-',markers=False):
  for _,s in raw.groupby('seed'):
   s=s.sort_values(x);ax.plot(s[x],s[y],color=color,alpha=.14,lw=.65,ls=style)
  ci=ci.sort_values(x);xx=ci[x].to_numpy();ax.plot(xx,ci['mean'],color=color,lw=2,ls=style,marker='o' if markers else None,ms=3,label=label);ax.fill_between(xx,ci.ci_low,ci.ci_high,color=color,alpha=.15)
 def switches(ax):
  for k in range(10,330,10):ax.axvline(k,color='grey',lw=.5,alpha=.22)
  ax.axvline(170,color='black',lw=1,ls='--',alpha=.7);ax.set_xlim(0,334);ax.axhline(0,color='grey',lw=.6);ax.set_xlabel('Scientific updates completed');ax.grid(axis='y',alpha=.13)
 def spec(ax,a):
  f=b[b.architecture==a];ci=bi[(bi.architecture==a)&(bi.metric=='specificity')]
  curve(ax,f,ci,'end','specificity',COLORS[0],'Own gain − cross gain',markers=True);ax.set_yscale('symlog',linthresh=1e-4);switches(ax)
  floor=floors[floors.architecture==a]
  # A visible bottom strip encodes the actual number of floor-limited networks.
  ax2=ax.twinx();ax2.fill_between(floor.end,0,floor.both_start_floor/20,color='#7570b3',alpha=.12,step='pre');ax2.set_ylim(0,6);ax2.set_yticks([0,1],['0/20','20/20']);ax2.set_ylabel('Both-start floor (strip)',color='#66509a',fontsize=8);ax2.spines['right'].set_visible(True)
  ax.set_ylabel('10-update specificity (nats)');ax.legend(fontsize=8,loc='upper right')
 fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained')
 for j,a in enumerate(ARCHS):
  ax=axs[0,j];ax.set_title(a+' — original familiar losses')
  for c,color,style in [('A',COLORS[0],'-'),('B',COLORS[1],'--')]:curve(ax,l[(l.architecture==a)&(l.cue==c)],li[(li.architecture==a)&(li.cue==c)],'update','ce',color,c,style)
  ax.axhline(.05,color='#7570b3',lw=.9,ls=':',label='0.05 floor marker');ax.set_yscale('symlog',linthresh=1e-4);ax.set_ylabel('CE (nats)');switches(ax);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
  ax=axs[1,j];ax.set_title('Signed learning specificity per completed block');spec(ax,a)
  ax=axs[2,j];f=sw[sw.architecture==a];ci=si[(si.architecture==a)&(si.metric=='adjusted_cost')];curve(ax,f,ci,'boundary','adjusted_cost',COLORS[1],'Incoming − outgoing, minus initial cue difference',markers=True);curve(ax,f,si[(si.architecture==a)&(si.metric=='raw_cost')],'boundary','raw_cost','#238b45','Raw incoming − outgoing','--');ax.set_yscale('symlog',linthresh=.01);switches(ax);ax.set_ylabel('Switch cost (nats)');ax.set_title('Saved raw and baseline-adjusted costs');ax.legend(fontsize=7)
 fig.suptitle('Context-specific learning across all 33 A10/B10 blocks',fontsize=16)
 fig.get_layout_engine().set(rect=(0,.045,1,.96));fig.text(.5,.012,'Thin lines: all20 seeds; bands: pointwise95% seed-bootstrap intervals. Block dots summarize preceding10 updates.\nVertical lines: every executed switch; dashed:170. Purple strip: both losses<0.05 at block start; low gains do not establish shared learning.',ha='center',fontsize=9)
 export(fig,NAMES[0])
 fig,axs=plt.subplots(3,2,figsize=(14,11),layout='constrained')
 for j,a in enumerate(ARCHS):
  ax=axs[0,j];ax.set_title(a+' — benefits of a full training block')
  for metric,color,style in [('own',COLORS[0],'-'),('cross',COLORS[1],'--')]:curve(ax,b[b.architecture==a],bi[(bi.architecture==a)&(bi.metric==metric)],'end',metric,color,'Practiced-context gain' if metric=='own' else 'Other-context gain',style,True)
  ax.set_yscale('symlog',linthresh=1e-4);switches(ax);ax.set_ylabel('10-update CE decrease (nats)');ax.legend(fontsize=8)
  for row,train in [(1,'A'),(2,'B')]:
   ax=axs[row,j];ax.set_title(f'Restored-Adam one-update diagnostic: train {train}')
   for cue,color,style in [(train,COLORS[0],'-'),('B' if train=='A' else 'A',COLORS[1],'--')]:
    raw=d[(d.architecture==a)&(d.train_context==train)&(d.evaluation_context==cue)];ci=di[(di.architecture==a)&(di.train_context==train)&(di.evaluation_context==cue)];curve(ax,raw,ci,'checkpoint','improvement',color,'Within context' if cue==train else 'Cross context',style,True)
   ax.set_yscale('symlog',linthresh=1e-5);switches(ax);ax.set_ylabel('One-update CE decrease (nats)');ax.set_xticks([0,40,80,120,170,250,330]);ax.legend(fontsize=8)
 fig.suptitle('Within-context and cross-context learning benefits',fontsize=16);fig.get_layout_engine().set(rect=(0,.045,1,.96));fig.text(.5,.012,'Negative gains are retained. Top: ten actual updates; lower rows: isolated one-update clones, full Adam moments restored.\nDiagnostics exist only at0,10,20,40,80,120,170,330; connecting lines are visual aids. Magnitudes have different update horizons.',ha='center',fontsize=9);export(fig,NAMES[1])
 fig,axs=plt.subplots(4,2,figsize=(14,14),layout='constrained')
 for j,a in enumerate(ARCHS):
  ax=axs[0,j];ax.set_title(a+' — completed-block specificity');spec(ax,a)
  ax=axs[1,j];ax.set_title('One-update directional specificity (no geometry criterion)')
  for metric,color,style in [('D_A',COLORS[0],'-'),('D_B',COLORS[1],'--')]:curve(ax,w[w.architecture==a],wi[(wi.architecture==a)&(wi.metric==metric)],'checkpoint',metric,color,metric,style,True)
  ax.set_yscale('symlog',linthresh=1e-5);switches(ax);ax.set_ylabel('One-update own − cross (nats)');ax.legend(fontsize=8)
  ax=axs[2,j];ax.set_title('Sparse saved separation:170 and330 only')
  for metric,color,offset in [('hidden_centroid_distance',COLORS[0],-2),('centered_logit_separation',COLORS[1],2)]:
   for k in [170,330]:
    f=g[(g.architecture==a)&(g.endpoint==k)].sort_values('seed');ci=gi[(gi.architecture==a)&(gi.endpoint==k)&(gi.metric==metric)].iloc[0];x=k+offset
    ax.scatter(x+(f.seed-9.5)*.45,f[metric],s=13,color=color,alpha=.35);ax.errorbar(x,ci['mean'],yerr=[[ci['mean']-ci.ci_low],[ci.ci_high-ci['mean']]],fmt='o',color=color,capsize=4,label=metric.replace('_',' ') if k==170 else None)
  ax.set_xlim(0,340);ax.set_xticks([0,40,80,120,170,250,330]);ax.set_xlabel('Scientific updates completed (no earlier saved geometry)');ax.set_ylabel('Euclidean distance (scale-dependent)');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
  ax=axs[3,j];ax.set_title('Endpoint geometry versus completed-block specificity')
  for k,color,mark in [(170,COLORS[0],'o'),(330,COLORS[1],'s')]:
   f=g[(g.architecture==a)&(g.endpoint==k)];ax.scatter(f.hidden_centroid_distance,f.specificity,color=color,marker=mark,alpha=.7,s=24,label=f'Update{k}:all20')
  ax.set_yscale('symlog',linthresh=1e-5);ax.axhline(0,color='grey',lw=.6);ax.set_xlabel('A/B hidden centroid distance');ax.set_ylabel('10-update specificity (nats)');ax.legend(fontsize=8)
 fig.suptitle('Functional learning and representational separation are distinct',fontsize=16);fig.get_layout_engine().set(rect=(0,.04,1,.96));fig.text(.5,.01,'All20 seeds retained. No geometry interpolation or geometry-based splitting. Linear low-gain periods remain floor-limited.\nDistances depend on architecture/hidden scale; the two Euclidean measures are not commensurate. Probability TV is retained in tables.',ha='center',fontsize=9);export(fig,NAMES[2]);return paths

if __name__=='__main__':print('\n'.join(generate()))
