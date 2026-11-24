"""Exactly three compact table-only figures, deterministic PNG/SVG companions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .common import *
from .rules import ORDER
COL={'A':'#2475b5','B':'#e78a20','C':'#269454','Mixed-average':'#8c57af','NoContext':'#666666'}
def confirmation_message(selected):return 'No candidates qualified for confirmation.\nNo candidate runs were launched on reserved seeds2–5.' if not selected else 'Reserved seeds2–5; fixed procedures, no reselection.'
def main(destination=None):
 dest=Path(destination) if destination else OUT/'figures';dest.mkdir(parents=True,exist_ok=True);s=pd.read_csv(OUT/'summary.csv');metrics=pd.read_csv(OUT/'metrics.csv');practice=pd.read_csv(OUT/'practice.csv.gz');selection=json.loads((OUT/'selection.json').read_text());ref=json.loads((OUT/'reference_thresholds.json').read_text());plt.rcParams.update({'font.size':9,'svg.hashsalt':'joint-phenotype-v1','axes.spines.top':False,'axes.spines.right':False})
 def save(fig,name):
  fig.savefig(dest/f'{name}.png',dpi=165,facecolor='white');fig.savefig(dest/f'{name}.svg',metadata={'Date':None});plt.close(fig)
 ids=['A00']+[x for x in ORDER if x.startswith('A')]+['B00']+[x for x in ORDER if x.startswith('B')];fig,axes=plt.subplots(1,7,figsize=(15,9),sharey=True,gridspec_kw={'width_ratios':[1,1,1.2,1.3,1.3,1.3,.9]});ys=np.arange(len(ids));titles=['Q_A','Q_B','E / H0','C CE','Mixed CE','NoContext CE','Acquired / joint']
 for ax,title in zip(axes,titles):ax.set_title(title,fontsize=10);ax.axvline(0,color='#ddd',lw=.7);ax.grid(axis='x',alpha=.15)
 for yi,candidate in enumerate(ids):
  f=s[s.candidate.eq(candidate)&s.seed.isin([0,1])]
  finite=f[f.status.eq('completed')]
  if finite.empty:
   axes[6].text(0,yi,str(f.status.iloc[0]) if len(f) else 'unavailable',fontsize=7,ha='center',color='#777');continue
  for j,field in enumerate(['Q_A','Q_B','E_normalized','C_ce','Mixed_ce','NoContext_ce']):
   for row in finite.itertuples():
    value=getattr(row,field);offset=-.11 if row.seed==0 else .11
    if np.isfinite(value):axes[j].scatter(value,yi+offset,s=24,color='#2879b7' if row.seed==0 else '#e3872d',zorder=3)
    else:axes[j].text(.02,yi+offset,'undefined',transform=axes[j].get_yaxis_transform(),fontsize=6,color='#777')
  axes[6].text(0,yi,f'{int(finite.familiar.sum())}/{len(finite)}   {int(finite.joint.sum())}/{len(finite)}',ha='center',va='center',fontsize=8)
  if candidate in selection['selected']:
   for ax in axes:ax.axhspan(yi-.4,yi+.4,color='#c9e9ca',alpha=.4,zorder=0)
 axes[0].set_yticks(ys,ids);axes[0].invert_yaxis();axes[0].set_ylim(len(ids)-.5,-.5)
 for j,k in enumerate(['Q_A','Q_B','E_normalized']):
  if ref['resolved']:axes[j].axvline(.5*ref[k],color='#a23d3d',ls='--',lw=1)
 for ax in axes[3:6]:ax.set_xscale('symlog',linthresh=.1);ax.set_xlim(left=0);ax.axvline(.5,color='#a23d3d',ls=':',lw=.8)
 axes[6].set_xlim(-.6,.6);axes[6].set_xticks([]);fig.suptitle('Fixed scale5/U pilot: every candidate and both seeds',fontsize=14);fig.text(.5,.025,'Blue: seed0 · orange: seed1. Dashed red: 50% Fresh reference; dotted CE=.5 is only one part of useful performance.\nUndefined Q is nonlearning/floor-limited, never shared-learning evidence. Green rows were promoted. A07 not triggered is not a failure.',ha='center',fontsize=8);fig.subplots_adjust(left=.08,right=.985,top=.91,bottom=.1,wspace=.34);save(fig,'figure_1_screen')
 chosen=selection['selected'] or selection['partial_display'];fig,axes=plt.subplots(max(1,len(chosen)),3,figsize=(14,max(4,3.6*len(chosen))),squeeze=False)
 def curve(ax,f,candidate,control,last):
  for tag,ls,alpha in [(control,'--',.5),(candidate,'-',.85)]:
   g=f[f.candidate.eq(tag)&f['update'].le(last)]
   for (seed,c),gg in g.groupby(['seed','training_cue']):
    # Separate native10-update blocks, preserving switch jumps and original observations.
    for _,block in gg.groupby((gg['update']-1)//10):ax.plot(block['update'],block.ce_before,color=COL[c],ls=ls,lw=.7,alpha=alpha)
  for k in range(0,last,10):ax.axvspan(k+.5,k+10.5,color=COL['A' if (k//10)%2==0 else 'B'],alpha=.035)
  ax.set_yscale('symlog',linthresh=.1);ax.set_ylim(bottom=0);ax.set_xlabel('Practice update (pre-update CE)');ax.set_ylabel('CE (nats)')
 for row,candidate in enumerate(chosen):
  control='A00' if candidate.startswith('A') else 'B00';f=practice[practice.seed.isin([0,1])];curve(axes[row,0],f,candidate,control,60);curve(axes[row,1],f,candidate,control,330)
  for tag,ls in [(control,'--'),(candidate,'-')]:
   f=metrics[metrics.candidate.eq(tag)&metrics.seed.isin([0,1])]
   for c in PROBES:
    g=f[f.cue.eq(c)]
    for _,seed in g.groupby('seed'):axes[row,2].plot(seed['update'],seed.ce,color=COL[c],ls=ls,alpha=.18,lw=.7)
    mean=g.groupby('update').ce.mean();axes[row,2].plot(mean.index,mean,color=COL[c],ls=ls,lw=1.3)
  axes[row,0].set_title(f'{candidate} vs {control}: first60');axes[row,1].set_title('Actual A/B practice, all330');axes[row,2].set_title('Same-weight frozen probes');axes[row,2].set_yscale('symlog',linthresh=.1);axes[row,2].set_ylim(bottom=0);axes[row,2].set_xlabel('Completed updates');axes[row,2].set_ylabel('CE (nats)')
 handles=[Line2D([0],[0],color=COL[c],label=c) for c in PROBES]+[Line2D([0],[0],color='black',ls='-',label='candidate'),Line2D([0],[0],color='black',ls='--',label='matched control')];fig.legend(handles=handles,loc='lower center',ncol=7,fontsize=8);fig.suptitle('Promoted pilot procedures' if selection['selected'] else 'Descriptive partial signals only — no promoted candidates',fontsize=13);fig.tight_layout(rect=(0,.07,1,.94));save(fig,'figure_2_behavior')
 selected=selection['selected']
 if not selected:
  fig,ax=plt.subplots(figsize=(9,3.5));ax.axis('off');ax.text(.5,.6,confirmation_message(selected),ha='center',va='center',fontsize=15,linespacing=1.8);ax.text(.5,.19,'A bounded negative screen. Reserved source controls are not new candidate confirmations.\nNo failed seed was replaced and no threshold was relaxed.',ha='center',va='center',fontsize=10);save(fig,'figure_3_confirmation')
 else:
  fig,axes=plt.subplots(len(selected),3,figsize=(12,3.7*len(selected)),squeeze=False)
  for row,candidate in enumerate(selected):
   control='A00' if candidate.startswith('A') else 'B00';a0=s[s.candidate.eq(candidate)&s.seed.isin([2,3,4,5])].set_index('seed');b=s[s.candidate.eq(control)&s.seed.isin([2,3,4,5])].set_index('seed')
   for seed in [2,3,4,5]:
    if seed not in a0.index:axes[row,0].text(seed,0,'missing',rotation=90);continue
    arow=a0.loc[seed];brow=b.loc[seed];axes[row,0].plot([seed-.13,seed+.13],[brow.C_ce,arow.C_ce],color='#777',lw=.8);axes[row,0].scatter(seed-.13,brow.C_ce,color='#aaa',s=28);axes[row,0].scatter(seed+.13,arow.C_ce,color=COL['C'],s=28)
    for off,key,color in [(-.16,'Q_A',COL['A']),(0.,'Q_B',COL['B']),(.16,'E_normalized','#555')]:
     value=arow[key]/ref[key]
     if np.isfinite(value):axes[row,1].scatter(seed+off,value,color=color,s=28)
     else:axes[row,1].text(seed+off,0,'?',ha='center')
    axes[row,2].text(seed,.65,'A/B '+('✓' if arow.familiar else '×'),ha='center');axes[row,2].text(seed,.3,'Joint '+('✓' if arow.joint else '×'),ha='center')
   axes[row,0].set_title(candidate+': paired C loss');axes[row,0].set_ylabel('CE (nats)');axes[row,0].axhline(.5,ls=':',color='#a44');axes[row,1].set_title('Fresh retention: Q_A, Q_B, E/H0');axes[row,1].axhline(.5,ls='--',color='#a44');axes[row,2].set_title('Same-seed gate components');axes[row,2].set_ylim(0,1);axes[row,2].set_yticks([])
   for ax in axes[row]:ax.set_xticks([2,3,4,5]);ax.set_xlim(1.6,5.4);ax.set_xlabel('Reserved seed')
  fig.suptitle('Confirmation: unchanged procedures, reserved four seeds');fig.tight_layout(rect=(0,0,1,.95));save(fig,'figure_3_confirmation')
 put(dest/'FIGURE_INDEX.json',dict(files={p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.png','.svg']},sources={n:sha(OUT/n) for n in ['summary.csv','metrics.csv','practice.csv.gz','selection.json','reference_thresholds.json']},model_evaluations=0,optimizer_updates=0));(dest/'README.md').write_text('# Figures\n\n[1 Screen overview](figure_1_screen.png) · [SVG](figure_1_screen.svg)\n\n[2 Pilot behavior](figure_2_behavior.png) · [SVG](figure_2_behavior.svg)\n\n[3 Confirmation](figure_3_confirmation.png) · [SVG](figure_3_confirmation.svg)\n\nEvery dot is a seed. All conditions, missing/undefined entries and matched controls are retained in the exact root CSVs. Figure2 uses the fixed descriptive partial ranking when no candidate advances; those arms are not winners. Table-only generation never evaluates a model or updates an optimizer.\n');return dest
