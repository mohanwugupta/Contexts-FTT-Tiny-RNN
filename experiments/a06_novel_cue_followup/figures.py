"""Five table-only figure families. Every registered seed and cue is retained."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .protocol import *
COL={'A00':'#444444','A00-static':'#568bbd','A06':'#148553'};SEED_COL=plt.get_cmap('tab10').colors[:6]
def main(destination=None):
 dest=Path(destination) if destination else OUT/'figures';dest.mkdir(parents=True,exist_ok=True);f=pd.read_csv(OUT/'original_cue_results.csv');novel=pd.read_csv(OUT/'unfamiliar_cue_results.csv');learning=pd.read_csv(OUT/'learning_measures.csv');practice=pd.read_csv(OUT/'practice.csv.gz');registry=pd.read_csv(OUT/'cue_registry.csv');plt.rcParams.update({'font.size':9,'svg.hashsalt':'a06-novel-followup-v1','axes.spines.top':False,'axes.spines.right':False})
 def save(fig,n,name):fig.savefig(dest/f'figure_{n}_{name}.png',dpi=160,facecolor='white');fig.savefig(dest/f'figure_{n}_{name}.svg',metadata={'Date':None});plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for row,c in enumerate(['A00','A06']):
  for col,last in enumerate([100,330]):
   ax=axes[row,col];g=practice[practice.condition.eq(c)&practice['update'].le(last)]
   for seed,z in g.groupby('seed'):
    ax.plot(z['update'],z.ce_before,color='#999',alpha=.18,lw=.6)
    for _,block in z.groupby((z['update']-1)//10):ax.plot(block['update'],block.ce_before,color='#2475b5' if block.training_cue.iloc[0]=='A' else '#e78a20',alpha=.35,lw=.8,marker=['o','s','^','v','D','P'][int(seed)],markevery=[0],ms=2)
   mean=g.groupby(['update','training_cue'],as_index=False).ce_before.mean()
   for _,block in mean.groupby((mean['update']-1)//10):ax.plot(block['update'],block.ce_before,color='#2475b5' if block.training_cue.iloc[0]=='A' else '#e78a20',lw=1.8)
   for k in range(10,last,10):ax.axvline(k+.5,color='#ccc',lw=.4,alpha=.4)
   ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.set_xlabel('Practice update (pre-update loss)');ax.set_ylabel('CE (nats)');ax.set_title(f'{c}: first {last} · all6 seeds + mean')
 fig.legend(handles=[Line2D([0],[0],color='#2475b5',label='A'),Line2D([0],[0],color='#e78a20',label='B')]+[Line2D([0],[0],color='#777',ls='',marker=m,label=f'seed{s}') for s,m in enumerate(['o','s','^','v','D','P'])],loc='lower center',ncol=8,frameon=False,fontsize=8);fig.suptitle('Actual A10/B10 acquisition: unchanged native task and schedule');fig.tight_layout(rect=(0,.06,1,.94));save(fig,1,'splitting')
 fig,axes=plt.subplots(3,2,figsize=(11,9))
 common=[0,10,20,30,40,50,60,100,170,330]
 for row,cue_ in enumerate(['C','NoContext','Mixed-average']):
  for condition in ['A00','A00-static','A06']:
   g=f[f.condition.eq(condition)&f.cue.eq(cue_)]
   for seed,z in g.groupby('seed'):
    for col,metric in enumerate(['ce','accuracy']):axes[row,col].plot(z['update'],z[metric]*(7 if metric=='accuracy' else 1),color=COL[condition],alpha=.22,lw=.7,marker='o' if condition=='A06' and seed<2 else None,ms=2.5)
   mean=g[g['update'].isin(common)].groupby('update')[['ce','correct_count']].mean()
   axes[row,0].plot(mean.index,mean.ce,color=COL[condition],ls=':' if condition=='A00-static' else '-',marker='o',ms=2,lw=1.6,label=condition);axes[row,1].plot(mean.index,mean.correct_count,color=COL[condition],ls=':' if condition=='A00-static' else '-',marker='o',ms=2,lw=1.6)
  axes[row,0].set_yscale('symlog',linthresh=.01);axes[row,0].set_ylim(bottom=0);axes[row,0].set_ylabel(cue_+' CE (nats)');axes[row,1].set_ylabel(cue_+' correct /7');axes[row,1].set_ylim(-.2,7.2);axes[row,1].axhline(6,ls='--',lw=.6,color='#aaa')
  for ax in axes[row]:ax.set_xlabel('Completed target updates');ax.set_xlim(0,330);ax.grid(alpha=.1)
 fig.legend(handles=[Line2D([0],[0],color=COL[c],label=c,ls=':' if c=='A00-static' else '-') for c in COL],loc='lower center',ncol=3,frameon=False);fig.suptitle('Generalization: individuals faint, six-seed means at10 shared saved states\nGaps in old A06 pilots are retained, not interpolated',fontsize=12);fig.tight_layout(rect=(0,.05,1,.93));save(fig,2,'C_NoContext_Mixed')
 cue_order=['C']+registry.cue.tolist()+['NoContext'];fig,axes=plt.subplots(6,2,figsize=(14,12),sharex=True)
 for seed in SEEDS:
  for i,cue_ in enumerate(cue_order):
   g=novel[novel.seed.eq(seed)&novel['update'].eq(330)&novel.cue.eq(cue_)].set_index('condition')
   for col,metric in enumerate(['ce','correct_count']):
    ax=axes[seed,col]
    if all(c in g.index for c in ['A00','A06']):ax.plot([i-.1,i+.1],[g.loc['A00',metric],g.loc['A06',metric]],color='#aaa',lw=.7)
    for condition,off in [('A00',-.1),('A06',.1)]:
     if condition in g.index:ax.scatter(i+off,g.loc[condition,metric],color=COL[condition],s=16,zorder=3)
   if cue_ in ['C','NoContext']:
    for ax in axes[seed]:ax.axvspan(i-.4,i+.4,color='#eee',alpha=.5,zorder=0)
  axes[seed,0].set_yscale('symlog',linthresh=.01);axes[seed,0].set_ylim(0,novel[novel.seed.eq(seed)&novel['update'].eq(330)&novel.cue.isin(cue_order)&novel.condition.isin(['A00','A06'])].ce.max()*1.5+.02);axes[seed,0].set_ylabel(f'Seed{seed} · CE');axes[seed,1].set_ylim(-.2,7.3);axes[seed,1].set_yticks([0,3,6,7]);axes[seed,1].set_ylabel('Correct /7')
 for ax in axes[-1]:ax.set_xticks(range(14),cue_order,rotation=40,ha='right')
 fig.legend(handles=[Line2D([0],[0],color=COL[c],marker='o',ls='',label=c) for c in ['A00','A06']],loc='lower center',ncol=2,frameon=False);fig.suptitle('All12 unfamiliar cues, learned C and NoContext · update330\nSeeds0/1 reused pilots;2–5 new exploratory follow-up',fontsize=13);fig.tight_layout(rect=(0,.045,1,.95));save(fig,3,'unfamiliar_cues')
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for row,c in enumerate(['A00','A06']):
  for seed in SEEDS:
   g=novel[novel.condition.eq(c)&novel.seed.eq(seed)&novel['update'].eq(330)&novel.kind.ne('original')]
   for kind,marker in [('random_sphere','o'),('C_arc','^')]:
    z=g[g.kind.eq(kind)];axes[row,0].scatter(z.similarity_C,z.ce,color=SEED_COL[seed],marker=marker,s=24,alpha=.8);axes[row,1].scatter(z.similarity_C,z.correct_count,color=SEED_COL[seed],marker=marker,s=24,alpha=.8)
  axes[row,0].set_yscale('symlog',linthresh=.01);axes[row,0].set_ylim(bottom=0);axes[row,0].set_ylabel(c+' CE (nats)');axes[row,1].set_ylabel(c+' correct /7');axes[row,1].set_ylim(-.2,7.2)
  for ax in axes[row]:ax.set_xlabel('Cosine similarity to C (equal-norm inputs)');ax.set_xlim(-1.05,1.05);ax.grid(alpha=.1)
 handles=[Line2D([0],[0],ls='',marker='o',color=SEED_COL[s],label=f'seed{s}') for s in SEEDS]+[Line2D([0],[0],ls='',marker='o',color='black',label='random sphere'),Line2D([0],[0],ls='',marker='^',color='black',label='C arc')];fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=8);fig.suptitle('Cue identity and transfer at330 · every cue and seed, no fitted direction');fig.tight_layout(rect=(0,.09,1,.94));save(fig,4,'similarity')
 fig,axes=plt.subplots(1,2,figsize=(12,5.5))
 for seed in SEEDS:
  g=learning[learning.seed.eq(seed)].set_index('condition')
  for col,metric in [(0,'C_ce'),(1,'NoContext_ce')]:
   ax=axes[col];xx=[g.loc[c,metric] for c in ['A00','A06']];yy=[(g.loc[c,'D_A']+g.loc[c,'D_B'])/2 for c in ['A00','A06']];ax.plot(xx,yy,color=SEED_COL[seed],alpha=.5,lw=.8)
   for c,marker in [('A00','o'),('A06','^')]:
    r=g.loc[c];x=r[metric];y=(r.D_A+r.D_B)/2;ax.scatter(x,y,marker=marker,s=85,facecolors=SEED_COL[seed] if r.familiar else 'none',edgecolors='#17703b' if r.original_early_phenotype else '#c33',linewidths=1.5,clip_on=False);ax.annotate(str(seed),(x,y),xytext=(7,-12) if seed==4 else (5,5),textcoords='offset points',fontsize=8)
    if not r.Mixed_useful:ax.scatter(x,y,marker='x',color='red',s=60)
    if r.net_A<=0 or r.net_B<=0:ax.annotate('!',(x,y),color='red')
  
 for ax,title in zip(axes,['Original C','NoContext diagnostic']):ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0);ax.set_xlabel(title+' absolute CE at330');ax.set_ylabel('(first-A D_A + first-B D_B)/2 (nats)');ax.set_title(title);ax.axhline(0,color='#aaa',lw=.7)
 handles=[Line2D([0],[0],ls='',marker='o',color='#555',label='A00'),Line2D([0],[0],ls='',marker='^',color='#555',label='A06'),Line2D([0],[0],ls='',marker='o',markerfacecolor='#777',markeredgecolor='#17703b',label='filled: strong A/B; green edge: original early flag'),Line2D([0],[0],ls='',marker='x',color='red',label='red ×: Mixed fails; !: net learning nonpositive')];fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=8);fig.suptitle('Joint components at330 · paired by trained seed · unchanged success rules');fig.tight_layout(rect=(0,.12,1,.93));save(fig,5,'joint')
 files={p.name:sha(p) for p in dest.iterdir() if p.suffix in ['.png','.svg']};put(dest/'FIGURE_INDEX.json',dict(files=files,sources={n:sha(OUT/n) for n in ['original_cue_results.csv','unfamiliar_cue_results.csv','learning_measures.csv','practice.csv.gz','cue_registry.csv']},optimizer_steps=0,model_forward_calls=0))
 (dest/'README.md').write_text('# Five figure families\n\n'+''.join(f'- [Figure{i} {name}](figure_{i}_{name}.png) · [SVG](figure_{i}_{name}.svg)\n' for i,name in [(1,'splitting'),(2,'C_NoContext_Mixed'),(3,'unfamiliar_cues'),(4,'similarity'),(5,'joint')])+'\nAll individual seeds/cues retained; no fitted cue direction. Means inFigure2 use shared recorded states, while unavailable old-pilot dense observations stay missing. Original early flags and screen-joint counts are distinct from the unassessed original late counterfactual gate.\n');return files
