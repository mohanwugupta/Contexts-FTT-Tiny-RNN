"""Two compact figures from saved comparisons and actual practice curves."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .analysis import ARMS
COL={'A00':'#444444','A00-static':'#5289bb','A06':'#168253'};STYLE={'A00':'-','A00-static':':','A06':'-'}
def render(out,f,p,practice):
 plt.rcParams.update({'font.size':9,'svg.hashsalt':'a06-cached-readout-v1','axes.spines.right':False,'axes.spines.top':False})
 def save(fig,name):fig.savefig(out/(name+'.png'),dpi=170,facecolor='white');fig.savefig(out/(name+'.svg'),metadata={'Date':None});plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True)
 for seed in [0,1]:
  for column,cue in [(0,'C'),(2,'NoContext')]:
   for arm in ARMS:
    g=f[f.seed.eq(seed)&f.cue.eq(cue)&f.arm.eq(arm)].sort_values('update');axes[seed,column].plot(g['update'],g.ce,color=COL[arm],ls=STYLE[arm],marker='o',ms=2.5,lw=1.4,label=arm);axes[seed,column+1].plot(g['update'],g.correct_count,color=COL[arm],ls=STYLE[arm],marker='o',ms=2.5,lw=1.4)
   axes[seed,column].set_yscale('symlog',linthresh=.01);axes[seed,column].set_ylim(bottom=0);axes[seed,column].set_ylabel(f'Seed {seed} · CE (nats)');axes[seed,column].set_title(cue+' loss');axes[seed,column+1].set_title(cue+' correct count');axes[seed,column+1].set_ylim(-.2,7.3);axes[seed,column+1].set_yticks([0,3,6,7]);axes[seed,column+1].axhline(6,color='#bbb',ls='--',lw=.7)
  for ax in axes[seed]:ax.set_xlim(0,330);ax.grid(alpha=.1);ax.set_xticks([0,60,170,330]);ax.set_xlabel('Completed target updates')
 fig.legend(handles=[Line2D([0],[0],color=COL[a],ls=STYLE[a],lw=2,label=a) for a in ARMS],loc='lower center',ncol=3,frameon=False);fig.suptitle('Cached readout comparison · same checkpoint and targets',fontsize=13);fig.tight_layout(rect=(0,.06,1,.94));save(fig,'figure_1_readout_comparison')
 fig,axes=plt.subplots(2,4,figsize=(14,6.5))
 for seed in [0,1]:
  for col,last in [(0,60),(1,330)]:
   ax=axes[seed,col]
   for arm,ls,alpha in [('A00','--',.7),('A06','-',1.)]:
    g=practice[practice.seed.eq(seed)&practice.candidate.eq(arm)&practice['update'].le(last)].sort_values('update');ax.plot(g['update'],g.ce_before,color='#aaa',ls=ls,lw=.5,alpha=.4)
    for _,block in g.groupby((g['update']-1)//10):ax.plot(block['update'],block.ce_before,color='#2475b5' if block.training_cue.iloc[0]=='A' else '#e78a20',ls=ls,lw=1.1,alpha=alpha)
   for k in range(10,last,10):ax.axvline(k+.5,color='#aaa',lw=.4,alpha=.3)
   if last==60:
    for k in range(6):ax.text((k*10+5.5)/60,.97,'A' if k%2==0 else 'B',ha='center',va='top',transform=ax.transAxes,color='#2475b5' if k%2==0 else '#e78a20',fontsize=8)
   ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.set_xlim(.5,last+.5);ax.set_xlabel('Practice update (before update)');ax.set_ylabel(f'Seed {seed} · CE (nats)');ax.set_title('Actual A/B: first 60' if last==60 else 'Actual A/B: full 330')
  for col,cue in [(2,'C'),(3,'NoContext')]:
   ax=axes[seed,col];target=None
   for j,arm in enumerate(ARMS):
    g=p[p.seed.eq(seed)&p['update'].eq(330)&p.cue.eq(cue)&p.arm.eq(arm)].sort_values('position')
    if g.empty:continue
    xx=g.position+(j-1)*.23;ax.bar(xx,g.ce,width=.22,color=COL[arm],alpha=.8);wrong=g[~g.correct];ax.scatter(wrong.position+(j-1)*.23,wrong.ce,marker='x',color='#b32222',s=25,zorder=5);target=g.target.tolist()
   ax.set_yscale('symlog',linthresh=.01);ax.set_ylim(bottom=0);ax.set_xticks(range(1,8),[f'{k}\n[{t}]' for k,t in zip(range(1,8),target)] if target else range(1,8));ax.set_xlabel('Position [target class]');ax.set_ylabel('Position CE (nats)');ax.set_title(cue+' at update 330')
 handles=[Line2D([0],[0],color='#2475b5',label='practiced A'),Line2D([0],[0],color='#e78a20',label='practiced B'),Line2D([0],[0],color='#777',ls='--',label='A00 practice'),Line2D([0],[0],color='#777',label='A06 practice')]+[Line2D([0],[0],color=COL[a],lw=5,label=a+' position CE') for a in ARMS]+[Line2D([0],[0],color='#b32222',ls='',marker='x',label='incorrect')]
 fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=8,frameon=False);fig.suptitle('Actual A/B acquisition and cached endpoint position errors',fontsize=13);fig.tight_layout(rect=(0,.1,1,.94));save(fig,'figure_2_split_and_position_errors')
