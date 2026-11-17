"""Five deterministic scientific figures. Saved CSVs only; no source/model calls."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
NAMES=['A','B','Mixed']
FIGURES=['fig_1_state_by_cue','fig_2_error_by_lag','fig_3_gap_components','fig_4_selective_repair','fig_5_reverse_controls']
COLORS=['#2166ac','#b35806','#333333'];STYLES=['-','--',':']


def data(src,name):return pd.read_csv(Path(src)/'tables'/f'{name}.csv',float_precision='round_trip')

def _dots(ax,values,x,label=None,color='#2166ac'):
 vals=np.asarray(values,float);vals=vals[np.isfinite(vals)]
 if len(vals):
  jitter=np.linspace(-.13,.13,len(vals));ax.scatter(x+jitter,vals,s=18,facecolors='none',edgecolors=color,lw=.8,zorder=3,label=label);ax.plot([x-.18,x+.18],[np.mean(vals)]*2,color=color,lw=2)
 return len(vals)

def generate(src,destination):
 dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
 for n in FIGURES:
  for ext in ['png','svg']:
   if (dest/(n+'.'+ext)).exists():raise FileExistsError('Immutable figure destination')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'context-failure-selective-repair-v1','figure.facecolor':'white','savefig.facecolor':'white'})
 paths=[]
 def export(fig,name):
  for ext in ['png','svg']:
   p=dest/(name+'.'+ext)
   fig.savefig(p,dpi=170,metadata={'Software':'context_failure_selective_repair_v1'} if ext=='png' else {'Date':None,'Creator':'context_failure_selective_repair_v1'})
   paths.append(str(p.relative_to(dest)))
  plt.close(fig)
 cells=data(src,'r1_cells');r=data(src,'r1_branch_rollouts');g=data(src,'r2_natural_gap_components');con=data(src,'r3_seed_contrasts');f=data(src,'r3_interventions');rev=data(src,'r3_reverse');fam=data(src,'r3_familiar_controls')
 # Figure1: exact factorial, no best reference, full20 dots.
 fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2,height_ratios=[1,.9],hspace=.45,wspace=.3);c=cells[cells.k==3]
 for col,endpoint in enumerate(['current_ce','future_ce']):
  ax=fig.add_subplot(gs[0,col]);matrix=c.groupby(['source_context','branch'])[endpoint].mean().unstack().reindex(index=NAMES,columns=NAMES);im=ax.imshow(matrix.to_numpy(),cmap='Greys',vmin=0,vmax=max(3.,matrix.to_numpy().max()));fig.colorbar(im,ax=ax,label='Mean CE (nats)',shrink=.8)
  for i,source in enumerate(NAMES):
   for j,cue in enumerate(NAMES):
    cc=c[(c.source_context==source)&(c.branch==cue)];text=f'{matrix.iloc[i,j]:.3f}'
    if endpoint=='future_ce':text+='\n'+str(int(cc.viable_continuation.sum()))+'/20 viable'
    ax.text(j,i,text,ha='center',va='center',color='white' if matrix.iloc[i,j]>im.norm.vmax*.5 else 'black',fontsize=10)
  ax.set(xticks=range(3),xticklabels=NAMES,yticks=range(3),yticklabels=NAMES,xlabel='One current cue at k=3',ylabel='Natural inherited history',title='Current prediction (k)' if col==0 else 'Unassisted future (k+1…7)')
 ax=fig.add_subplot(gs[1,:]);labels=[]
 for i,source in enumerate(NAMES):
  for j,cue in enumerate(NAMES):
   pos=i*3+j;cc=c[(c.source_context==source)&(c.branch==cue)];_dots(ax,cc.future_ce,pos,color=COLORS[i]);labels.append(source+' / '+cue)
 ax.axhline(0,color='.75',lw=.7);ax.set(xticks=range(9),xticklabels=labels,ylabel='Future CE (nats)',xlabel='History / current cue; every future cue is Mixed',title='Each open circle is one of the 20 trained seeds; horizontal mark = mean')
 fig.suptitle('State × cue factorial under a common physical Mixed future',fontsize=15);fig.subplots_adjust(top=.90,bottom=.1);export(fig,FIGURES[0])
 # Figure2: current separated visually from future, full step loss/margin.
 fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True);q=r[r.k==3]
 for col,source in enumerate(NAMES):
  for j,cue in enumerate(NAMES):
   qq=q[(q.source_context==source)&(q.branch==cue)]
   for row,metric in enumerate(['raw_ce','correct_target_margin']):
    ax=axs[row,col]
    for _,s in qq.groupby('seed'):ax.plot(s.lag,s[metric],color=COLORS[j],alpha=.08,lw=.6)
    mean=qq.groupby('lag')[metric].mean();ax.plot(mean.index,mean,color=COLORS[j],ls=STYLES[j],lw=2,label='current '+cue)
  axs[0,col].set_title(source+' inherited history (20 seeds)')
  for ax in axs[:,col]:ax.axvspan(-.15,.15,color='.9',zorder=0);ax.axvline(.5,color='.6',ls='--',lw=.8);ax.set_xticks(range(5));ax.grid(axis='y',alpha=.15)
  axs[1,col].axhline(0,color='.5',lw=.8);axs[1,col].set_xlabel('Lag (0=current,1…4=Mixed future)')
 axs[0,0].set_ylabel('Correct-target CE (nats)');axs[1,0].set_ylabel('Correct-target margin (logits)');axs[0,2].legend(fontsize=9)
 fig.suptitle('One current prediction and its subsequent error trajectory (k=3)',fontsize=15);fig.tight_layout(rect=(0,.02,1,.94));export(fig,FIGURES[1])
 # Figure3: both fixed reference sides averaged within seed, not norms as CE causation.
 fig,axs=plt.subplots(2,2,figsize=(12,8));s=data(src,'r2_future_sensitivity').drop_duplicates(['seed','k'])
 for col,k in enumerate([3,5]):
  a=g[(g.k==k)&(g.reference_kind=='factorial')].groupby('seed').mean(numeric_only=True)
  for j,name in enumerate(['gap','visible','silent','U','W']):_dots(axs[0,col],a[name+'_norm'],j,color=COLORS[j%3])
  axs[0,col].set(xticks=range(5),xticklabels=['Full gap','Visible V','Silent N','Sensitive U','Complement W'],ylabel='Euclidean norm',title=f'k={k}: natural Mixed→fixed A/B reference gaps')
  vals=s[s.k==k][[f'singular_{j}' for j in range(5)]].to_numpy()
  for v in vals:axs[1,col].semilogy(range(1,6),v,color='.6',alpha=.5,lw=.8)
  axs[1,col].semilogy(range(1,6),vals.mean(0),'o-',color=COLORS[0],lw=2,label='Mean across 20 seeds');axs[1,col].axvline(2.5,color='.4',ls='--');axs[1,col].set(xlabel='Singular direction within current-output kernel',ylabel='Singular value of G_fam N',xticks=range(1,6));axs[1,col].legend(fontsize=8)
 fig.suptitle('Current invisibility and familiar future sensitivity are different measurements',fontsize=14);fig.text(.5,.015,'Dots: trained-seed means of both fixed references. Norm fractions are not fractions of CE caused. U dimension fixed at 2.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,FIGURES[2])
 # Figure4: sparse partial-state support is not hidden by full-reference rescue dots.
 fig,axs=plt.subplots(2,3,figsize=(14,8))
 for row,k in enumerate([3,5]):
  cc=con[(con.group=='forward')&(con.k==k)];counts=[]
  for j,name in enumerate(['U_minus_natural','U_minus_W','U_minus_random']):
   values=cc[cc.contrast==name].effect;counts.append(_dots(axs[row,0],values,j,color=COLORS[j]))
  axs[row,0].axhline(0,color='.5',lw=.8);axs[row,0].ticklabel_format(axis='y',style='sci',scilimits=(-2,2));axs[row,0].set(xticks=range(3),xticklabels=[f'U−natural\nn={counts[0]}/20',f'U−W\nn={counts[1]}/20',f'U−random\nn={counts[2]}/20'],ylabel='Paired future CE difference (nats)',title=f'k={k}: conditional estimable effects')
  ref=cc[cc.contrast=='reference_minus_natural'];_dots(axs[row,1],ref.effect,0,color=COLORS[2]);axs[row,1].axhline(0,color='.5',lw=.8);axs[row,1].set(xticks=[0],xticklabels=['Full observed A/B reference\nn=20 seeds (both sides averaged)'],ylabel='Future CE−natural (nats)',title='Full-state bound, separate scale')
  arms=['natural','reference','visible','silent','U','W','random'];nn=[]
  for arm in arms:nn.append(f[(f.k==k)&(f.arm==arm)&(f.status=='valid')].seed.nunique())
  axs[row,2].barh(range(7),nn,color='.65',edgecolor='black');axs[row,2].set(yticks=range(7),yticklabels=arms,xlim=(0,22),xlabel='Seeds with ≥1 valid reference /20',title='Feasibility coverage, no replacements')
  for j,v in enumerate(nn):axs[row,2].text(v+.2,j,str(v),va='center',fontsize=9)
 fig.suptitle('Selective repair gate failed; matched interventions were rarely admissible',fontsize=15);fig.text(.5,.015,'Positive CE differences are adverse. k3 has one estimable reference cell; k5 has six. No full-cohort selective effect is identified.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.05,1,.94));export(fig,FIGURES[3])
 # Figure5: no numeric reverse effects fabricated when all matched proposals fail.
 fig,axs=plt.subplots(2,2,figsize=(12,8))
 for row,k in enumerate([3,5]):
  base=rev[(rev.k==k)&(rev.arm=='baseline')&(rev.status=='valid')];valid=rev[(rev.k==k)&(rev.arm=='U')&(rev.status=='valid')]
  labels=['Originally viable\nreference cells','Valid reverse\nU/W/random'];nn=[len(base),len(valid)];axs[row,0].bar(range(2),nn,color=['.65','.3'],edgecolor='black');axs[row,0].set(xticks=range(2),xticklabels=labels,ylabel='Reference cells /40 planned',ylim=(0,24),title=f'k={k}: reverse future effects not estimable')
  for j,v in enumerate(nn):axs[row,0].text(j,v+.5,str(v),ha='center')
  counts=[];labels=[]
  for context in ['A','B']:
   for sign in [-1,1]:
    cc=fam[(fam.k==k)&(fam.arm=='U')&(fam.context==context)&(fam.sign==sign)&(fam.status=='valid')];counts.append(cc.seed.nunique());labels.append(context+(' −' if sign<0 else ' +'))
  axs[row,1].bar(range(4),counts,color='.65',edgecolor='black');axs[row,1].set(xticks=range(4),xticklabels=labels,ylim=(0,20),ylabel='Seeds with feasible matched controls /20',title=f'k={k}: familiar own-cue, both-sign controls')
  for j,v in enumerate(counts):axs[row,1].text(j,v+.4,str(v),ha='center')
 fig.suptitle('Fixed reverse sign retained; infeasibility is not a null behavioral effect',fontsize=14);fig.text(.5,.015,'No sign/amplitude search. Familiar controls do not establish sequence-position information; all statuses remain in source tables.',ha='center',fontsize=9);fig.tight_layout(rect=(0,.04,1,.94));export(fig,FIGURES[4])
 return paths
