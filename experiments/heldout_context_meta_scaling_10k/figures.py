"""Six prespecified figure families; saved tables only, no checkpoint evaluation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/heldout_context_meta_scaling_10k/v1'
from .protocol import DOSES
MAIN=[f'figure_{i}' for i in range(1,7)];DETAIL=[960,3840,7680,10000];OTHER=[0,240,1920,5760];CONDS=['P1-I','P1-C','P2-I','P2-C'];COLORS={'A':'#2166ac','B':'#ef8a24','C':'#d62748','Mixed-average':'#26954b','Mixed-both':'#8e63ad','NoContext':'#777777'};ARM={'Seen':'--','Transfer':'-','Historical':':'};AC={'Seen':'#3279b8','Transfer':'#cb334c','Historical':'#777777'}
plt.rcParams.update({'svg.hashsalt':'heldout_context_meta_scaling_10k_v1','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':140})
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def plotting_cases(d):return [('Historical',0)] if d==0 else [('Seen',d),('Transfer',d)]
def pick(f,c,a,d):return f[(f.condition==c)&(f.arm==a)&(f.dose==d)]
def save(fig,out,name):
 fig.tight_layout(pad=1.2,rect=(0,.045,1,.965));fig.canvas.draw();fig.savefig(out/f'{name}.png',metadata={'Software':'heldout_context_meta_scaling_10k_v1'});fig.savefig(out/f'{name}.svg',metadata={'Date':None,'Creator':'heldout_context_meta_scaling_10k_v1'});plt.close(fig)
def dose_axis(ax,doses=DOSES):
 ax.set_xscale('symlog',linthresh=240);ax.set_xlim(0,10000);ax.set_xticks(doses,[str(d) for d in doses],rotation=40,fontsize=7);ax.set_xlabel('Cumulative FOMAML episodes (0 = Reptile baseline)')
def band(ax,raw,summary,time,metric,color,label=None,style='-'):
 for _,s in raw.groupby('seed'):
  s=s.sort_values(time);ax.plot(s[time],s[metric],color=color,alpha=.07,lw=.5)
 s=summary[summary.metric==metric].sort_values(time);ax.plot(s[time],s['mean'],color=color,ls=style,label=label,lw=1.4);ax.fill_between(s[time],s.ci_low,s.ci_high,color=color,alpha=.11,lw=0)
def grid(title,n=4):
 fig,axs=plt.subplots(n,4,figsize=(19,3*n),squeeze=False);fig.suptitle(title);return fig,axs

def practice(out):
 raw=read('training');ci=read('training_intervals')
 for doses,suffix in [(DETAIL,''),(OTHER,'_other_doses')]:
  for stop,view in [(330,''),(100,'_early'),(20,'_first20')]:
   fig,axs=grid(f'Actual A10/B10 pre-update loss, repetitions1–{stop}; dashed Seen / solid Transfer')
   for i,d in enumerate(doses):
    for j,c in enumerate(CONDS):
     ax=axs[i,j]
     for a,dd in plotting_cases(d):
      x=pick(raw,c,a,dd);x=x[x['update']<=stop];z=pick(ci,c,a,dd);z=z[z['update']<=stop]
      for _,s in x.groupby('seed'):ax.plot(s['update'],s.ce_before,color='.5',alpha=.08,lw=.5)
      ax.plot(z['update'],z['mean'],color='.5',alpha=.4,lw=.6)
      for start in range(1,stop+1,10):
       cue='A' if (start-1)%20==0 else 'B';s=z[z['update'].between(start,start+9)];ax.plot(s['update'],s['mean'],color=COLORS[cue],ls=ARM[a],label=f'{cue} {a}' if start<=11 else None,lw=1.5);ax.fill_between(s['update'],s.ci_low,s.ci_high,color=COLORS[cue],alpha=.12,lw=0)
     ax.set(title=f'{c} | FOMAML{d}',xlabel='Target update',ylabel='Actual practiced CE (nats)');ax.legend(frameon=False,ncol=2,fontsize=6)
   fig.text(.5,.01,'All20 seeds per arm; whole-seed95% intervals. Connecting segments preserve context-switch jumps.',ha='center');save(fig,out,'figure_1'+suffix+view)

def frozen(out):
 raw=read('evaluations');ci=read('evaluation_intervals')
 for doses,suffix in [(DETAIL,''),(OTHER,'_other_doses')]:
  maxima={c:raw[(raw.condition==c)&raw.dose.isin(doses)].ce.max() for c in CONDS}
  for stop,view in [(330,''),(120,'_early'),(20,'_first20')]:
   fig,axs=grid(f'Frozen six-cue predictions, target states0–{stop}; dashed Seen / solid Transfer')
   for i,d in enumerate(doses):
    for j,c in enumerate(CONDS):
     ax=axs[i,j]
     for a,dd in plotting_cases(d):
      x=pick(raw,c,a,dd);z=pick(ci,c,a,dd)
      for cue,color in COLORS.items():band(ax,x[(x.cue==cue)&x['update'].le(stop)],z[(z.cue==cue)&z['update'].le(stop)],'update','ce',color,cue if a in ['Seen','Historical'] else None,ARM[a])
     ax.axhline(np.log(4),color='.4',ls=':',lw=.7);ax.set(title=f'{c} | FOMAML{d}',xlabel='Frozen target state',ylabel='CE (nats)',ylim=(-.02*maxima[c],1.05*maxima[c]));ax.legend(frameon=False,ncol=3,fontsize=6)
   fig.text(.5,.01,'Same y-axis within each model-family column across doses and views. All seeds retained; chance=ln4.',ha='center');save(fig,out,'figure_2'+suffix+view)

def series_with_baseline(raw,summary,c,arm):
 x=raw[(raw.condition==c)&((raw.arm==arm)|(raw.arm=='Historical'))];s=summary[(summary.condition==c)&((summary.arm==arm)|(summary.arm=='Historical'))];return x,s

def scaling(out):
 f=read('endpoints');f=f[f['update'].eq(330)];ci=read('endpoint_summary');ci=ci[ci['update'].eq(330)];aq=read('acquisition');aci=read('post_acquisition_summary');mi=read('gap_median_intervals')
 for org,suffix in [('I',''),('C','_centered')]:
  fig,axs=plt.subplots(2,3,figsize=(17,9));fig.suptitle(f'FOMAML scaling: {"independent" if org=="I" else "initially centered"} A/B organization')
  for ax,(metric,cue,title) in zip(axs.flat,[('ce','C','C CE (nats)'),('gap','C','C minus familiar CE (nats)'),('median_gap','C','Median C gap (nats)'),('strict_all','C','Strict C fraction (all three targets)'),('ce','Mixed-both','Mixed-both CE (nats)'),('first_familiar','C','First strong A/B acquisition (updates)')]):
   for prior,color in [('P1','#2166ac'),('P2','#d87722')]:
    c=prior+'-'+org
    for a in ['Seen','Transfer']:
     if metric=='first_familiar':x,s=series_with_baseline(aq,aci,c,a);band(ax,x,s,'dose',metric,color,prior+' '+a,ARM[a])
     elif metric=='median_gap':
      x,s=series_with_baseline(f[f.cue==cue],ci[(ci.cue==cue)&ci.metric.eq('gap')],c,a);s=s.merge(mi,on=['condition','arm','dose'],validate='one_to_one');s=s.sort_values('dose')
      for _,v in x.groupby('seed'):ax.plot(v.sort_values('dose').dose,v.sort_values('dose').gap,color=color,alpha=.05,lw=.4)
      ax.plot(s.dose,s['median'],color=color,ls=ARM[a],label=prior+' '+a);ax.fill_between(s.dose,s.median_ci_low,s.median_ci_high,color=color,alpha=.12,lw=0)
     else:x,s=series_with_baseline(f[f.cue==cue],ci[ci.cue==cue],c,a);band(ax,x,s,'dose',metric,color,prior+' '+a,ARM[a])
   if metric=='gap' or metric=='median_gap':ax.axhline(0,color='black',lw=.6)
   if metric=='ce':ax.axhline(np.log(4),color='.4',ls=':',lw=.7)
   if metric=='strict_all':ax.set_ylim(-.03,1.03)
   ax.set_ylabel(title);dose_axis(ax);ax.legend(frameon=False,fontsize=7,ncol=2)
  fig.text(.5,.01,'Historical0 is the same reused baseline for both arms, not two replications. Paired20-seed intervals; no selected doses.',ha='center');save(fig,out,'figure_3'+suffix)
 raw=read('scaling_seed_effects');ci=read('scaling_intervals');raw=raw[raw.cue.eq('C')&raw['update'].eq(330)&raw.metric.eq('gap')];ci=ci[ci.cue.eq('C')&ci['update'].eq(330)&ci.metric.eq('gap')];fig,axs=plt.subplots(2,2,figsize=(13,8));fig.suptitle('Paired C-gap effects across FOMAML doses')
 for ax,c in zip(axs.flat,CONDS):
  x=raw[(raw.condition==c)&raw.contrast.eq('Transfer-Seen')].rename(columns={'effect':'effect_value'});s=ci[(ci.condition==c)&ci.contrast.eq('Transfer-Seen')].copy();s['metric']='effect_value';band(ax,x,s,'dose','effect_value',AC['Transfer'],'Transfer−Seen');ax.axhline(0,color='black',lw=.7);ax.set(title=c,ylabel='Paired C-gap difference (nats)');dose_axis(ax);ax.legend(frameon=False)
 save(fig,out,'figure_3_paired_effects')

def functional(out):
 b=read('block_transfer');bi=read('block_intervals')
 for metric,name,title in [('specificity','figure_4','First A/B block: signed own-minus-cross gains'),('practiced_gain','figure_4_own','First A/B block: practiced-context improvement'),('unpracticed_gain','figure_4_cross','First A/B block: unpracticed-context improvement')]:
  fig,axs=plt.subplots(2,4,figsize=(19,7));fig.suptitle(title)
  for row,(cue,end) in enumerate([('A',10),('B',20)]):
   for ax,c in zip(axs[row],CONDS):
    for a in ['Seen','Transfer']:
     x,s=series_with_baseline(b[b.end.eq(end)],bi[bi.end.eq(end)],c,a);band(ax,x,s,'dose',metric,AC[a],a,ARM[a])
    ax.axhline(0,color='black',lw=.6);ax.set(title=f'{c} | {cue} practice',ylabel=f'{metric.replace("_"," ")} (nats)');dose_axis(ax);ax.legend(frameon=False)
  fig.text(.5,.01,'Fixed first A0→10 and B10→20 blocks. Actual start/end losses and floor flags in block_transfer.csv.',ha='center');save(fig,out,name)
 m=read('matched_acquisition');ms=read('matched_summary')
 for stratum,suffix in [('all',''),('A','_A'),('B','_B')]:
  fig,axs=plt.subplots(4,4,figsize=(19,13));fig.suptitle('First eligible matched-acquisition specificity | '+stratum)
  for i,rng in enumerate(['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5']):
   for ax,c in zip(axs[i],CONDS):
    for a in ['Seen','Transfer']:
     x,s=series_with_baseline(m[(m.performance_range==rng)&m.stratum.eq(stratum)],ms[(ms.performance_range==rng)&ms.stratum.eq(stratum)],c,a);band(ax,x,s,'dose','specificity',AC[a],a,ARM[a])
    ax.axhline(0,color='black',lw=.6);ax.set(title=f'{c} | {rng}',ylabel='Signed own−cross (nats)');dose_axis(ax);ax.legend(frameon=False)
  fig.text(.5,.01,'Missing eligible bins remain missing. Conditional paired denominators and separate A/B start losses are saved.',ha='center');save(fig,out,'figure_4_matched'+suffix)
 sw=read('switch_costs');sc=read('switch_intervals');fig,axs=plt.subplots(2,4,figsize=(19,7));fig.suptitle('Early same-weight baseline-adjusted context switch costs')
 for i,k in enumerate([10,20]):
  for ax,c in zip(axs[i],CONDS):
   for a in ['Seen','Transfer']:
    x,s=series_with_baseline(sw[sw.boundary.eq(k)],sc[sc.boundary.eq(k)],c,a);band(ax,x,s,'dose','adjusted_cost',AC[a],a,ARM[a])
   ax.axhline(0,color='black',lw=.6);ax.set(title=f'{c} | {"A→B" if k==10 else "B→A"}',ylabel='Adjusted switch cost (nats)');dose_axis(ax);ax.legend(frameon=False)
 save(fig,out,'figure_4_switches')
 raw=read('learning_effects');ci=read('directional_probe_summary')
 for k in [20,40,80,120,170,330]:
  fig,axs=plt.subplots(2,4,figsize=(19,7));fig.suptitle(f'Restored-Adam one-update effects at frozen target{k}')
  for i,(cue,metric) in enumerate([('A','D_A'),('B','D_B')]):
   for ax,c in zip(axs[i],CONDS):
    for a in ['Seen','Transfer']:
     x,s=series_with_baseline(raw[raw['update'].eq(k)],ci[ci['update'].eq(k)],c,a);band(ax,x,s,'dose',metric,AC[a],a,ARM[a])
    ax.axhline(0,color='black',lw=.6);ax.set(title=f'{c} | {cue} update',ylabel='Signed own−cross gain (nats)');dose_axis(ax);ax.legend(frameon=False)
  fig.text(.5,.01,'Original endpoint criterion unchanged. Small late effects at a performance floor do not establish shared learning.',ha='center');save(fig,out,f'figure_4_restored_adam_t{k}')
  fig,axs=plt.subplots(2,4,figsize=(19,7));fig.suptitle(f'Restored-Adam own versus cross gains at target{k}')
  for i,(cue,own,cross) in enumerate([('A','delta_A_A','delta_B_A'),('B','delta_B_B','delta_A_B')]):
   for ax,c in zip(axs[i],CONDS):
    for a in ['Seen','Transfer']:
     x,s=series_with_baseline(raw[raw['update'].eq(k)],ci[ci['update'].eq(k)],c,a)
     for metric,ls,label in [(own,ARM[a],'own'),(cross,':','cross')]:band(ax,x,s,'dose',metric,AC[a],a+' '+label,ls)
    ax.axhline(0,color='black',lw=.6);ax.set(title=f'{c} | {cue} update',ylabel='CE improvement (nats)');dose_axis(ax);ax.legend(frameon=False,ncol=2,fontsize=6)
  save(fig,out,f'figure_4_restored_gains_t{k}')
 for d in DETAIL:
  fig,axs=plt.subplots(2,4,figsize=(19,7));fig.suptitle(f'All330 target updates: signed block specificity | FOMAML{d}')
  for i,cue in enumerate(['A','B']):
   for ax,c in zip(axs[i],CONDS):
    for a in ['Seen','Transfer']:
     x=pick(b,c,a,d);x=x[x.practiced==cue];s=pick(bi,c,a,d);s=s[s.practiced==cue];band(ax,x,s,'end','specificity',AC[a],a,ARM[a]);floors=x[x.floor_at_start];ax.plot(floors.end,np.zeros(len(floors)),color=AC[a],marker='|',ls='none',alpha=.08)
    ax.axhline(0,color='black',lw=.6);ax.set(title=f'{c} | {cue} practice',xlabel='End of10-update block',ylabel='Signed own−cross (nats)');ax.legend(frameon=False)
  save(fig,out,f'figure_4_blocks_d{d}')
 eu=read('early_updates');ec=read('early_update_intervals')
 for d in DETAIL:
  fig,axs=plt.subplots(1,4,figsize=(19,4));fig.suptitle(f'First20 individual target updates | FOMAML{d}')
  for ax,c in zip(axs,CONDS):
   for a in ['Seen','Transfer']:
    for cue in ['A','B']:
     x=pick(eu,c,a,d);s=pick(ec,c,a,d);band(ax,x[x.practiced==cue],s[s.practiced==cue],'update','specificity',COLORS[cue],cue+' '+a,ARM[a])
   ax.axvline(10.5,color='.5',lw=.5);ax.axhline(0,color='black',lw=.6);ax.set(title=c,xlabel='Target update',ylabel='Signed own−cross (nats)');ax.legend(frameon=False,ncol=2,fontsize=6)
  save(fig,out,f'figure_4_first20_d{d}')

def joint(out):
 e=read('endpoints');e=e[e.cue.eq('C')&e['update'].eq(330)];j=read('joint_early20');j=j[j['update'].eq(330)];f=j.merge(e[['seed','condition','arm','dose','gap']],on=['seed','condition','arm','dose'],validate='one_to_one');fig,axs=plt.subplots(2,2,figsize=(14,10));fig.suptitle('Joint transfer and first20-update functional learning (all seeds/doses)');cm=plt.colormaps['viridis'];colors={d:cm(i/(len(DOSES)-1)) for i,d in enumerate(DOSES)}
 for ax,c in zip(axs.flat,CONDS):
  for d in DOSES:
   for a,dd in plotting_cases(d):
    x=pick(f,c,a,dd);eligible=x.strong_familiar.eq(1)&x.early20_bidirectional.eq(1);marker={'Historical':'x','Seen':'^','Transfer':'o'}[a]
    for flag in [True,False]:
     z=x[eligible==flag];ax.scatter(z.early20_specificity,z.gap,s=22,marker=marker,color=colors[d] if flag or a=='Historical' else None,facecolors=None if flag or a=='Historical' else 'none',edgecolors=colors[d] if a!='Historical' else None,alpha=.6,linewidths=.7)
  ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6);ax.set(title=c,xlabel='Mean signed first A/B-block specificity (nats)',ylabel='C−familiar CE at target330 (nats)')
 handles=[Line2D([],[],marker='o',ls='none',color=colors[d],label=str(d)) for d in DOSES];fig.legend(handles=handles,loc='lower center',ncol=8,title='Cumulative FOMAML episodes');fig.text(.5,.07,'Circles Transfer; triangles Seen; crosses historical. Filled: strong A/B and positive nonfloor own/specificity in both first blocks.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.14,1,.965));fig.canvas.draw();fig.savefig(out/'figure_5.png',metadata={'Software':'heldout_context_meta_scaling_10k_v1'});fig.savefig(out/'figure_5.svg',metadata={'Date':None,'Creator':'heldout_context_meta_scaling_10k_v1'});plt.close(fig)
 js=read('joint_early20_summary');fig,axs=plt.subplots(2,4,figsize=(19,7));fig.suptitle('Descriptive first20 coexistence versus unchanged original endpoint gate')
 for i,(metric,title) in enumerate([('early20_coexistence','Familiar + C + Mixed-both + bidirectional first-block learning'),('functional_split','Original functional splitting criterion')]):
  for ax,c in zip(axs[i],CONDS):
   for a in ['Seen','Transfer']:
    x,s=series_with_baseline(j,js[js['update'].eq(330)],c,a);band(ax,x,s,'dose',metric,AC[a],a,ARM[a])
   ax.set(title=c,ylabel=title,ylim=(-.03,1.03));dose_axis(ax);ax.legend(frameon=False)
 save(fig,out,'figure_5_joint_rates')

def conflict(out):
 raw=read('heldout_seed');ci=read('heldout_summary');wrong=read('heldout_wrong_seed');wc=read('heldout_wrong_summary');doses=[0,960,3840,7680,10000]
 for org,suffix in [('I',''),('C','_centered')]:
  for extra,metrics in [(False,[('ce','Overall CE (nats)'),('conflict_ce','Conflict-position CE (nats)'),('conflict_accuracy','Conflict-position accuracy'),('appropriate_context_change','Both conflicting responses appropriate'),('correct_gain','Correct-context CE improvement'),('wrong_gain','Wrong-context CE improvement')]),(True,[('correct_CE_after','Correct-context CE'),('wrong_CE_after','Wrong-context CE'),('context_benefit','Wrong−correct CE'),('conflict_correct_ce','Correct conflict-position CE'),('conflict_wrong_ce','Wrong conflict-position CE'),('context_gain_advantage','Correct−wrong learning gain')])]:
   fig,axs=plt.subplots(2,3,figsize=(17,9));fig.suptitle(f'Reserved unseen conflict tasks, {org} organization;20 diagnostic adaptation updates')
   for ax,(metric,title) in zip(axs.flat,metrics):
    rr,ss=(wrong,wc) if metric in wrong.columns else (raw,ci)
    for prior,color in [('P1','#2166ac'),('P2','#d87722')]:
     for a in ['Seen','Transfer']:
      x,s=series_with_baseline(rr,ss,prior+'-'+org,a);band(ax,x,s,'dose',metric,color,prior+' '+a,ARM[a])
    ax.set_ylabel(title);dose_axis(ax,doses);ax.legend(frameon=False,ncol=2,fontsize=7)
   fig.text(.5,.01,'12 fixed corrected-v2 tasks averaged within each of20 seeds. Historical/960 reused; no diagnostic-based training selection.',ha='center');save(fig,out,'figure_6'+suffix+('_context_effects' if extra else ''))

def mechanisms(out):
 raw=read('geometry');ci=read('geometry_summary');raw=raw[raw['update'].eq(330)];ci=ci[ci['update'].eq(330)]
 for metric in ['baseline_ce','C_ce','C_minus_avg_centered_norm','C_minus_baseline_CE','u_norm','v_norm','C_response_norm','C_norm','C_common_cosine','K_centered_norm','recurrent_spectral_norm','hidden_separation','centered_logit_separation']:
  fig,axs=plt.subplots(1,4,figsize=(19,4));fig.suptitle('Saved Linear mechanism at target330: '+metric.replace('_',' '))
  for ax,c in zip(axs,CONDS):
   for a in ['Seen','Transfer']:
    x,s=series_with_baseline(raw,ci,c,a);band(ax,x,s,'dose',metric,AC[a],a,ARM[a])
   ax.set(title=c,ylabel=metric.replace('_',' '));dose_axis(ax);ax.legend(frameon=False)
  save(fig,out,'mechanism_'+metric)

def draw_all(out=None):
 out=Path(out or OUT/'figures');out.mkdir(parents=True,exist_ok=True)
 for fn in [practice,frozen,scaling,functional,joint,conflict,mechanisms]:fn(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);a=p.parse_args();draw_all(a.output_dir)
