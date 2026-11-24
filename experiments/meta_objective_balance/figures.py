"""Five confirmation figure families and compact screen; saved tables only."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/meta_objective_balance/v1';MAIN=[f'figure_{i}' for i in range(1,6)];CC={'A':'#2166ac','B':'#ed8b23','C':'#ce3451','Mixed-average':'#27934c','Mixed-both':'#8a5bb4','NoContext':'#777777'}
plt.rcParams.update({'svg.hashsalt':'meta_objective_balance_v1','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':140})
def read(n):return pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip',low_memory=False)
def confirmation_cases(selected):return [(d,a) for d in [960,1920] for a in selected+['C']]
def save(fig,out,n):
 fig.tight_layout(rect=(0,.045,1,.95));fig.canvas.draw();fig.savefig(out/f'{n}.png',metadata={'Software':'meta_objective_balance_v1'});fig.savefig(out/f'{n}.svg',metadata={'Date':None,'Creator':'meta_objective_balance_v1'});plt.close(fig)
def pick(f,a,d):return f[f.arm.eq(a)&f.dose.eq(d)]
def band(ax,raw,ci,t,m,color,label=None):
 for _,g in raw.groupby('seed'):
  g=g.sort_values(t);ax.plot(g[t],g[m],color=color,alpha=.12,lw=.6)
 g=ci[ci.metric.eq(m)].sort_values(t);ax.plot(g[t],g['mean'],color=color,lw=1.6,label=label);ax.fill_between(g[t],g.ci_low,g.ci_high,color=color,alpha=.14,lw=0)
def points(ax,raw,summary,arms,metric,color,offset=0,label=None):
 for j,a in enumerate(arms):
  g=raw[raw.arm.eq(a)];x=j+offset;v=g[metric].astype(float);ax.scatter(x+np.linspace(-.06,.06,len(v)),v,color=color,s=11,alpha=.3)
  r=summary[summary.arm.eq(a)&summary.metric.eq(metric)]
  if len(r):
   r=r.iloc[0];ax.errorbar(x,r['mean'],yerr=np.array([[max(0,r['mean']-r.ci_low)],[max(0,r.ci_high-r['mean'])]]),fmt='o',color=color,capsize=3,label=label if j==0 else None)
 ax.set_xticks(range(len(arms)),arms);ax.set_xlabel('Objective condition')

def screen(out):
 f=read('screen_seed_outcomes');f=f[f.dose.eq(480)];s=read('screen_outcome_summary');s=s[s.dose.eq(480)];arms=list('ABCDEF');fig,axs=plt.subplots(1,3,figsize=(14,4));fig.suptitle('Exploratory six-seed screen: fixed objective conditions')
 for ax,m,title in zip(axs,['gap','conflict_accuracy','early_specificity'],['C gap at target 330','Reserved conflict accuracy','First A/B mean signed specificity']):
  points(ax,f,s,arms,m,'#426a92');ax.set(title=title,xlabel='Condition',ylabel='Accuracy' if m=='conflict_accuracy' else 'Nats');ax.axhline(0,color='.6',lw=.6)
 fig.text(.5,.015,'Six paired screening seeds; pointwise seed-bootstrap intervals. C = balanced matched control; no confirmatory significance claim.',ha='center',fontsize=9);save(fig,out,'screen_summary')

def practice(out,arms):
 f=read('confirm_training');ci=read('confirm_training_intervals')
 for stop,suffix in [(330,''),(100,'_first100'),(20,'_first20')]:
  fig,axs=plt.subplots(2,3,figsize=(16,7));fig.suptitle(f'Actual A10/B10 pre-update practiced CE, repetitions 1–{stop}')
  for ax,(d,a) in zip(axs.flat,confirmation_cases(arms)):
   raw=pick(f,a,d);raw=raw[raw['update']<=stop];summary=pick(ci,a,d);summary=summary[summary['update']<=stop]
   for _,g in raw.groupby('seed'):ax.plot(g['update'],g.ce_before,color='.5',alpha=.1,lw=.5)
   ax.plot(summary['update'],summary['mean'],color='.5',lw=.7)
   for start in range(1,stop+1,10):
    c='A' if (start-1)//10%2==0 else 'B';x=raw[raw['update'].between(start,start+9)];z=summary[summary['update'].between(start,start+9)]
    for _,g in x.groupby('seed'):ax.plot(g['update'],g.ce_before,color=CC[c],alpha=.11,lw=.6)
    ax.plot(z['update'],z['mean'],color=CC[c],lw=1.6,label=c if start<=11 else None);ax.fill_between(z['update'],z.ci_low,z.ci_high,color=CC[c],alpha=.15)
   ax.set(title=f'Objective {a} | +{d} episodes',xlabel='Target repetition/update',ylabel='Pre-update CE (nats)');ax.legend(frameon=False)
  fig.text(.5,.012,'All 14 confirmation seeds; blue A/orange B. Connecting segments retain context-switch jumps; no smoothing.',ha='center');save(fig,out,'figure_1'+suffix)

def frozen(out,arms):
 f=read('confirm_evaluations');ci=read('confirm_evaluations_intervals')
 for stop,suffix in [(330,''),(120,'_first120')]:
  fig,axs=plt.subplots(2,3,figsize=(16,8));fig.suptitle('Frozen same-weight six-cue sequence CE')
  for ax,(d,a) in zip(axs.flat,confirmation_cases(arms)):
   for c,color in CC.items():
    x=pick(f,a,d);x=x[x.cue.eq(c)&x['update'].le(stop)];s=pick(ci,a,d);s=s[s.cue.eq(c)&s['update'].le(stop)];band(ax,x,s,'update','ce',color,c)
   ax.axhline(np.log(4),color='.5',ls=':',lw=.8);ax.set(title=f'Objective {a} | +{d} episodes',xlabel='Frozen target state',ylabel='CE (nats)');ax.legend(frameon=False,ncol=2,fontsize=7)
  fig.text(.5,.012,'All 14 seeds; whole-seed 95% intervals; dotted chance ln(4). No C/Mixed target practice.',ha='center');save(fig,out,'figure_2'+suffix)

def tradeoff(out,arms):
 f=read('confirm_seed_outcomes');s=read('confirm_outcome_summary');fig,axs=plt.subplots(1,2,figsize=(12,5));fig.suptitle('C transfer versus strictly unseen conflict learning')
 for ax,d in zip(axs,[960,1920]):
  for a,col,marker in zip(arms+['C'],['#3075ad','#d2802b','#555555'],['o','^','s']):
   g=pick(f,a,d);ax.scatter(g.gap,g.conflict_accuracy,color=col,marker=marker,alpha=.5,label=a);r=pick(s,a,d).set_index('metric');x=r.loc['gap'];y=r.loc['conflict_accuracy'];ax.errorbar(x['mean'],y['mean'],xerr=[[max(0,x['mean']-x.ci_low)],[max(0,x.ci_high-x['mean'])]],yerr=[[max(0,y['mean']-y.ci_low)],[max(0,y.ci_high-y['mean'])]],color=col,fmt=marker,ms=9,capsize=3)
  ax.set(title=f'+{d} episodes',xlabel='C CE − familiar mean CE (nats)',ylabel='Conflict-position accuracy');ax.legend(frameon=False);ax.axvline(0,color='.6',lw=.6)
 fig.text(.5,.015,'Each small marker is one trained seed (12 diagnostic tasks averaged first). Desired region: low gap, high accuracy.',ha='center');save(fig,out,'figure_3')

def functional(out,arms):
 b=read('confirm_blocks');ci=read('confirm_blocks_intervals');order=arms+['C'];fig,axs=plt.subplots(2,3,figsize=(15,7));fig.suptitle('First A10 and B10: raw gains and signed learning specificity')
 for i,context in enumerate(['A','B']):
  for j,metric in enumerate(['practiced_gain','unpracticed_gain','specificity']):
   ax=axs[i,j]
   for d,col,off in [(960,'#6f8aa6',-.13),(1920,'#b84f3f',.13)]:
    x=b[b.dose.eq(d)&b.end.eq(10 if context=='A' else 20)];s=ci[ci.dose.eq(d)&ci.end.eq(10 if context=='A' else 20)];points(ax,x,s,order,metric,col,off,'+'+str(d))
   ax.axhline(0,color='.5',lw=.7);ax.set(title=f'{context} practice: '+metric.replace('_',' '),ylabel='CE decrease / own−cross (nats)');ax.legend(frameon=False)
 fig.text(.5,.01,'Frozen start/end weights; signed effects retained. Floor flags and starting A/B losses are in confirm_blocks.csv.',ha='center');save(fig,out,'figure_4')
 sw=read('confirm_switch_costs');sc=read('confirm_switch_intervals')
 for metric,suffix in [('adjusted_cost','_switch_costs'),('raw_cost','_raw_switch_costs')]:
  fig,axs=plt.subplots(2,3,figsize=(16,7));fig.suptitle('Same-weight context-switch cost: '+metric)
  for ax,(d,a) in zip(axs.flat,confirmation_cases(arms)):
   for direction,col in [('A_to_B',CC['A']),('B_to_A',CC['B'])]:
    x=pick(sw,a,d);x=x[x.direction.eq(direction)];s=pick(sc,a,d);s=s[s.direction.eq(direction)];band(ax,x,s,'boundary',metric,col,direction)
   ax.axhline(0,color='.5',lw=.6);ax.set(title=f'Objective {a} | +{d}',xlabel='A/B boundary',ylabel='CE difference (nats)');ax.legend(frameon=False)
  save(fig,out,'figure_4'+suffix)
 m=read('confirm_matched_acquisition');ms=read('confirm_matched_summary');bins=['CE2-4','CE1-2','CE0.5-1','CE0.1-0.5']
 for stratum in ['all','A','B']:
  fig,axs=plt.subplots(2,4,figsize=(17,7));fig.suptitle('First eligible matched-acquisition block: '+stratum)
  for i,d in enumerate([960,1920]):
   for j,rng in enumerate(bins):
    ax=axs[i,j];x=m[m.dose.eq(d)&m.stratum.eq(stratum)&m.performance_range.eq(rng)];s=ms[ms.dose.eq(d)&ms.stratum.eq(stratum)&ms.performance_range.eq(rng)];points(ax,x,s,order,'specificity','#456d94');ax.axhline(0,color='.6',lw=.6);ns=[int(x[x.arm.eq(a)].specificity.notna().sum()) for a in order];ax.set(title=f'+{d} | {rng}\neligible n={ns}',ylabel='Signed specificity (nats)')
  fig.text(.5,.01,'Fixed bins; unavailable seeds remain missing, not zero. Conditional comparison is not causal mediation.',ha='center');save(fig,out,'figure_4_matched_'+stratum)
 raw=read('confirm_learning_effects');rs=read('confirm_diagnostic_intervals');fig,axs=plt.subplots(2,3,figsize=(16,7));fig.suptitle('Restored-Adam one-update specificity; main trajectories unchanged')
 for ax,(d,a) in zip(axs.flat,confirmation_cases(arms)):
  for metric,col in [('D_A',CC['A']),('D_B',CC['B'])]:band(ax,pick(raw,a,d),pick(rs,a,d),'update',metric,col,metric)
  ax.axhline(0,color='.5',lw=.6);ax.set(title=f'Objective {a} | +{d}',xlabel='Frozen checkpoint',ylabel='Own−cross CE improvement');ax.legend(frameon=False)
 save(fig,out,'figure_4_restored_adam')

def dynamics(out,arms):
 raw=read('confirm_meta_windows');ci=read('confirm_meta_window_intervals');metrics=['shared_query_ce','conflict_query_ce','conflict_critical_ce','shared_norm','conflict_norm','combined_norm','gradient_cosine','clipped'];fig,axs=plt.subplots(2,4,figsize=(18,8));fig.suptitle('Paired FOMAML dynamics: fixed 60-pair windows')
 for ax,m in zip(axs.flat,metrics):
  for a,col in zip(arms+['C'],['#3075ad','#d2802b','#555555']):band(ax,raw[raw.arm.eq(a)],ci[ci.arm.eq(a)],'window_end',m,col,a)
  if m=='gradient_cosine':ax.axhline(0,color='.6',lw=.7)
  ax.axvline(480,color='.6',lw=.6,ls=':');ax.set(title=m.replace('_',' '),xlabel='Additional paired outer updates',ylabel='Fraction' if m=='clipped' else m.replace('_',' '));ax.legend(frameon=False)
 fig.text(.5,.015,'All 14 paired seeds. Component norms precede weighting; combined norm precedes one clip. Cosine: shared vs beta-weighted conflict.',ha='center');save(fig,out,'figure_5')

def safe_destination(value):
 out=Path(value) if value is not None else OUT/'figures'
 if (OUT/'manifest.json').exists() and out.resolve().is_relative_to(OUT.resolve()):raise ValueError('Sealed figures require an external output directory')
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);p.add_argument('--output-dir',type=Path);a=p.parse_args();out=safe_destination(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 if a.stage=='screen':screen(out)
 else:
  arms=json.loads((OUT/'audit/selection_seal.json').read_text())['selected']
  for fn in [practice,frozen,tradeoff,functional,dynamics]:fn(out,arms)
if __name__=='__main__':main()
