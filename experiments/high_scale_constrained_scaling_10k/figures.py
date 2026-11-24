"""Seven scientific figure families from numerical tables only, never model code."""
import argparse,json,hashlib,os
from pathlib import Path
from functools import lru_cache
import numpy as np,pandas as pd
os.environ.setdefault('MPLCONFIGDIR','/tmp/hscs_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/hscs_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/high_scale_constrained_scaling_10k/v1'
FAMILIES=list(range(1,8));VIEWS=[330,100,20,990];ARMS=['T','B','O1','O2'];COLORS={'U':'#3366aa','P':'#c44e52','A':'#1f77b4','B':'#ff7f0e'};STYLES={'U':'--','P':'-'}
REPTILE_RETENTION_DOSES=[0,240,960,1920];EXPECTED_PAIRS=46
METRIC_FAMILIES={3:['D_A','D_B','early_cost','A_to_B','B_to_A','sustained_cost'],4:['C_ce','gap','Mixed_average_ce','Mixed_both_ce','NoContext_ce','strict_C','Mixed_both_success']}
LABELS={'D_A':'First-A specificity','D_B':'First-B specificity','early_cost':'Balanced early switch cost','A_to_B':'Adjusted A→B cost','B_to_A':'Adjusted B→A cost','sustained_cost':'Sustained cost, cycles 2–3','C_ce':'C CE','gap':'Signed C gap','Mixed_average_ce':'Mixed-average CE','Mixed_both_ce':'Mixed-both CE','NoContext_ce':'NoContext CE','strict_C':'Strict C fraction','Mixed_both_success':'Mixed-both success fraction','conflict_accuracy':'Conflict-position accuracy','critical_ce':'Critical-position CE','conflict_context_benefit':'Wrong-cue − correct-cue CE'}
plt.rcParams.update({'svg.hashsalt':'high_scale_constrained_scaling_10k_v1','font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
IDX=np.random.default_rng(20260925).integers(6,size=(10000,6))

@lru_cache(maxsize=32768)
def bounds(buf):
 x=np.frombuffer(buf,dtype=np.float64);draw=x[IDX];den=np.isfinite(draw).sum(1);means=np.divide(np.nansum(draw,axis=1),den,out=np.full(10000,np.nan),where=den>0)
 return np.nanmean(x),*np.nanquantile(means,[.025,.975])

def interval(x):
 x=np.asarray(x,dtype=np.float64);assert x.shape==(6,)
 return bounds(x.tobytes()) if np.isfinite(x).any() else (np.nan,np.nan,np.nan)

def trace(ax,g,x,y,c,label=None,zero=False):
 g=g.sort_values(x)
 for _,z in g.groupby('seed'):ax.plot(z[x],z[y],color=COLORS[c],ls=STYLES[c],alpha=.17,lw=.6,marker='.',ms=2)
 rows=[]
 for k,z in g.groupby(x):
  assert not z.seed.duplicated().any();rows.append([k,*interval(z.set_index('seed')[y].reindex(range(6)).to_numpy(float))])
 v=np.array(rows,float)
 if len(v):ax.plot(v[:,0],v[:,1],color=COLORS[c],ls=STYLES[c],lw=1.8,label=label or c);ax.fill_between(v[:,0],v[:,2],v[:,3],color=COLORS[c],alpha=.12)
 if zero:ax.axhline(0,color='k',lw=.45)

def save(fig,dest,name):
 fig.savefig(dest/(name+'.png'),dpi=150,metadata={'Software':'high_scale_constrained_scaling_10k/v1'});fig.savefig(dest/(name+'.svg'),metadata={'Date':None});plt.close(fig)

def destination(root,dest):
 p=Path(dest) if dest else root/'figures'
 if (root/'manifest.json').exists() and p.resolve().is_relative_to(root.resolve()):raise ValueError('Sealed figures: regenerate into an external destination')
 p.mkdir(parents=True,exist_ok=True);return p

def with_parent(out,arm,update=330):
 f=out[out['update'].eq(update)];return pd.concat([f[f.phase.eq('R')&f.dose.eq(1920)].assign(dose=0),f[f.phase.eq(arm)]],ignore_index=True)

def familiar_failures(g):return g.loc[g.strong_familiar.eq(0)]

def line_panels(g,metrics,x,title,dest,name,xlabel=None):
 fig,axes=plt.subplots(len(metrics),3,figsize=(14,2.65*len(metrics)),squeeze=False,sharex=True)
 for i,m in enumerate(metrics):
  for j,k in enumerate([4,5,6]):
   ax=axes[i,j]
   for c in ['U','P']:trace(ax,g[g.scale.eq(k)&g.condition.eq(c)],x,m,c,zero=True)
   ax.set_title(f'scale {k} · {LABELS.get(m,m)}');ax.set_xlabel(xlabel or ('FOMAML episodes (0 = own Reptile1920)' if x=='dose' else x));ax.set_ylabel('Fraction' if m in ['strict_C','Mixed_both_success','conflict_accuracy'] else 'Nats')
   if m in ['strict_C','Mixed_both_success','conflict_accuracy']:ax.set_ylim(-.05,1.05)
   if i==0 and j==0:ax.legend()
 fig.suptitle(title+'\nSix paired seeds; mean and pointwise 95% whole-seed bootstrap intervals',fontsize=11);fig.tight_layout(rect=(0,0,1,.965));save(fig,dest,name)

def main(source=OUT,output_dir=None):
 source=Path(source);dest=destination(source,output_dir);read=lambda n:pd.read_csv(source/f'tables/{n}.csv',float_precision='round_trip');out=read('outcomes');scales=read('scale_trajectory_plot');geom=read('meta_geometry');held=read('conflict_seed');training=read('training');training_ci=read('final_training_intervals');manifest=[]
 # Figure1: exact sampled outer-state RMS, with each seed's registered floor.
 for arm in ['R']+ARMS:
  fig,axes=plt.subplots(6,3,figsize=(14,16),sharex=True,squeeze=False)
  for i,group in enumerate(['token','context_A','context_B','context_C','output','recurrent_spectral_norm']):
   for j,k in enumerate([4,5,6]):
    ax=axes[i,j];z=scales[scales.phase.eq(arm)&scales.scale.eq(k)]
    if i==5:z=z[z.group.eq('token')];y=group
    else:z=z[z.group.eq(group)];y='projected_rms'
    for c in ['U','P']:
     g=z[z.condition.eq(c)]
     for _,ss in g.groupby('seed'):
      ax.plot(ss.meta_episode,ss[y],color=COLORS[c],ls=STYLES[c],alpha=.32,lw=.7)
      if i<5:ax.axhline(ss.floor.iloc[0],color='#777777',alpha=.13,lw=.6)
     means=g.groupby('meta_episode')[y].mean();ax.plot(means.index,means,color=COLORS[c],ls=STYLES[c],label=c,lw=1.8)
    ax.set_title(f'scale {k} · {group}');ax.set_xlabel(('Reptile' if arm=='R' else 'FOMAML')+' episodes');ax.set_ylabel('Spectral norm' if i==5 else 'Actual RMS')
    if i==0 and j==0:ax.legend()
  fig.suptitle(f'1 · {arm}: registered floors and actual weight-scale histories\nExact sampled states every60 outer steps plus registered doses; every step retained in scale_histories/*.npz',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));save(fig,dest,'figure_1_scale_'+arm)
 # Figure2: original pre-update practice curves, separate uninterrupted ten-trial blocks.
 for arm in ['R']+ARMS:
  dose=1920 if arm=='R' else 10000
  for stop in VIEWS:
   g=training[training.phase.eq(arm)&training.dose.eq(dose)&training['update'].le(stop)];fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True,squeeze=False);top=float(g.ce_before.max())*1.02
   for i,c in enumerate(['U','P']):
    for j,k in enumerate([4,5,6]):
     ax=axes[i,j];z=g[g.scale.eq(k)&g.condition.eq(c)];ci=training_ci[training_ci.phase.eq(arm)&training_ci.condition.eq(c)&training_ci.scale.eq(k)] if arm!='R' else None
     for start in range(1,stop+1,10):
      cue='A' if ((start-1)//10)%2==0 else 'B';end=min(start+9,stop);block=z[z['update'].between(start,end)]
      for _,ss in block.groupby('seed'):ax.plot(ss['update'],ss.ce_before,color=COLORS[cue],alpha=.16,lw=.6)
      if ci is None:
       rr=[]
       for t,ss in block.groupby('update'):rr.append([t,*interval(ss.set_index('seed').ce_before.reindex(range(6)).to_numpy(float))])
       v=np.array(rr,float);xx=v[:,0];mu=v[:,1];lo=v[:,2];hi=v[:,3]
      else:q=ci[ci['update'].between(start,end)].sort_values('update');xx=q['update'];mu=q['mean'];lo=q.ci_low;hi=q.ci_high
      ax.plot(xx,mu,color=COLORS[cue],lw=1.6,label=cue if start<=20 else None);ax.fill_between(xx,lo,hi,color=COLORS[cue],alpha=.15)
      if start>1:ax.axvline(start-.5,color='k',alpha=.06,lw=.4)
     ax.set(title=f'{c} · scale {k}',xlabel='Target repetition (pre-update)',ylabel='Actual practiced CE (nats)',xlim=(1,stop),ylim=(0,top));ax.legend(fontsize=8)
   fig.suptitle(f'2 · {arm}{dose}: original A10/B10 acquisition; no smoothing across switches',fontsize=11);fig.tight_layout(rect=(0,0,1,.95));save(fig,dest,f'figure_2_{arm}_{stop}')
 # Figures3/4: matched330 target state across every registered dose, own parent at0.
 fresh=out[out.phase.eq('Fresh')&out['update'].eq(330)]
 reptile=pd.concat([out[out.phase.eq('R')&out['update'].eq(330)]]+[fresh.assign(condition=c) for c in ['U','P']],ignore_index=True)
 assert set(reptile.dose)==set(REPTILE_RETENTION_DOSES)
 line_panels(reptile,METRIC_FAMILIES[3],'dose','3 · Fresh → Reptile: functional splitting at target330',dest,'figure_3_Reptile',xlabel='Reptile episodes (0 = shared Fresh)')
 for arm in ARMS:
  g=with_parent(out,arm)
  line_panels(g,METRIC_FAMILIES[3],'dose',f'3 · {arm}: functional splitting retention at target330',dest,'figure_3_splitting_'+arm)
  line_panels(g,METRIC_FAMILIES[4],'dose',f'4 · {arm}: generalization at target330; Mixed cues remain distinct',dest,'figure_4_transfer_'+arm)
  # Figure5: all components retained; lines connect each seed, not a weighted score.
  ys=['early_cost','D_A','D_B','A_to_B','B_to_A','Mixed_both_success'];fig,axes=plt.subplots(len(ys),3,figsize=(14,16),squeeze=False)
  for i,y in enumerate(ys):
   for j,k in enumerate([4,5,6]):
    ax=axes[i,j]
    for c in ['U','P']:
     z=g[g.scale.eq(k)&g.condition.eq(c)].sort_values('dose')
     for s,ss in z.groupby('seed'):ax.plot(ss.gap,ss[y],color=COLORS[c],ls=STYLES[c],alpha=.3,lw=.7,marker='.',ms=3)
     failed=familiar_failures(z);ax.scatter(failed.gap,failed[y],marker='x',color='k',s=12,lw=.6,alpha=.5,zorder=6)
     means=z.groupby('dose')[['gap',y]].mean();ax.plot(means.gap,means[y],color=COLORS[c],ls=STYLES[c],lw=1.8,label=c)
     if len(means):ax.scatter(means.gap.iloc[0],means[y].iloc[0],marker='s',s=35,color=COLORS[c]);ax.scatter(means.gap.iloc[-1],means[y].iloc[-1],marker='*',s=75,color=COLORS[c])
    ax.set_xscale('symlog',linthresh=.1);ax.axhline(0,color='k',lw=.4);ax.axvline(0,color='k',lw=.4);ax.set(title=f'scale {k} · {LABELS[y]}',xlabel='C gap (nats; symlog, linear ±.1)',ylabel='Success' if y=='Mixed_both_success' else 'Signed nats')
    if i==0 and j==0:ax.scatter([],[],marker='x',color='k',s=15,label='A/B criterion not met');ax.legend()
  fig.suptitle(f'5 · {arm}: joint components, target330\nSquare = parent; star = FOMAML10000; × = familiar A/B criterion not met; all six seeds retained',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));save(fig,dest,'figure_5_joint_'+arm)
 # Figure6: effective centered four-class contrasts at every one of seven positions.
 eff=read('effective_context_by_position');gg=geom.copy();gg['context_AB_RMS']=np.sqrt((gg.context_A**2+gg.context_B**2)/2);eff=eff.merge(gg[['seed','scale','condition','phase','dose','context_AB_RMS']],on=['seed','scale','condition','phase','dose'],validate='many_to_one')
 for arm in ARMS:
  fig,axes=plt.subplots(7,3,figsize=(14,18),squeeze=False)
  for t in range(1,8):
   for j,k in enumerate([4,5,6]):
    ax=axes[t-1,j]
    for c in ['U','P']:
     z=eff[eff.position.eq(t)&eff.scale.eq(k)&eff.condition.eq(c)&eff.phase.isin(['R',arm])].copy();z['stage']=np.where(z.phase.eq('R'),z.dose/1920,z.dose/10000+1);z=z.sort_values('stage')
     for _,ss in z.groupby('seed'):ax.plot(ss.context_AB_RMS,ss.centered_contrast,color=COLORS[c],ls=STYLES[c],alpha=.25,lw=.6,marker='.',ms=2)
     means=z.groupby('stage')[['context_AB_RMS','centered_contrast']].mean();ax.plot(means.context_AB_RMS,means.centered_contrast,color=COLORS[c],ls=STYLES[c],lw=1.8,label=c);ax.scatter(means.context_AB_RMS.iloc[2],means.centered_contrast.iloc[2],marker='s',s=30,color=COLORS[c]);ax.scatter(means.context_AB_RMS.iloc[-1],means.centered_contrast.iloc[-1],marker='*',s=60,color=COLORS[c])
    fresh=eff[eff.position.eq(t)&eff.scale.eq(k)&eff.phase.eq('Fresh')];ax.scatter(fresh.context_AB_RMS,fresh.centered_contrast,color='k',marker='x',s=20,label='Fresh' if t==1 and j==0 else None);ax.set(title=f'scale {k} · position {t}',xlabel='A/B input-column RMS: √[(rA²+rB²)/2]',ylabel='‖Π Kₜ(wA−wB)‖')
    if t==1 and j==0:ax.legend(fontsize=8)
  fig.suptitle(f'6 · {arm}: raw scale versus effective contextual computation\nMeta initializer BEFORE target practice: crosses = fresh; squares = Reptile1920; stars = FOMAML10000.',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));save(fig,dest,'figure_6_effective_'+arm)
 # Figure7: twelve strictly unseen tasks averaged within each trained seed.
 for arm in ARMS:
  g=pd.concat([held[held.phase.eq('R')].assign(dose=0),held[held.phase.eq(arm)]],ignore_index=True)
  line_panels(g,['conflict_accuracy','critical_ce','conflict_context_benefit'],'dose',f'7 · {arm}: independent unseen conflict adaptation (12 tasks averaged within seed)',dest,'figure_7_conflicts_'+arm)
 files=sorted(p.name for p in dest.glob('figure_*') if p.suffix in ['.png','.svg']);assert len(files)==2*EXPECTED_PAIRS
 for n in files:
  with (dest/n).open('rb') as f:manifest.append(dict(file=n,sha256=hashlib.sha256(f.read()).hexdigest()))
 index=['# Seven registered scientific figure families','', 'Every objective, nominal scale and constraint arm is retained. PNG and SVG share the same numerical tables. No model is evaluated or trained by this renderer.','']
 for family in FAMILIES:
  index.extend([f'## Family {family}',''])
  for n in files:
   if n.startswith(f'figure_{family}_') and n.endswith('.png'):
    base=n[:-4];index.append(f'- {base}: [PNG]({n}) · [SVG]({base}.svg)')
  index.append('')
 (dest/'README.md').write_text('\n'.join(index)+'\n')
 (dest/'FIGURE_INDEX.json').write_text(json.dumps(dict(families=FAMILIES,files=manifest,sources='Saved tables only; no model imports, evaluations or optimization',families_count=7,PNG_SVG_pairs=EXPECTED_PAIRS),indent=2)+'\n')
 if dest.resolve()==(source/'figures').resolve():(source/'audit/figures_complete.json').write_text(json.dumps(dict(passed=True,pairs=EXPECTED_PAIRS,table_only=True),indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=OUT);p.add_argument('--output-dir',type=Path);a=p.parse_args();main(a.source,a.output_dir)
