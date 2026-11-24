"""Six registered figure families, from cached numerical tables only."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from functools import lru_cache
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'ab_update_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .statistics import indices
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/ab_update_transfer_analysis/v1'
COL={'A':'#1f77b4','B':'#ff7f0e','C':'#2ca02c','NoContext':'#666666','Mixed-average':'#9467bd','U':'#3776b5','P':'#c84045'}
ARMS=['T','B','O1','O2'];PROBES=['A','B','C','NoContext'];PRIMARY=[0,10,20,40,60]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'ab-update-transfer-v1','axes.spines.top':False,'axes.spines.right':False})

def read(n):
 for suffix in ['.csv','.csv.gz']:
  p=OUT/'tables'/(n+suffix)
  if p.exists():return pd.read_csv(p,float_precision='round_trip')
 raise FileNotFoundError(n)
@lru_cache(8192)
def bands(shape,data):
 a=np.frombuffer(data,dtype=np.float64).reshape(shape);valid=np.isfinite(a);idx=indices();w=np.stack([(idx==i).sum(1)/6 for i in range(6)],1);den=w@valid;draw=np.divide(w@np.nan_to_num(a),den,out=np.full((10000,a.shape[1]),np.nan),where=den>0)
 with np.errstate(invalid='ignore'):
  mean=np.divide(np.nansum(a,axis=0),valid.sum(0),out=np.full(a.shape[1],np.nan),where=valid.sum(0)>0)
  lo=np.nanquantile(draw,.025,axis=0);hi=np.nanquantile(draw,.975,axis=0)
 return mean,lo,hi

def band(ax,g,x,y,color,label=None,seed_lines=True):
 if not len(g):return
 p=g.pivot(index='seed',columns=x,values=y).reindex(range(6)).sort_index(axis=1);v=np.asarray(p,float);mean,lo,hi=bands(v.shape,v.tobytes());xx=np.asarray(p.columns,float)
 if seed_lines:
  for row in v:ax.plot(xx,row,color=color,alpha=.13,lw=.6)
 ax.fill_between(xx,lo,hi,color=color,alpha=.12,lw=0);ax.plot(xx,mean,color=color,lw=1.6,label=label,marker='.' if len(xx)<12 else None,ms=3)

def save(fig,dest,name,title):
 fig.suptitle(title+'\nSix paired seeds; signed measurements; pointwise95% whole-seed intervals',fontsize=12);fig.tight_layout(rect=[0,.025,1,.94]);fig.savefig(dest/(name+'.png'),dpi=140,metadata={'Software':'AB update analysis'});fig.savefig(dest/(name+'.svg'),metadata={'Date':None,'Creator':'AB update analysis'});plt.close(fig)

def grid6(rows=1,size=None):
 fig,axes=plt.subplots(rows,6,figsize=size or (22,max(3.5,rows*3.0)),squeeze=False)
 for col,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
  for row in range(rows):axes[row,col].set_title(f'{reg} · scale{scale}')
 return fig,axes

def actual_curves(dest,curve):
 q=curve[curve.cohort.eq('final')]
 for arm in ARMS:
  z=q[q.objective.eq(arm)]
  for stop in [60,990]:
   fig,ax=grid6()
   for j,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    a=ax[0,j];g=z[z.regime.eq(reg)&z.scale.eq(scale)&z.target_update.le(stop)]
    for c in PROBES:band(a,g[g.probe_context.eq(c)],'target_update','ce',COL[c],c)
    for n in range(0,stop,10):
     source='A' if (n//10)%2==0 else 'B';a.axvspan(n,min(n+10,stop),color=COL[source],alpha=.025,lw=0);a.axvline(n,color='.88',lw=.35)
     if stop==60:a.text(n+5,.975,source,transform=a.get_xaxis_transform(),ha='center',va='top',fontsize=6,color=COL[source])
    a.set_yscale('symlog',linthresh=.05);a.set_ylim(bottom=0);a.set_xlabel('Completed target updates');a.set_ylabel('CE (nats; symlog)');a.axhline(.05,color='.6',ls=':',lw=.7)
   ax[0,0].legend(fontsize=7);save(fig,dest,f'figure_1_{arm}_{stop}',f'1 · {arm}: actual A10/B10 trajectory; A blue/B orange; first block A; dotted CE floor .05')
  for metric,cues in [('margin',PROBES),('ce',['Mixed-average'])]:
   fig,ax=grid6()
   for j,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    a=ax[0,j];g=z[z.regime.eq(reg)&z.scale.eq(scale)]
    for c in cues:band(a,g[g.probe_context.eq(c)],'target_update',metric,COL[c],c)
    a.set_xlabel('Completed target updates');a.set_ylabel(metric+' (native seven-position mean)')
   ax[0,0].legend(fontsize=7);save(fig,dest,f'figure_1_{arm}_{metric}_companion',f'1 companion · {arm}: '+metric+'; Mixed-average is the interpolation control' if metric=='ce' else f'1 companion · {arm}: original prediction margins')

def gain_matrices(dest,effects):
 q=effects[effects.cohort.eq('final')&effects.branch_type.eq('native')&effects.target_update.isin(PRIMARY)&effects.probe_context.isin(PROBES)]
 for h in [1,10]:
  zz=q[q.branch_step.eq(h)];vmax=max(float(abs(zz.gain).max()),1e-12)
  for arm in ARMS:
   z=zz[zz.objective.eq(arm)];fig,axes=plt.subplots(6,5,figsize=(18,17),squeeze=False)
   for i,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    for j,n in enumerate(PRIMARY):
     a=axes[i,j];g=z[z.regime.eq(reg)&z.scale.eq(scale)&z.target_update.eq(n)];p=g.groupby(['source_context','probe_context']).gain.mean().unstack().reindex(index=['A','B'],columns=PROBES);im=a.imshow(p.to_numpy(),cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
     for r in range(2):
      for c in range(4):a.text(c,r,f'{p.iloc[r,c]:.2g}',ha='center',va='center',fontsize=8)
     a.set_xticks(range(4),['A','B','C','N']);a.set_yticks([0,1],['A','B']);a.set_ylabel('Update source' if j==0 else '');a.set_title(f'{reg} s{scale} · target{n}')
   fig.subplots_adjust(left=.065,right=.90,top=.90,bottom=.055,wspace=.38,hspace=.55);cax=fig.add_axes([.93,.25,.012,.5]);fig.colorbar(im,cax=cax,label='Gain (nats; positive improves)');fig.suptitle(f'2 · {arm}: {h}-step matched gain matrix\nColumns A/B/C/N (N=NoContext); common symmetric scale across objectives/seeds/checkpoints for this horizon',fontsize=13);fig.savefig(dest/f'figure_2_{arm}_h{h}.png',dpi=140);fig.savefig(dest/f'figure_2_{arm}_h{h}.svg',metadata={'Date':None});plt.close(fig)
   fig,axes=plt.subplots(6,6,figsize=(24,17),squeeze=False)
   cols=pd.MultiIndex.from_product([PRIMARY,PROBES])
   for i,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    for seed in range(6):
     a=axes[i,seed];g=z[z.regime.eq(reg)&z.scale.eq(scale)&z.seed.eq(seed)];p=g.pivot(index='source_context',columns=['target_update','probe_context'],values='gain').reindex(index=['A','B'],columns=cols);a.imshow(p.to_numpy(),cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto');a.set_xticks([1.5+4*j for j in range(5)],PRIMARY);a.set_yticks([0,1],['A','B']);a.set_title(f'{reg} s{scale} · seed{seed}')
   save(fig,dest,f'figure_2_{arm}_h{h}_seeds',f'2 individual seeds · {arm}, horizon{h}; within each checkpoint columns A,B,C,NoContext; common scale±{vmax:.3g}nats')

def historical_stages(dest,effects):
 q=effects[effects.branch_type.eq('native')&effects.branch_step.eq(1)]
 for arm in ARMS:
  for source,n in [('A',0),('B',10)]:
   fig,axes=grid6(rows=3)
   for j,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    g=q[q.scale.eq(scale)&q.target_update.eq(n)&q.source_context.eq(source)&((q.meta_stage.eq('fresh'))|(q.regime.eq(reg)&q.meta_stage.eq('reptile'))|(q.regime.eq(reg)&q.objective.eq(arm)))].copy();g['stage_index']=np.where(g.meta_stage.eq('fresh'),0,np.where(g.meta_stage.eq('reptile'),1,np.where(g.meta_episodes.eq(1920),2,3)))
    for i,(metric,probe) in enumerate([('specificity','C'),('gain','C'),('gain','NoContext')]):
     a=axes[i,j];z=g[g.probe_context.eq(probe)];band(a,z,'stage_index',metric,COL[reg],reg);a.set_xticks(range(4),['Fresh','R1920','F1920','F10000'],rotation=25);a.axhline(0,color='.6',lw=.6);a.set_title(f'{reg} s{scale} · '+('D' if i==0 else probe));a.set_ylabel(('Specificity' if i==0 else 'Probe gain')+' (nats)')
   save(fig,dest,f'figure_2_{arm}_{source}_historical',f'2 stage companion · {arm}, {source} one-step from target{n}; Fresh/R anchors shared across objectives, not extra seeds')

def joint(dest,effects):
 q=effects[effects.cohort.eq('final')&effects.branch_type.eq('native')&effects.target_update.isin(PRIMARY)]
 for arm in ARMS:
  for h in [1,10]:
   fig,axes=plt.subplots(4,3,figsize=(15,15),squeeze=False)
   for i,(source,probe) in enumerate((s,p) for s in ['A','B'] for p in ['C','NoContext']):
    for j,scale in enumerate([4,5,6]):
     a=axes[i,j];g=q[q.objective.eq(arm)&q.scale.eq(scale)&q.source_context.eq(source)&q.probe_context.eq(probe)&q.branch_step.eq(h)]
     for reg in ['U','P']:
      for seed in range(6):
       x=g[g.regime.eq(reg)&g.seed.eq(seed)].sort_values('target_update');a.plot(x.specificity,x.gain,color=COL[reg],alpha=.25,lw=.7);a.scatter(x.specificity,x.gain,color=COL[reg],s=12,alpha=.6)
       bad=x[x.own_gain.le(0)];a.scatter(bad.specificity,bad.gain,marker='x',c='k',s=32)
       bad=x[x.cross_gain.lt(0)];a.scatter(bad.specificity,bad.gain,marker='v',facecolors='none',edgecolors='k',s=35)
     a.axhline(0,color='.5',lw=.7);a.axvline(0,color='.5',lw=.7);a.set_title(f's{scale}: {source} updates → {probe}');a.set_xlabel('Own − familiar cross gain D (nats)');a.set_ylabel('Probe gain (nats)')
   save(fig,dest,f'figure_3_{arm}_h{h}',f'3 · {arm}: specificity and transfer, horizon{h}; blue U/red P; × own gain≤0; ▽ cross deterioration; lines connect0/10/20/40/60')

def starting_error(dest,curve,branch):
 q=curve[curve.cohort.eq('final')&curve.probe_context.isin(['C','NoContext'])]
 for arm in ARMS:
  fig,axes=plt.subplots(4,3,figsize=(15,14),squeeze=False)
  for i,(reg,c) in enumerate((r,c) for r in ['U','P'] for c in ['C','NoContext']):
   for j,scale in enumerate([4,5,6]):
    a=axes[i,j];g=q[q.objective.eq(arm)&q.scale.eq(scale)&q.regime.eq(reg)&q.probe_context.eq(c)].sort_values(['seed','target_update']).copy();parts=[]
    for seed,x in g.groupby('seed'):
     x=x.sort_values('target_update').copy();d=x.ce.shift()-x.ce;x['starting_error']=x.ce.iloc[0];x['gain_A']=d.where(x.practiced_cue_previous.eq('A'),0).fillna(0).cumsum();x['gain_B']=d.where(x.practiced_cue_previous.eq('B'),0).fillna(0).cumsum();np.testing.assert_allclose(x.starting_error-x.gain_A-x.gain_B,x.ce,atol=1e-10,rtol=1e-12);parts.append(x)
    g=pd.concat(parts)
    for m,col in [('starting_error','.45'),('ce',COL[c]),('gain_A',COL['A']),('gain_B',COL['B'])]:band(a,g,'target_update',m,col,m)
    a.axhline(0,color='.6',lw=.5);a.set_title(f'{reg} s{scale} · {c}');a.set_xlabel('Completed target updates');a.set_ylabel('Error / cumulative gain (nats)')
  axes[0,0].legend(fontsize=7);save(fig,dest,f'figure_4_{arm}_realized',f'4 · {arm}: starting error − realized A gain − realized B gain = remaining error; all990 updates')
  for source in ['A','B']:
   fig,axes=grid6(rows=2)
   for j,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    for i,c in enumerate(['C','NoContext']):
     a=axes[i,j];g=branch[branch.cohort.eq('final')&branch.objective.eq(arm)&branch.regime.eq(reg)&branch.scale.eq(scale)&branch.source_context.eq(source)&branch.branch_type.eq('native')&branch.target_update.isin(PRIMARY)&branch.probe_context.eq(c)]
     for n,color in zip(PRIMARY,plt.cm.viridis(np.linspace(0,1,5))):band(a,g[g.target_update.eq(n)],'branch_step','gain',color,f'target{n}')
     a.axhline(0,color='.5',lw=.6);a.set_title(f'{reg} s{scale} · {c}');a.set_xlabel('Constant-source branch step');a.set_ylabel('Cumulative gain (nats)')
   axes[0,0].legend(fontsize=7);save(fig,dest,f'figure_4_{arm}_{source}_branches',f'4 companion · {arm}: {source}-only10-step counterfactual branches from fixed checkpoints')

def parameter_effects(dest,params):
 variants=['full','shared_only','cue_only','practiced_cue_only','other_cue_only','interaction'];colors=['k','#1b9e77','#d95f02','#7570b3','#e7298a','#a6761d'];q=params[params.cohort.eq('final')]
 for arm in ARMS:
  for source in ['A','B']:
   fig,axes=grid6(rows=4)
   for j,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
    for i,c in enumerate(PROBES):
     a=axes[i,j];g=q[q.objective.eq(arm)&q.source_context.eq(source)&q.regime.eq(reg)&q.scale.eq(scale)&q.probe_context.eq(c)]
     for v,col in zip(variants,colors):band(a,g[g.variant.eq(v)],'target_update','gain',col,v,seed_lines=False)
     a.axhline(0,color='.6',lw=.6);a.set_title(f'{reg} s{scale} · {c}');a.set_xlabel('Checkpoint before one update');a.set_ylabel('Endpoint replacement gain (nats)')
     if c in ['C','NoContext']:a.text(.02,.02,'Cue-only zero is structural',transform=a.transAxes,fontsize=7)
   axes[0,0].legend(fontsize=6);save(fig,dest,f'figure_5_{arm}_{source}',f'5 · {arm}: cached {source} update parameter replacements; interaction=full−shared−cue; no extra optimizer step')
  g=q[q.objective.eq(arm)&q.variant.isin(['baseline','full'])&q.probe_context.isin(['C','NoContext'])];keys=['seed','scale','regime','source_context','target_update','variant'];p=g.pivot(index=keys,columns='probe_context',values='post_ce').reset_index();p['C_minus_N']=p.C-p.NoContext
  fig,axes=grid6(rows=2)
  for j,(reg,scale) in enumerate((r,s) for r in ['U','P'] for s in [4,5,6]):
   for i,source in enumerate(['A','B']):
    a=axes[i,j];z=p[p.regime.eq(reg)&p.scale.eq(scale)&p.source_context.eq(source)]
    for v,col in [('baseline','.45'),('full',COL['C'])]:band(a,z[z.variant.eq(v)],'target_update','C_minus_N',col,v)
    a.axhline(0,color='.6',lw=.6);a.set_title(f'{reg} s{scale} · {source} update');a.set_xlabel('Checkpoint');a.set_ylabel('C CE − NoContext CE (nats)')
  axes[0,0].legend();save(fig,dest,f'figure_5_{arm}_C_cue',f'5 C-cue companion · {arm}: negative C−NoContext favors learned C cue; position operators in exact table')

def adam_history(dest,adam):
 for arm in ARMS:
  for reg in ['U','P']:
   fig,axes=plt.subplots(4,6,figsize=(23,13),squeeze=False)
   for i,c in enumerate(PROBES):
    for j,scale in enumerate([4,5,6]):
     z=adam[adam.objective.eq(arm)&adam.regime.eq(reg)&adam.scale.eq(scale)&adam.probe_context.eq(c)].copy();z['x']=(z.target_update.eq(20).astype(int)*2+z.source_context.eq('B').astype(int))
     for offset,(kind,col) in enumerate([('held','.5'),('native',COL['A']),('zero',COL['B'])]):
      for mode,colindex in [('gain',j),('ce',j+3)]:
       a=axes[i,colindex];g=z.copy();g['value']=g['held_gain' if mode=='gain' else 'held_ce'] if kind=='held' else g[f'gain_{kind}' if mode=='gain' else f'post_ce_{kind}'];g['x']=g.x+(offset-1)*.1;band(a,g,'x','value',col,kind)
       a.set_xticks(range(4),['10A','10B','20A','20B']);a.set_title(f's{scale} · {c} · '+mode);a.set_xlabel('Checkpoint/source');a.set_ylabel(('Gain' if mode=='gain' else 'Endpoint CE')+' (nats)')
     axes[i,j].axhline(0,color='.5',lw=.6)
   axes[0,0].legend(fontsize=7);save(fig,dest,f'figure_6_{arm}_{reg}',f'6 · {arm} {reg}: held vs native vs zero-gradient Adam; existing moments retained; held counter n, stepped n+1')

def main(output_dir=None):
 dest=Path(output_dir) if output_dir else OUT/'figures';dest.mkdir(parents=True,exist_ok=True);assert not list(dest.iterdir()),'Use an empty figure directory'
 curve=read('native_curve_metrics');e=read('update_effects');b=read('branch_metrics');p=read('parameter_interventions');a=read('adam_controls')
 for fn,args in [(actual_curves,(curve,)),(gain_matrices,(e,)),(historical_stages,(e,)),(joint,(e,)),(starting_error,(curve,b)),(parameter_effects,(p,)),(adam_history,(a,))]:fn(dest,*args);print('Figure family completed',fn.__name__,flush=True)
 files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.suffix in ['.png','.svg']};sources={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for n in ['native_curve_metrics','update_effects','branch_metrics','parameter_interventions','adam_controls'] for suffix in ['.csv','.csv.gz'] if (p:=OUT/'tables'/(n+suffix)).exists()};(dest/'FIGURE_INDEX.json').write_text(json.dumps(dict(files=files,sources=sources,table_only=True,bootstrap_seed=20261122,paired_seeds=6,heatmap_scale='symmetric full individual-seed range within each horizon'),indent=2)+'\n');(dest/'README.md').write_text('# Figure index\n\nAll objective/scale/regime conditions are retained. A/B sources are distinct from objectiveB. Gains are signed and bootstrap whole paired seeds.\n\n'+'\n'.join(f'- [{p.stem}]({p.name}) ([SVG]({p.with_suffix(".svg").name}))' for p in sorted(dest.glob('*.png')))+'\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output-dir');main(parser.parse_args().output_dir)
