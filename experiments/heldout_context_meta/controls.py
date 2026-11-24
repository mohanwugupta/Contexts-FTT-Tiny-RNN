"""Frozen C-drive ablations/scales/random probes; reuse natural/zero branches."""
from .common import *
from .data import registry,directory
from .analysis import summarize
from experiments.nonzero_context_generalization.core import inject,operators,baseline64,validate_numeric,assess,fingerprint

def drives(c,directions,include_random):
 vectors=[.5*np.asarray(c,float),1.5*np.asarray(c,float)];names=['C-half','C-onehalf']
 if include_random:
  vectors.extend(np.asarray(directions)*np.linalg.norm(c));names.extend([f'Random-{i:02d}' for i in range(len(directions))])
 return np.stack(vectors),names

def main():
 r=registry();r=r[r.dose.isin([0,960])];metrics=[];positions=[];controls=[];seed=[]
 for row in r.itertuples():
  src=directory(row);bundle=torch.load(ROOT/row.path,weights_only=False);ev=pd.read_csv(src/'evaluations.csv',float_precision='round_trip');pos=pd.read_csv(src/'position_predictions.csv.gz',float_precision='round_trip');dirs=np.load(OUT/f'registry/random_directions_seed_{row.seed}.npy');ids={k:getattr(row,k) for k in ['seed','condition','arm','dose']}
  for t in [0,170,330]:
   state=bundle['states'][t]
   for name,cue in [('Natural-C','C'),('NoContext','NoContext'),('C-removed','NoContext')]:
    x=ev[(ev['update']==t)&(ev.cue==cue)].iloc[0];metrics.append(dict(**ids,update=t,control=name,ce=x.ce,accuracy=x.accuracy,success=bool(x.ce<=.5 and x.accuracy>=6/7) if np.isfinite(x.ce) else np.nan,provenance='exact natural branch reuse; removed=NoContext identity' if name=='C-removed' else 'exact natural branch reuse'))
    for q in pos[(pos['update']==t)&(pos.cue==cue)].to_dict('records'):positions.append(dict(**q,**ids,control=name,provenance='archived natural/zero branch'))
   if state['status']!='completed':continue
   d=state['model'];before=fingerprint(d);c=d['recurrent.weight_ih_l0'][:,6].double().numpy();v,names=drives(c,dirs,include_random=not row.reused);z,h=inject(d,v);s,k=operators(d);z0,h0=baseline64(d);co=validate_numeric(d,v,z,h,s,k,z0,h0);a=assess(z)
   for q in co:controls.append(dict(**ids,update=t,**q))
   for i,name in enumerate(names):
    ce=a['ce'][i].mean();acc=a['accuracy'][i].mean();metrics.append(dict(**ids,update=t,control=name,ce=ce,accuracy=acc,success=bool(ce<=.5 and acc>=6/7),provenance='new fixed frozen drive'))
    for j in range(7):positions.append(dict(**ids,update=t,control=name,position=j+1,target=int(core.TARGET[j]),ce=a['ce'][i,j],correct=bool(a['accuracy'][i,j]),margin=a['margin'][i,j],**{f'logit_{zz}':z[i,j,zz] for zz in range(4)},**{f'probability_{zz}':a['probabilities'][i,j,zz] for zz in range(4)},**{f'hidden_{zz}':h[i,j,zz] for zz in range(8)}))
   assert fingerprint(d)==before
  if row.reused:
   prefix=f's{row.seed:02d}_{row.prior}_{row.organization}_d1920';receipt=json.loads((SOURCE/f'random_inputs/{prefix}.json').read_text());assert receipt['source_sha']==row.sha256
   old=pd.read_csv(SOURCE/f'random_inputs/{prefix}_metrics.csv',float_precision='round_trip');oldp=pd.read_csv(SOURCE/f'random_inputs/{prefix}_positions.csv',float_precision='round_trip');oldv=pd.read_csv(SOURCE/f'random_inputs/{prefix}_vectors.csv',float_precision='round_trip')
   for tt,g in oldv.groupby('update'):
    vv=g.sort_values('vector')[[f'drive_{j}' for j in range(8)]].to_numpy();norm=old[old['update']==tt].C_norm.iloc[0];np.testing.assert_allclose(vv,dirs*norm,rtol=1e-14,atol=1e-14)
   for q in old.to_dict('records'):metrics.append(dict(**ids,update=q['update'],control=f'Random-{q["vector"]:02d}',ce=q['ce'],accuracy=q['accuracy'],success=q['success'],provenance='exact scaling1920 normmatched random cache'))
   for q in oldp.to_dict('records'):q['control']=f'Random-{q.pop("vector"):02d}';positions.append(dict(**q,**ids,provenance='exact random cache reuse'))
 f=pd.DataFrame(metrics);p=pd.DataFrame(positions);f['control_family']=np.where(f.control.str.startswith('Random-'),'Random',f.control)
 for keys,g in f.groupby(['seed','condition','arm','dose','update','control_family']):seed.append(dict(zip(['seed','condition','arm','dose','update','control'],keys),ce=g.ce.mean(),accuracy=g.accuracy.mean(),success_fraction=g.success.astype(float).mean(),n_directions=len(g)))
 sd=pd.DataFrame(seed)
 for name,ff in [('control_metrics',f),('control_positions',p),('control_numerical',pd.DataFrame(controls)),('control_seed',sd),('control_summary',summarize(sd,['condition','arm','dose','update','control'],['ce','accuracy','success_fraction']))]:write_frame(OUT/'tables'/f'{name}.csv',ff)
 dump(OUT/'audit/controls_complete.json',dict(passed=True,models=len(r),states=[0,170,330],natural_reused=True,removed_equals_NoContext=True,historical_random_cache_reused=True,new_frozen_drives=11040,optimizer_updates=0));append('## Frozencontrolscomplete\nNaturalC/NoContextreused,Cremoved=NoContextalgebraaliasnotindependentrescue. Half/1.5Cfixeddrives+20normmatchedsealedrandomdirectionsat0/+960,states0/170/330. Historicalrandomcacheexactsource/vectorcorrespondenceverifiedandreused. Allweightsunchanged,numericalboundsfixed;randomdirectionsaveragedwithinseed,n20. Nooptimization/selection.')
if __name__=='__main__':main()
