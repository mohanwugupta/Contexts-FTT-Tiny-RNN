"""Fixed norm-matched unfamiliar drives; read-only, never training or selection."""
import argparse
from .common import *
from .analysis import interval
from .data import registry
from experiments.nonzero_context_generalization.core import inject,column_rollout,operators,baseline64,assess,validate_numeric,fingerprint

def norm_matched(directions,norm):
 directions=np.asarray(directions,float);np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,rtol=1e-13,atol=1e-13);v=directions*float(norm);np.testing.assert_allclose(np.linalg.norm(v,axis=1),norm,rtol=1e-13,atol=1e-13);return v

def probe(row):
 dest=OUT/'random_inputs'/row['run_id'];dest.parent.mkdir(exist_ok=True)
 if dest.with_suffix('.json').exists():return json.loads(dest.with_suffix('.json').read_text())
 b=torch.load(ROOT/row['path'],weights_only=False);vectors=np.load(OUT/f'registry/random_directions_seed_{row["seed"]}.npy');rows=[];positions=[];controls=[];vh=[]
 for k in [0,170,330]:
  st=b['states'][k]
  if st['status']!='completed':
   for i in range(20):rows.append(dict(update=k,vector=i,status='blocked_'+st['status'],ce=np.nan,accuracy=np.nan,success=np.nan,C_norm=np.nan))
   continue
  d=st['model'];before=fingerprint(d);norm=float(d['recurrent.weight_ih_l0'][:,6].double().norm());v=norm_matched(vectors,norm);z,h=inject(d,v);zd,hd=inject(d,v,dtype=torch.float64);s,kk=operators(d);z0,h0=baseline64(d);co=validate_numeric(d,v,z,h,s,kk,z0,h0);tol=1e-10+1e-12*max(abs(zd).max(),abs(hd).max());assert np.all(abs(zd-(z0[None]+np.einsum('tcj,nj->ntc',kk,v)))<=tol);assert np.all(abs(hd-(h0[None]+np.einsum('tij,nj->nti',s,v)))<=tol);zc,hc=column_rollout(d,v);coc=validate_numeric(d,v,zc,hc,s,kk,z0,h0);a=assess(z)
  for x in co:controls.append(dict(update=k,method='injection',**x))
  for x in coc:controls.append(dict(update=k,method='hypothetical_column',**x))
  for i in range(20):
   ce=float(a['ce'][i].mean());acc=float(a['accuracy'][i].mean());rows.append(dict(update=k,vector=i,status='completed',ce=ce,accuracy=acc,success=bool(ce<=.5 and acc>=6/7),C_norm=norm));vh.append(dict(update=k,vector=i,**{f'drive_{j}':v[i,j] for j in range(8)}))
   for t in range(7):positions.append(dict(update=k,vector=i,position=t+1,target=int(old.TARGET[t]),ce=a['ce'][i,t],correct=bool(a['accuracy'][i,t]),margin=a['margin'][i,t],**{f'logit_{j}':z[i,t,j] for j in range(4)},**{f'probability_{j}':a['probabilities'][i,t,j] for j in range(4)},**{f'hidden_{j}':h[i,t,j] for j in range(8)}))
  assert fingerprint(d)==before
 for suffix,values in [('metrics',rows),('positions',positions),('controls',controls),('vectors',vh)]:write_frame(dest.with_name(dest.name+'_'+suffix+'.csv'),values)
 result=dict(run_id=row['run_id'],seed=int(row['seed']),condition=row['condition'],dose=int(row['dose']),source_sha=row['sha256'],metrics=60,states=3,forward_weights_unchanged=True);dump(dest.with_suffix('.json'),result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--available',action='store_true');args=p.parse_args();r=pd.read_csv(OUT/'registry/reuse_registry.csv') if args.available else registry('full');r=r[r.dose.isin([0,240,480,960,1920])];records=[]
 for row in r.to_dict('records'):records.append(probe(row))
 if args.available:print('Existing120random-probe histories complete',flush=True);return
 metrics=[];seed=[]
 for row in records:
  f=pd.read_csv(OUT/f'random_inputs/{row["run_id"]}_metrics.csv',float_precision='round_trip');conds=[f'P1-{row["condition"][-1]}',f'P2-{row["condition"][-1]}'] if row['dose']==0 else [row['condition']]
  for c in conds:
   z=f.copy();z['condition']=c;z['seed']=row['seed'];z['dose']=row['dose'];z['physical_run_id']=row['run_id'];metrics.append(z)
   for k,g in z.groupby('update'):seed.append(dict(seed=row['seed'],condition=c,dose=row['dose'],update=k,random_ce=g.ce.mean(),random_accuracy=g.accuracy.mean(),random_success_fraction=g.success.astype(float).mean(),valid_vectors=int(g.ce.notna().sum())))
 f=pd.concat(metrics,ignore_index=True);sd=pd.DataFrame(seed);summary=[]
 for keys,g in sd.groupby(['condition','dose','update']):
  for metric in ['random_ce','random_accuracy','random_success_fraction']:summary.append(dict(zip(['condition','dose','update'],keys),metric=metric,**interval(g.set_index('seed')[metric].reindex(range(20)))))
 for name,frame in [('random_metrics',f),('random_robustness_seed',sd),('random_robustness_summary',pd.DataFrame(summary))]:write_frame(OUT/'tables'/f'{name}.csv',frame)
 dump(OUT/'audit/random_inputs_complete.json',dict(passed=True,physical_runs=len(r),physical_cells=len(r)*60,logical_cells=len(f),seeds=20,vectors_per_seed=20,all_weights_frozen=True,selection=False));append('## Norm-matched randominputdiagnosticcomplete\nSealed20directionsperseed,normmatchedtofixedCcolumn,all5doses/3states. Native/independentdouble/hypotheticalcolumnnumericcontrols pass;no weightmutation/training/selection. Directionfractions averagedwithinseedbefore10kCI;naturalCseparate.')
if __name__=='__main__':main()
