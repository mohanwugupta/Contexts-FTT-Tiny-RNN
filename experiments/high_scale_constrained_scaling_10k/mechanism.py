"""Weight-only Linear algebra and complete table exports; no adaptation or probes."""
import concurrent.futures,multiprocessing
from .common import *
from .verify import INDEX,GROUPS,history
from experiments.nonzero_context_generalization.core import operators
from experiments.high_scale_constrained_meta.audit import metrics
FIELDS=['before_outer_rms','raw_rms','projected_rms','floor','multiplier','group_displacement','projected_flag','enabled_flag']
PARAMS=['recurrent.weight_ih_l0','recurrent.weight_hh_l0','recurrent.bias_ih_l0','recurrent.bias_hh_l0','sequence_head.weight','sequence_head.bias']

def center(z):return z-z.mean(-1,keepdims=True)
def flat(state):return np.concatenate([state[n].detach().numpy().ravel() for n in PARAMS])
def cosine(a,b):
 d=np.linalg.norm(a)*np.linalg.norm(b)
 return float(a@b/d) if d else np.nan

def dense_geometry(values):
 values=np.asarray(values,dtype=np.float64);assert values.shape==(172,);wi=values[:56].reshape(8,7);R=values[56:120].reshape(8,8);O=values[136:168].reshape(4,8);a,b,c=wi[:,4],wi[:,5],wi[:,6];h=np.zeros(8);norms=[];reference_scale=0.
 for _ in range(7):h=a-b+R@h;z=O@h;norms.append(float(np.linalg.norm(center(z))));reference_scale=max(reference_scale,float(abs(h).max()),float(abs(z).max()))
 return dict(AB_cosine=cosine(a,b),AB_column_difference=float(np.linalg.norm(a-b)),shared_C_cosine=cosine((a+b)/2,c),centered_contrast_mean=float(np.mean(norms)),position_norms=np.asarray(norms),double_reference_scale=reference_scale)

def geometry(state,fresh):
 _,K=operators(state);wi=state[PARAMS[0]].double().numpy();R=state[PARAMS[1]].double().numpy();wo=state[PARAMS[4]].double().numpy();a,b,c=wi[:,4],wi[:,5],wi[:,6];delta=a-b;response=np.einsum('tcj,j->tc',K,delta);pi=center(response);h=np.zeros(8);independent=[]
 for _ in range(7):h=delta+R@h;independent.append(wo@h)
 error=float(abs(np.asarray(independent)-response).max());tol=1e-10+1e-12*max(1.,abs(response).max());assert error<=tol
 v=flat(state).astype(float);v0=flat(fresh).astype(float);common=(a+b)/2;directions=[];q=dict(**metrics(state),AB_cosine=cosine(a,b),AB_column_difference=float(np.linalg.norm(delta)),shared_C_cosine=cosine(common,c),shared_context_norm=float(np.linalg.norm(common)),context_specific_norm=float(np.linalg.norm(delta/2)),centered_contrast_mean=float(np.linalg.norm(pi,axis=1).mean()),AB_identity_error=error,AB_identity_tolerance=tol)
 for name,idx in INDEX.items():q['cosine_fresh_'+name]=cosine(v[idx],v0[idx]);q['displacement_fresh_'+name]=float(np.linalg.norm(v[idx]-v0[idx]))
 for i,(x,y) in enumerate(zip(v,v0)):directions.append(dict(parameter_index=i,value=x,fresh_value=y,change=x-y))
 positions=[dict(position=t+1,centered_contrast=float(np.linalg.norm(pi[t])),uncentered_contrast=float(np.linalg.norm(response[t])),**{f'centered_class_{j}':pi[t,j] for j in range(4)},**{f'raw_class_{j}':response[t,j] for j in range(4)}) for t in range(7)]
 return q,positions,directions

def legacy_projection(previous,after,f):
 rows=[]
 for g in GROUPS:
  idx=INDEX[g];before=float(np.sqrt(np.mean(previous[idx].astype(float)**2)));r=float(np.sqrt(np.mean(after[idx].astype(float)**2)));rows.append([before,r,r,f[g],1.,float(np.linalg.norm(after[idx].astype(float)-after[idx].astype(float))),0.,0.])
 return np.array(rows,float)

def export_history(job):
 s,k,c,a=job;configure();folder=OUT/'tables/scale_histories';folder.mkdir(exist_ok=True);name=f's{s:02d}_scale{k}_{c}_{a}';path=folder/(name+'.npz');receipt=folder/(name+'.json');identity=dict(seed=s,scale=k,condition=c,phase=a,analysis_sha=sha(Path(__file__)))
 if receipt.exists():
  r=json.loads(receipt.read_text());assert r['analysis_sha']==identity['analysis_sha'] and r['sha256']==sha(path);return r
 f=torch.load(OUT/f'initial/s{s:02d}_scale{k}.pt',weights_only=False)['floors'];arrays=[];scalars=[];dense_positions=[];sources=[];last=0
 journals=([SOURCE/f'meta/s{s:02d}_scale{k}/{a}_journal.sqlite'] if c=='U' else [])+([OUT/f'meta/s{s:02d}_scale{k}_{c}/{a}_journal.sqlite'] if c=='P' or a!='R' else [])
 for src in journals:
  sources.append(dict(path=str(src.relative_to(ROOT)),sha256=sha(src)))
  for r in history(src):
   dose=r['meta_episode'];assert dose>last;last=dose
   if 'projection' in r:pr=np.asarray(r['projection'],float)
   else:pr=legacy_projection(r['before'][0],r['after'][0],f)
   weights=r['projected_parameters'] if 'projected_parameters' in r else r['after'][0];dg=dense_geometry(weights)
   if np.isfinite(r['centered_contrast']):np.testing.assert_allclose(dg['centered_contrast_mean'],r['centered_contrast'],atol=1e-10+1e-12*dg['double_reference_scale'],rtol=0)
   dense_positions.append(dg['position_norms'])
   assert pr.shape==(5,8);arrays.append(pr);scalars.append([dose,r['episode'],r['recurrent_spectral_norm'],r.get('AB_cosine',np.nan),r.get('AB_column_difference',np.nan),r['centered_contrast'],r['episodes'],r['inner_updates'],r['outer_updates'],r['interpolations'],r['sequence_presentations'],r['query_examples'],r['query_gradient_evaluations'],dg['centered_contrast_mean'],dg['AB_cosine'],dg['AB_column_difference'],dg['shared_C_cosine']])
 expected=1920 if a=='R' else 10000;assert last==expected and len(arrays)==expected//(2 if a=='B' else 1);p=np.stack(arrays);v=np.array(scalars,float)
 with path.open('xb') as out:np.savez_compressed(out,projection=p,scalars=v,dense_position_contrast_norms=np.stack(dense_positions))
 # Plotting uses exact every60th outer state plus every registered dose. Every step remains in NPZ.
 mask=(v[:,1].astype(int)%60==0)|np.isin(v[:,0],DOSES+[1,2,1921,1922]);sample=[];summary=[]
 for j,g in enumerate(GROUPS):
  for i in np.flatnonzero(mask):sample.append(dict(identity,group=g,meta_episode=int(v[i,0]),outer_iteration=int(v[i,1]),**dict(zip(FIELDS,p[i,j])),recurrent_spectral_norm=v[i,2],AB_cosine=v[i,3],AB_column_difference=v[i,4],centered_contrast=v[i,5],dense_centered_contrast=v[i,13],dense_AB_cosine=v[i,14],dense_AB_column_difference=v[i,15],dense_shared_C_cosine=v[i,16]))
  summary.append(dict(seed=s,scale=k,condition=c,phase=a,group=g,records=len(p),projection_count=int(p[:,j,6].sum()),projection_frequency=float(p[:,j,6].mean()),floor=f[g],minimum_after_rms=float(p[:,j,2].min()),final_rms=float(p[-1,j,2]),minimum_floor_difference=float((p[:,j,2]-p[:,j,3]).min()),max_multiplier=float(p[:,j,4].max()),sum_group_projection_displacement=float(p[:,j,5].sum())))
 samplepath=folder/(name+'_plot.csv');summarypath=folder/(name+'_summary.csv');write_frame(samplepath,sample);write_frame(summarypath,summary)
 r=dict(identity,path=str(path.relative_to(OUT)),sha256=sha(path),rows=len(p),last_meta_episode=last,sources=sources,plot_csv=str(samplepath.relative_to(OUT)),summary_csv=str(summarypath.relative_to(OUT)),sequence_presentations=int(v[:,10].sum()),query_examples=int(v[:,11].sum()),query_gradient_evaluations=int(v[:,12].sum()),episodes=int(v[:,6].sum()),inner_updates=int(v[:,7].sum()),outer_updates=int(v[:,8].sum()),interpolations=int(v[:,9].sum()));dump(receipt,r);return r

def main():
 receipt=OUT/'audit/mechanism_complete.json'
 if receipt.exists():return
 assert json.loads((OUT/'audit/phase_D_verified.json').read_text())['passed'];configure();jobs=[(s,k,c,a) for s in SEEDS for k in SCALES for c in ['U','P'] for a in ['R']+ARMS]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:reg=list(pool.map(export_history,jobs))
 write_frame(OUT/'registry/scale_histories.csv',[{k:v for k,v in r.items() if k!='sources'} for r in reg]);sample=pd.concat([pd.read_csv(OUT/r['plot_csv'],float_precision='round_trip') for r in reg],ignore_index=True);write_frame(OUT/'tables/scale_trajectory_plot.csv',sample);summ=pd.concat([pd.read_csv(OUT/r['summary_csv'],float_precision='round_trip') for r in reg],ignore_index=True);write_frame(OUT/'tables/projection_summary.csv',summ)
 dump(OUT/'tables/scale_histories/SCHEMA.json',dict(projection_shape=['outer_iterations',GROUPS,FIELDS],scalar_columns=['meta_episode','outer_iteration','recurrent_spectral_norm','AB_cosine','AB_column_difference','centered_contrast','episodes','inner_updates','outer_updates','interpolations','sequence_presentations','query_examples','query_gradient_evaluations','dense_centered_contrast','dense_AB_cosine','dense_AB_column_difference','dense_shared_C_cosine'],dense_position_contrast_norms_shape=['outer_iterations',7],dense_geometry='float64 weight-only recurrence, no model rollout; original missing logged fields retained separately',historical_U_AB_cosine='original logged field not saved; NaN retained; dense_AB_cosine is separately derived exactly from weights at every outer step',all_native_steps_retained=True,plotting_sampling='exact every60 outer iterations plus registered and phase-initial steps; no averaging/smoothing',NPZ_source_hashes='per-file JSON receipts',legacy_U_displacement='projection displacement exactly zero; raw outer displacement remains available in immutable source before/after arrays'))
 rows=[];positions=[];directions=[]
 for s in SEEDS:
  for k in SCALES:
   freshpath=OUT/f'initial/s{s:02d}_scale{k}.pt';fresh=torch.load(freshpath,weights_only=False)['model']
   for c,a,ds in [('Shared','Fresh',[0])]+[(c,a,[240,960,1920] if a=='R' else DOSES) for c in ['U','P'] for a in ['R']+ARMS]:
    for dose in ds:
     state,path=(fresh,freshpath) if a=='Fresh' else initial(s,'P2','I',f'{k}-{c}-{a}',dose);q,pp,dd=geometry(state,fresh);tag=dict(seed=s,scale=k,condition=c,phase=a,dose=dose,source=str(path.relative_to(ROOT)),source_sha=sha(path));rows.append(dict(tag,**q));positions.extend(dict(tag,**p) for p in pp);directions.extend(dict(tag,**d) for d in dd)
 geom=pd.DataFrame(rows);assert len(geom)==1134;out=pd.read_csv(OUT/'tables/outcomes.csv',float_precision='round_trip');z=out[out['update'].eq(0)][['seed','scale','condition','phase','dose','NoContext_ce','NoContext_accuracy','C_ce','gap']];geom=geom.merge(z,on=['seed','scale','condition','phase','dose'],validate='one_to_one');assert len(geom)==1134;write_frame(OUT/'tables/meta_geometry.csv',geom);write_frame(OUT/'tables/effective_context_by_position.csv',positions);write_frame(OUT/'tables/parameter_directions.csv',directions)
 dump(receipt,dict(passed=True,full_trajectories=len(reg),geometry_checkpoints=len(geom),positions=len(positions),parameter_coordinates=len(directions),no_model_rollouts=True,no_optimizer_updates=True,all_projection_steps_exported=True));append('## Weight-only mechanism analysis\nAll180 U/P phase trajectories exported as complete per-outer-update five-group arrays with exact before/raw/projected RMS, floors, multipliers, projection displacements/flags. Original U prefixes are referenced, not replayed. Exact sampled plotting rows and full NPZ schema/hash receipts retained. All1134 registered fresh/meta states have seven-position centered effective-context contrasts, directions relative to fresh, recurrent norm and independent recurrence-sum identity checks. NoContext computation comes from saved zero-target-update evaluations. Geometry is descriptive, not proof of independent skills or causal mediation.')
if __name__=='__main__':main()
