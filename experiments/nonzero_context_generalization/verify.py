"""Read-only reconstruction from frozen tensors and cached logits; no native rollout."""
import argparse
from .core import *
from .analysis import reduce_network,paired
from experiments.centered_context_linear.core import interval

def verify(sealed=True):
 pre=json.loads((OUT/'audit/preflight_complete.json').read_text());done=json.loads((OUT/'audit/evaluation_complete.json').read_text());check_hashes(ROOT,pre['inputs']);assert done['inputs']==pre['inputs'];assert sha(SOURCE/'manifest.json')==pre['parent_manifest_sha'];assert sha(OUT/'C_registry.npz')==pre['registry_sha']==done['registry_sha'];parent=json.loads((SOURCE/'manifest.json').read_text());check_hashes(ROOT,parent['code'])
 for name in ['core','audit']:assert sha(ROOT/f'experiments/nonzero_context_generalization/{name}.py')==pre['code'][f'experiments/nonzero_context_generalization/{name}.py']
 assert sha(ROOT/'experiments/nonzero_context_generalization/evaluate.py')==done['code']['experiments/nonzero_context_generalization/evaluate.py'];assert done['native_batches_new']==104000 and done['native_batches_reused']==1920 and done['training_updates']==done['optimizer_constructions']==done['parameter_mutations']==0
 read=lambda n:pd.read_csv(OUT/'tables'/f'{n}.csv',float_precision='round_trip');reg=read('checkpoint_registry');assert len(reg)==80 and not reg.duplicated(['seed','arm']).any();raw,v=registry();saved=np.load(OUT/'C_registry.npz');np.testing.assert_array_equal(raw,saved['raw_float64']);np.testing.assert_array_equal(v,saved['used_float32']);assert (np.linalg.norm(v,axis=-1)>0).all()
 refs=read('reference_evaluations');assert len(refs)==132400;rp=refs.pivot(index=['seed','arm','update'],columns='cue',values='ce');ra=refs.pivot(index=['seed','arm','update'],columns='cue',values='accuracy');seed=read('seed_trajectories');primary=read('primary_vector_trajectories');numeric=read('numerical_checks');assert len(numeric)==741440 and numeric.passed.all();assert (numeric.logit_residual<=numeric.logit_allowance+numeric.double_tolerance).all();assert (numeric.ce_residual<=2*numeric.logit_allowance+2*numeric.double_tolerance).all();positions=read('primary_position_predictions');assert len(positions)==67200 and not positions.duplicated(['seed','arm','update','vector','position']).any();ops=read('response_operator_positions');assert len(ops)==185360;zero=read('archived_zero_positions');network_count=0;allcells=0;primarycells=0
 for f in pd.read_csv(OUT/'tables/vector_generalization.csv.gz',chunksize=33100,float_precision='round_trip'):
  assert network_count<80;rr=reg.iloc[network_count];network_count+=1;assert len(f)==33100 and f.seed.eq(rr.seed).all() and f.arm.eq(rr.arm).all() and not f.duplicated(['update','alpha','vector']).any();assert set(f.alpha)==set(SCALES) and set(f['update'])==set(range(331)) and set(f.vector)==set(range(20));allcells+=len(f)
  path=ROOT/rr.path;assert sha(path)==rr.sha256;payload=torch.load(path,map_location='cpu',weights_only=False);cache=np.load(OUT/'cache'/f'network_seed_{rr.seed}_{rr.arm}.npz');assert cache['checkpoint_sha'].item()==rr.sha256 and cache['registry_sha'].item()==pre['registry_sha'];np.testing.assert_array_equal(cache['vectors'],v[rr.seed]);np.testing.assert_array_equal(cache['scales'],NONZERO);zall=cache['logits'].astype(float);ph=cache['primary_hidden'].astype(float);assert zall.shape==(331,4,20,7,4) and ph.shape==(331,20,7,8);K=cache['K'];S=cache['S'];zbase=cache['zero_logits'].astype(float);hbase=cache['zero_hidden'].astype(float)
  sourcezero=zero[(zero.seed==rr.seed)&(zero.arm==rr.arm)].sort_values(['update','position']);np.testing.assert_array_equal(zbase,sourcezero[[f'logit_{j}' for j in range(4)]].to_numpy().reshape(331,7,4));np.testing.assert_array_equal(hbase,sourcezero[[f'hidden_{j}' for j in range(8)]].to_numpy().reshape(331,7,8));index=pd.MultiIndex.from_arrays([np.full(331,rr.seed),np.full(331,rr.arm),np.arange(331)],names=['seed','arm','update']);czero=rp.loc[index,'C-zero'].to_numpy();azero=ra.loc[index,'C-zero'].to_numpy();lf=rp.loc[index,['A','B']].mean(1).to_numpy();good=((rp.loc[index,['A','B']]<=.5).all(1)&(ra.loc[index,['A','B']]>=6/7).all(1)).to_numpy();tol=1e-10+1e-12*max(abs(zall).max(),abs(zbase).max());base=assess(zbase)
  for alpha in SCALES:
   x=f[f.alpha==alpha].sort_values(['update','vector']);assert len(x)==6620
   if alpha==0:
    expectedce=np.repeat(czero,20);expectedacc=np.repeat(azero,20);expectedmargin=np.repeat(base['margin'].mean(1),20);response_norm=np.zeros((331,20));native_norm=response_norm;change=response_norm;fixed=response_norm;penalty=response_norm
   else:
    z=zall[:,NONZERO.index(alpha)];a=assess(z);expectedce=a['ce'].mean(-1).ravel();expectedacc=a['accuracy'].mean(-1).ravel();expectedmargin=a['margin'].mean(-1).ravel();response=np.einsum('ktcj,nj->kntc',K,(alpha*v[rr.seed]).astype(float));delta=z-zbase[:,None];response_norm=np.linalg.norm(centered(response),axis=-1).mean(-1);native_norm=np.linalg.norm(centered(delta),axis=-1).mean(-1);change=(a['margin']-base['margin'][:,None]).mean(-1);wrong=np.where(np.arange(4)==TARGET[:,None],-np.inf,zbase).argmax(-1);ii=np.broadcast_to(wrong[:,None,:,None],(331,20,7,1));targetii=np.broadcast_to(TARGET[None,None,:,None],(331,20,7,1));fixed=(np.take_along_axis(delta,targetii,axis=-1)[...,0]-np.take_along_axis(delta,ii,axis=-1)[...,0]).mean(-1);penalty=(np.where(np.arange(4)==TARGET[:,None],-np.inf,z).max(-1)-np.take_along_axis(z,ii,axis=-1)[...,0]).mean(-1)
   for field,expected in [('ce',expectedce),('accuracy',expectedacc),('margin',expectedmargin),('delta_zero',expectedce-np.repeat(czero,20)),('gap',expectedce-np.repeat(lf,20)),('margin_change',change.ravel()),('novel_contribution_norm',response_norm.ravel()),('native_delta_norm',native_norm.ravel()),('fixed_competitor_margin',fixed.ravel()),('competitor_switch_penalty',penalty.ravel())]:np.testing.assert_allclose(x[field],expected,atol=tol,rtol=1e-12)
   np.testing.assert_array_equal(x.strong_familiar,np.repeat(good,20));np.testing.assert_array_equal(x.success,success(expectedce,expectedacc));np.testing.assert_array_equal(x.above_chance,expectedce>CHANCE)
  calculated=reduce_network(f);savedseed=seed[(seed.seed==rr.seed)&(seed.arm==rr.arm)].sort_values(['seed','arm','update','alpha']).reset_index(drop=True);pd.testing.assert_frame_equal(calculated,savedseed,check_exact=False,atol=tol,rtol=1e-12);pp=primary[(primary.seed==rr.seed)&(primary.arm==rr.arm)].sort_values(['update','vector']).reset_index(drop=True);pd.testing.assert_frame_equal(pp,f[f.alpha==1].sort_values(['update','vector']).reset_index(drop=True),check_exact=False,atol=1e-12);primarycells+=len(pp)
  assert np.all(np.linalg.norm(ph[:,:,0]-hbase[:,None,0],axis=-1)>0)
  on=ops[(ops.seed==rr.seed)&(ops.arm==rr.arm)].sort_values(['update','position']);np.testing.assert_allclose(on.K_raw_spectral_norm,np.linalg.norm(K,2,axis=(-2,-1)).ravel(),atol=1e-10,rtol=1e-12);np.testing.assert_allclose(on.K_centered_spectral_norm,np.linalg.norm(K-K.mean(-2,keepdims=True),2,axis=(-2,-1)).ravel(),atol=1e-10,rtol=1e-12)
  for kstep,st in enumerate(payload['states']):
   assert st['update']==kstep and not torch.count_nonzero(st['model'][PARAMS[0]][:,6]);ss,kk=operators(st['model']);np.testing.assert_array_equal(S[kstep],ss);np.testing.assert_array_equal(K[kstep],kk)
  for kstep in CHECKS:
   x=positions[(positions.seed==rr.seed)&(positions.arm==rr.arm)&(positions['update']==kstep)].sort_values(['vector','position']);z=zall[kstep,NONZERO.index(1.)];a=assess(z);np.testing.assert_array_equal(x[[f'logit_{j}' for j in range(4)]],z.reshape(-1,4));np.testing.assert_array_equal(x[[f'hidden_{j}' for j in range(8)]],ph[kstep].reshape(-1,8));np.testing.assert_allclose(x[[f'probability_{j}' for j in range(4)]],a['probabilities'].reshape(-1,4),atol=1e-12);np.testing.assert_allclose(x.ce,a['ce'].ravel(),atol=1e-10);np.testing.assert_allclose(x.margin,a['margin'].ravel(),atol=1e-10)
 assert network_count==80 and allcells==2648000 and primarycells==529600 and len(seed)==132400
 summary=read('endpoint_summary');end=read('seed_endpoints')
 for r in summary.itertuples():
  f=end[(end.arm==r.arm)&(end['update']==r.update)&(end.alpha==r.alpha)];f=f if r.subgroup=='all' else f[f.strong_familiar];assert len(f)==r.n;np.testing.assert_allclose(interval(f[r.metric]),[r.mean,r.ci_low,r.ci_high],atol=1e-10,rtol=1e-12,equal_nan=True)
 contrasts=read('factorial_seed_contrasts')
 for metric in contrasts.metric.unique():
  expected=paired(end,metric).sort_values(['seed','update','alpha']);actual=contrasts[contrasts.metric==metric].sort_values(['seed','update','alpha']);np.testing.assert_allclose(expected[['normalization','centering','interaction','NC_minus_NI']],actual[['normalization','centering','interaction','NC_minus_NI']],atol=1e-10)
 for r in read('factorial_intervals').itertuples():
  f=contrasts[(contrasts['update']==r.update)&(contrasts.alpha==r.alpha)&(contrasts.metric==r.metric)];assert len(f)==20;np.testing.assert_allclose(interval(f[r.effect]),[r.mean,r.ci_low,r.ci_high],atol=1e-10)

 # Mechanism comparisons use the same paired seed estimands, not separate-cohort intervals.
 mechanisms=read('primary_mechanism');mraw=read('mechanism_seed_contrasts')
 for metric in mraw.metric.unique():
  mm=mechanisms[mechanisms['update'].isin(PROBES)].copy();mm['alpha']=1.;expected=paired(mm,metric).sort_values(['seed','update']);actual=mraw[mraw.metric==metric].sort_values(['seed','update']);np.testing.assert_allclose(expected[['normalization','centering','interaction','NC_minus_NI']],actual[['normalization','centering','interaction','NC_minus_NI']],atol=1e-10)
 for r in read('mechanism_paired_intervals').itertuples():
  f=mraw[(mraw['update']==r.update)&(mraw.metric==r.metric)];assert len(f)==20;np.testing.assert_allclose(interval(f[r.effect]),[r.mean,r.ci_low,r.ci_high],atol=1e-10)

 if sealed:
  m=json.loads((OUT/'manifest.json').read_text());check_hashes(OUT,m['outputs']);check_hashes(ROOT,m['code'])
 return dict(passed=True,networks=80,trained_seeds=20,vectors_per_seed=20,all_scale_probe_cells=allcells,primary_probe_cells=primarycells,primary_position_rows=67200,training_updates=0,optimizer_constructions=0,parameter_mutations=0,native_model_rollouts_during_verification=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--unsealed',action='store_true');p.add_argument('--regenerate',type=Path);a=p.parse_args();r=verify(not a.unsealed)
 if a.regenerate:
  if a.regenerate.exists() and list(a.regenerate.iterdir()):raise ValueError('Empty regeneration destination required')
  from .figures import generate
  names=generate(a.regenerate,regenerate_training=True)
  for n in names:assert sha(a.regenerate/n)==sha(OUT/'figures'/n),n
  r['byte_identical_figure_files']=len(names)
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
