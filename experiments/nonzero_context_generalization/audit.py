"""Pre-outcome source/registry seal and prescribed frozen numerical preflight."""
import shutil,platform,sys
from .core import *

def validate_checkpoint(d,v,zarchive,harchive,archive_check=True):
 before=fingerprint(d);s,k=operators(d);z0,h0=baseline64(d);controls=[];zs=[];hs=[];zero_info={}
 for n in [1,20]:
  zz,hh=inject(d,np.zeros((n,8),np.float32));boundh,boundz=numeric_bounds(d,np.zeros((n,8)),zz,hh)
  if archive_check:
   ah,az=numeric_bounds(d,np.zeros((1,8)),zarchive[None],harchive[None]);allowh=boundh+ah;allowz=boundz+az;tol=1e-10+1e-12*max(abs(z0).max(),abs(h0).max());assert np.all(abs(zz-zarchive)<=allowz+tol) and np.all(abs(hh-harchive)<=allowh+tol);prob=assess(zz)['probabilities'];ref=assess(zarchive)['probabilities'];assert np.all(abs(prob-ref)<=np.minimum(1,2*allowz.max(-1)[...,None])+2*tol)
   zero_info.update({f'zero_batch{n}_logit_error':float(abs(zz-zarchive).max()),f'zero_batch{n}_hidden_error':float(abs(hh-harchive).max()),f'zero_batch{n}_probability_error':float(abs(prob-ref).max()),f'zero_batch{n}_exact_values':bool(np.array_equal(zz,np.broadcast_to(zarchive,zz.shape))),f'zero_batch{n}_passed':True})
 for alpha in NONZERO:
  vv=np.asarray(alpha*v,np.float32);z,h=inject(d,vv);columnz,columnh=column_rollout(d,vv);doublez,doubleh=column_rollout(d,vv,torch.float64);expectedz=z0[None]+np.einsum('tcj,nj->ntc',k,vv.astype(float));expectedh=h0[None]+np.einsum('tij,nj->nti',s,vv.astype(float));tol=1e-10+1e-12*max(abs(expectedz).max(),abs(expectedh).max());assert abs(doublez-expectedz).max()<=tol and abs(doubleh-expectedh).max()<=tol
  a=validate_numeric(d,vv,z,h,s,k,z0,h0);b=validate_numeric(d,vv,columnz,columnh,s,k,z0,h0)
  assert np.all(np.linalg.norm(h[:,0]-inject(d,np.zeros((1,8)))[1][0,0],axis=1)>0)
  for t,r in enumerate(a):controls.append(dict(alpha=alpha,**r,direct_column_logit_error=float(abs(columnz[:,t]-z[:,t]).max()),direct_column_allowance=r['logit_allowance']+b[t]['logit_allowance']+2*tol,double_column_logit_error=float(abs(doublez[:,t]-expectedz[:,t]).max()),double_column_hidden_error=float(abs(doubleh[:,t]-expectedh[:,t]).max()),double_column_tolerance=tol,native_double_success_disagreements=int((success(assess(z)['ce'].mean(1),assess(z)['accuracy'].mean(1))!=success(assess(expectedz)['ce'].mean(1),assess(expectedz)['accuracy'].mean(1))).sum())))
  zs.append(z.astype(np.float32));hs.append(h.astype(np.float32))
 assert fingerprint(d)==before
 return controls,dict(logits=np.stack(zs),hidden=np.stack(hs),zero_info=zero_info)

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if (OUT/'audit/preflight_started.json').exists():raise RuntimeError('Preflight already started; no repeated probe branches')
 parent=json.loads((SOURCE/'manifest.json').read_text());check_hashes(SOURCE,parent['outputs']);check_hashes(ROOT,parent['code']);source_sha=sha(SOURCE/'manifest.json');dump(OUT/'audit/preflight_started.json',dict(parent_manifest_sha=source_sha,code=code_hashes(),protocol_sha=sha(ROOT/'docs/nonzero_context_generalization/PRD.md'),numerical_tolerance='1e-10+1e-12scale;native64eps absolute summands propagated',checks=CHECKS,training_updates=0))
 raw,vectors=registry();np.savez_compressed(OUT/'C_registry.npz',raw_float64=raw,used_float32=vectors);vr=[]
 for seed in range(20):
  for j in range(20):vr.append(dict(seed=seed,vector=j,used_norm=float(np.linalg.norm(vectors[seed,j].astype(float))),**{f'w_{q}':float(vectors[seed,j,q]) for q in range(8)},**{f'raw64_{q}':float(raw[seed,j,q]) for q in range(8)}))
 pd.DataFrame(vr).to_csv(OUT/'tables/C_vectors.csv',index=False);dump(OUT/'audit/registry.json',dict(rng='NumPy PCG64/SeedSequence([20261120,90473,seed])',shape=[20,20,8],distribution='independent N(0,4 I8)',raw_dtype='float64',used_dtype='float32',alphas=SCALES,primary_alpha=1.,registry_sha256=sha(OUT/'C_registry.npz'),all_vectors_nonzero=bool(np.all(np.linalg.norm(vectors,axis=2)>0)),paired_across_arms=True,fixed_across_states_and_positions=True))
 registry_frame=pd.read_csv(BASE/'tables/checkpoint_registry.csv',float_precision='round_trip');registry_frame['path']=registry_frame.path.map(lambda n:str((SOURCE/n).relative_to(ROOT)));registry_frame.to_csv(OUT/'tables/checkpoint_registry.csv',index=False)
 inputs={str((SOURCE/n).relative_to(ROOT)):h for n,h in parent['outputs'].items() if n.startswith('part_b/') and (n.endswith('.csv') or n.endswith('.pt') or '/figures/01_' in n)}
 inputs[str((SOURCE/'manifest.json').relative_to(ROOT))]=source_sha
 refs=pd.read_csv(BASE/'tables/evaluations.csv',float_precision='round_trip');refs=refs[refs.cue!='NoContext'].copy();refs.loc[refs.cue=='C','cue']='C-zero';refs[['seed','arm','update','cue','ce','accuracy']].to_csv(OUT/'tables/reference_evaluations.csv',index=False)
 for name,new in [('training','source_training'),('training_intervals','source_training_intervals'),('geometry','source_geometry'),('block_transfer','source_block_transfer'),('switch_costs','source_switch_costs')]:shutil.copyfile(BASE/'tables'/f'{name}.csv',OUT/'tables'/f'{new}.csv')
 for name in ['01_training_loss','01_training_loss_scale_companion']:
  for ext in ['png','svg']:shutil.copyfile(BASE/'figures'/f'{name}.{ext}',OUT/'figures'/f'{name}.{ext}')
 zero=[];fidelity=[]
 for x in pd.read_csv(BASE/'tables/position_predictions.csv',chunksize=42000,float_precision='round_trip'):
  zero.append(x[x.cue=='C']);fidelity.append(x[(x.cue.isin(['A','B']))&x['update'].isin(CHECKS)])
 zeros=pd.concat(zero,ignore_index=True);ab=pd.concat(fidelity,ignore_index=True);zeros.to_csv(OUT/'tables/archived_zero_positions.csv',index=False);ab.to_csv(OUT/'tables/archived_AB_validation_positions.csv',index=False);zg=zeros.groupby(['seed','arm','update']);ag=ab.groupby(['seed','arm','update','cue']);all_checks=[];zero_checks=[];source_checks=[]
 for rr in registry_frame.itertuples():
  path=ROOT/rr.path;assert sha(path)==rr.sha256;payload=torch.load(path,map_location='cpu',weights_only=False);before=fingerprint(payload);cachedz=[];cachedh=[]
  for step in CHECKS:
   d=payload['states'][step]['model'];f=zg.get_group((rr.seed,rr.arm,step)).sort_values('position');z=f[[f'logit_{i}' for i in range(4)]].to_numpy();h=f[[f'hidden_{i}' for i in range(8)]].to_numpy();assert not torch.count_nonzero(d[PARAMS[0]][:,6]);controls,cache=validate_checkpoint(d,vectors[rr.seed],z,h);meta=dict(seed=rr.seed,arm=rr.arm,update=step);all_checks.extend(dict(**meta,**r) for r in controls);zero_checks.append(dict(**meta,**cache['zero_info']));cachedz.append(cache['logits']);cachedh.append(cache['hidden'])
   for cue,c in [('A',[1,0,0]),('B',[0,1,0])]:
    zz,hh=inject(d,np.zeros((1,8),np.float32),cue=c);ref=ag.get_group((rr.seed,rr.arm,step,cue)).sort_values('position');az=ref[[f'logit_{i}' for i in range(4)]].to_numpy();ah=ref[[f'hidden_{i}' for i in range(8)]].to_numpy();tol=1e-10+1e-12*max(abs(az).max(),abs(ah).max());assert abs(zz[0]-az).max()<=tol and abs(hh[0]-ah).max()<=tol;source_checks.append(dict(**meta,cue=cue,logit_error=float(abs(zz[0]-az).max()),hidden_error=float(abs(hh[0]-ah).max()),passed=True))
  assert fingerprint(payload)==before and sha(path)==rr.sha256;np.savez_compressed(OUT/'cache'/f'validation_seed_{rr.seed}_{rr.arm}.npz',logits=np.stack(cachedz),hidden=np.stack(cachedh),checks=np.array(CHECKS),checkpoint_sha=np.array(rr.sha256),registry_sha=np.array(sha(OUT/'C_registry.npz')));print(f'Preflight seed{rr.seed:02d} {rr.arm}:six frozen states passed',flush=True)
 pd.DataFrame(all_checks).to_csv(OUT/'tables/preflight_numeric_checks.csv',index=False);pd.DataFrame(zero_checks).to_csv(OUT/'tables/zero_path_checks.csv',index=False);pd.DataFrame(source_checks).to_csv(OUT/'tables/AB_unchanged_checks.csv',index=False);check_hashes(ROOT,inputs)
 dump(OUT/'audit/environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,threads=1,deterministic_algorithms=True))
 dump(OUT/'audit/preflight_complete.json',dict(parent_manifest_sha=source_sha,inputs=inputs,code=code_hashes(),registry_sha=sha(OUT/'C_registry.npz'),models=80,validation_states=480,nonzero_native_batches_cached=1920,training_updates=0,optimizer_constructions=0,parameter_mutations=0,all_checks_passed=True))
 append('## Prescribed preflight complete\nAll80 original payloads andallsource hashes unchanged;20vectors/seed fixed. Allsix prespecified checkpoints pass zero-path,nonzero injection,column equivalence,double affine andnativebounds checks. A/B source replay unchanged. Nonzero nativevalidation results cached for reuse by fullrun. No optimizer,backward,modelconstruction orweightmutation. Beforethisgate no performance-driven decision occurred.')
if __name__=='__main__':main()
