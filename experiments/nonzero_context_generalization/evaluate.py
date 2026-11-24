"""One fixed frozen evaluation; validation probe cells reused, never retrained."""
import gzip
from .core import *
from .analysis import reduce_network

def append_csv(path,frame,compressed=False):
 exists=path.exists()
 if compressed:
  with gzip.open(path,'at',compresslevel=6) as f:frame.to_csv(f,index=False,header=not exists)
 else:frame.to_csv(path,index=False,mode='a',header=not exists)

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if (OUT/'audit/evaluation_started.json').exists():raise RuntimeError('Fixed frozen evaluation already started; no outcome-driven reruns')
 pre=json.loads((OUT/'audit/preflight_complete.json').read_text());check_hashes(ROOT,pre['inputs']);assert sha(OUT/'C_registry.npz')==pre['registry_sha'];dump(OUT/'audit/evaluation_started.json',dict(preflight_sha=sha(OUT/'audit/preflight_complete.json'),code=code_hashes(),primary_alpha=1.,scales=SCALES,planned_nonzero_probe_cells=2118400,planned_vector_cells_all_scales=2648000,training_updates=0,optimizers=0))
 probes=np.load(OUT/'C_registry.npz')['used_float32'];reg=pd.read_csv(OUT/'tables/checkpoint_registry.csv');zero=pd.read_csv(OUT/'tables/archived_zero_positions.csv',float_precision='round_trip');zg=zero.groupby(['seed','arm','update']);refs=pd.read_csv(OUT/'tables/reference_evaluations.csv',float_precision='round_trip');ce=refs.pivot(index=['seed','arm','update'],columns='cue',values='ce');ac=refs.pivot(index=['seed','arm','update'],columns='cue',values='accuracy');oldgeo=pd.read_csv(OUT/'tables/source_geometry.csv',float_precision='round_trip').set_index(['seed','arm','update']);native_calls=0;reused_calls=0;checks_total=0
 for rr in reg.itertuples():
  path=ROOT/rr.path;assert sha(path)==rr.sha256;payload=torch.load(path,map_location='cpu',weights_only=False);before=fingerprint(payload);val=np.load(OUT/'cache'/f'validation_seed_{rr.seed}_{rr.arm}.npz');assert val['checkpoint_sha'].item()==rr.sha256 and val['registry_sha'].item()==pre['registry_sha'];v=probes[rr.seed];all_logits=[];primary_hidden=[];allk=[];alls=[];zeroz=[];zeroh=[];vector_rows=[];position_rows=[];numeric=[];mechanism=[];oprows=[]
  for st in payload['states']:
   kstep=st['update'];key=(rr.seed,rr.arm,kstep);meta=dict(seed=rr.seed,arm=rr.arm,update=kstep);d=st['model'];assert torch.count_nonzero(d[PARAMS[0]][:,6])==0;sourcezero=zg.get_group(key).sort_values('position');zbase=sourcezero[[f'logit_{i}' for i in range(4)]].to_numpy();hbase=sourcezero[[f'hidden_{i}' for i in range(8)]].to_numpy();zeroz.append(zbase.astype(np.float32));zeroh.append(hbase.astype(np.float32));s,k=operators(d);allk.append(k);alls.append(s);z0,h0=baseline64(d);lf=float((ce.loc[key,'A']+ce.loc[key,'B'])/2);strong=bool(ce.loc[key,'A']<=.5 and ce.loc[key,'B']<=.5 and ac.loc[key,'A']>=6/7 and ac.loc[key,'B']>=6/7);czero=float(ce.loc[key,'C-zero']);azero=float(ac.loc[key,'C-zero']);am=assess(zbase);mzero=float(am['margin'].mean());current=[]
   for alpha in SCALES:
    if alpha==0:
     values={n:np.zeros(20) for n in ['margin_change','novel_contribution_norm','native_delta_norm','fixed_competitor_margin','competitor_switch_penalty']};values.update(ce=np.full(20,czero),accuracy=np.full(20,azero),margin=np.full(20,mzero));origin='certified archived C-zero copy'
    else:
     ia=NONZERO.index(alpha);vv=np.asarray(v*alpha,np.float32)
     if kstep in CHECKS:
      iv=CHECKS.index(kstep);z=val['logits'][iv,ia].astype(float);h=val['hidden'][iv,ia].astype(float);reused_calls+=1;origin='reused preflight native probe'
     else:z,h=inject(d,vv);native_calls+=1;origin='new native frozen probe'
     ctrl=validate_numeric(d,vv,z,h,s,k,z0,h0);numeric.extend(dict(**meta,alpha=alpha,**x) for x in ctrl);checks_total+=len(ctrl);values=native_metrics(z,zbase,k,vv);current.append(z.astype(np.float32))
     if alpha==1:
      primary_hidden.append(h.astype(np.float32))
      if kstep in CHECKS:
       scored=assess(z);response=np.einsum('tcj,nj->ntc',k,vv.astype(float));delta=z-zbase[None];wrong=np.where(np.arange(4)==TARGET[:,None],-np.inf,zbase).argmax(1)
       for j in range(20):
        for t in range(7):position_rows.append(dict(**meta,vector=j,alpha=1.,position=t+1,token=int(SEQ[t]),target=int(TARGET[t]),ce=scored['ce'][j,t],correct=bool(scored['accuracy'][j,t]),margin=scored['margin'][j,t],baseline_margin=am['margin'][t],margin_change=scored['margin'][j,t]-am['margin'][t],novel_response_norm=float(np.linalg.norm(centered(response[j,t]))),fixed_C_competitor=int(wrong[t]),native_fixed_margin=delta[j,t,TARGET[t]]-delta[j,t,wrong[t]],**{f'logit_{q}':z[j,t,q] for q in range(4)},**{f'probability_{q}':scored['probabilities'][j,t,q] for q in range(4)},**{f'hidden_{q}':h[j,t,q] for q in range(8)},**{f'analytic_response_{q}':response[j,t,q] for q in range(4)}))
    for j in range(20):
     row=dict(**meta,alpha=alpha,vector=j,provenance=origin,strong_familiar=strong,familiar_ce=lf,baseline_ce=czero,**{n:float(x[j]) for n,x in values.items()});row.update(success=bool(success(row['ce'],row['accuracy'])),above_chance=bool(row['ce']>CHANCE),delta_zero=row['ce']-czero,gap=row['ce']-lf);vector_rows.append(row)
   all_logits.append(np.stack(current));g=oldgeo.loc[key];knorm=np.linalg.norm(k,2,axis=(-2,-1));pk=k-k.mean(1,keepdims=True);pknorm=np.linalg.norm(pk,2,axis=(-2,-1))
   mechanism.append(dict(**meta,K_raw_norm=float(knorm.mean()),K_centered_norm=float(pknorm.mean()),recurrent_spectral_norm=g.recurrent_spectral_norm,u_norm=g.u_norm,v_norm=g.v_norm,baseline_ce=czero,baseline_accuracy=azero,baseline_margin=mzero,hidden_separation=g.hidden_separation,centered_logit_separation=g.centered_logit_separation,hidden_centroid_distance=g.hidden_centroid_distance,g_norm=g.g_norm,d_norm=g.d_norm,strong_familiar=strong))
   for t in range(7):oprows.append(dict(**meta,position=t+1,K_raw_spectral_norm=knorm[t],K_centered_spectral_norm=pknorm[t],hidden_response_spectral_norm=np.linalg.norm(s[t],2),recurrent_spectral_norm=g.recurrent_spectral_norm))
  frame=pd.DataFrame(vector_rows);summary=reduce_network(frame);cachepath=OUT/'cache'/f'network_seed_{rr.seed}_{rr.arm}.npz';np.savez_compressed(cachepath,logits=np.stack(all_logits),primary_hidden=np.stack(primary_hidden),K=np.stack(allk),S=np.stack(alls),zero_logits=np.stack(zeroz),zero_hidden=np.stack(zeroh),vectors=v,scales=np.array(NONZERO),checkpoint_sha=np.array(rr.sha256),registry_sha=np.array(pre['registry_sha']))
  append_csv(OUT/'tables/vector_generalization.csv.gz',frame,True);append_csv(OUT/'tables/primary_vector_trajectories.csv',frame[frame.alpha==1]);append_csv(OUT/'tables/seed_trajectories.csv',summary);append_csv(OUT/'tables/primary_position_predictions.csv',pd.DataFrame(position_rows));append_csv(OUT/'tables/numerical_checks.csv',pd.DataFrame(numeric));append_csv(OUT/'tables/mechanism_trajectories.csv',pd.DataFrame(mechanism));append_csv(OUT/'tables/response_operator_positions.csv',pd.DataFrame(oprows));assert fingerprint(payload)==before and sha(path)==rr.sha256;print(f'Frozen seed{rr.seed:02d} {rr.arm}:331 states,20 probes,allscales retained',flush=True)
  # Per-network completion journal provides a resumable evidence boundary, not permission to rerun.
  with (OUT/'audit/completed_networks.jsonl').open('a') as f:f.write(json.dumps(dict(seed=rr.seed,arm=rr.arm,checkpoint_sha=rr.sha256,cache_sha=sha(cachepath)))+'\n')
 check_hashes(ROOT,pre['inputs']);dump(OUT/'audit/evaluation_complete.json',dict(inputs=pre['inputs'],parent_manifest_sha=pre['parent_manifest_sha'],code=code_hashes(),registry_sha=pre['registry_sha'],native_batches_new=native_calls,native_batches_reused=reused_calls,nonzero_probe_cells=2118400,all_scale_vector_cells=2648000,primary_cells=529600,seed_scale_rows=132400,numeric_position_checks=checks_total,full_primary_positions_exported=67200,training_updates=0,optimizer_constructions=0,parameter_mutations=0))
 append(f'## Fixed frozen evaluation complete\n{native_calls} new nativebatches plus{reused_calls} preflightbatches reused. All529600 primary and2648000 allscale vectorcells retained,includingcertifiedalpha0. Nativef32 logit arrays cachedforallstates/scales;primaryhidden,S,K,sourcezero and67200 explicitpositionrows saved. Allnumericchecks pass;allsource/model/policy/Adam fingerprints unchanged. No tuning,training,optimizers orparameterwrites.')
if __name__=='__main__':main()
