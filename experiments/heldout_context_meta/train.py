"""Full960-episode FOMAML runs, complete lossless inner/outer histories."""
import concurrent.futures,multiprocessing
from .common import *
from .protocol import support,queries
from .fomaml import episode,outer_optimizer,packed,layout

def run(job):
 initialize_process();seed,prior,org,arm=job;rid=f's{seed:02d}_{prior}_{org}_{arm}';done=OUT/f'meta/{rid}_complete.json'
 if done.exists():return json.loads(done.read_text())
 start=time.monotonic();state,source=initial(seed,prior,org);m=source_model(state);o=outer_optimizer(m);assert not o.state;failed=None;n=ng=0;blocks=[]
 for b in range(1,5):
  path=OUT/f'meta/{rid}_b{b}.pt';receipt=path.with_suffix('.json')
  if receipt.exists():
   rr=json.loads(receipt.read_text());assert sha(path)==rr['sha256'];bundle=torch.load(path,weights_only=False);last=bundle['records'][-1];failed=None if last['status']=='completed' else last['status']
   if failed is None:m.load_state_dict(last['meta']);o.load_state_dict(bundle['outer_optimizer'])
   n+=rr['inner_steps'];ng+=rr['outer_updates'];blocks.append(rr);continue
  registry=json.loads((OUT/f'registry/seed_{seed}_b{b}.json').read_text());records=[dict(episode=(b-1)*240,status='completed' if failed is None else failed,meta=copy.deepcopy(m.state_dict()) if failed is None else None)];history=np.full((240,21,3,174),np.nan,np.float32);steps=np.full((240,21,8),np.nan,np.float32);outerhist=np.full((241,3,172),np.nan,np.float32);outersteps=np.full((241,6),np.nan,np.float32);grad=np.full((240,172),np.nan,np.float32);ql=np.full((240,4,7,4),np.nan,np.float32);qh=np.full((240,4,7,8),np.nan,np.float32);summ=[];updates=[];count=ocount=0
  outerhist[0],outersteps[0]=packed(m,o)
  for i,ep in enumerate(registry['episodes']):
   e=ep['episode']
   if failed:
    records.append(dict(episode=e,status='unavailable_after_'+failed,meta=None));summ.append(dict(episode=e,kind=ep['kind'],status='unavailable_after_'+failed,inner_steps=0,outer_executed=False,query_ce=np.nan,query_gradient_norm=np.nan,query_unchanged=None,policy_reset=None,inner_optimizer_empty=None,query_count=len(queries(ep,prior)),Wh_norm_before=np.nan,Wh_norm_adapted=np.nan,Wh_norm_after=np.nan));continue
   r=episode(m,o,ep,prior,arm);count+=r['inner_steps'];ocount+=int(r['outer_executed']);failed=None if r['status']=='completed' else r['status'];history[i,:len(r['inner_history'])]=r['inner_history'];steps[i,:len(r['inner_steps_history'])]=r['inner_steps_history'];grad[i]=r['gradient'];qcount=len(r['query_cells'])
   if r['query_logits'] is not None:ql[i,:qcount]=r['query_logits'];qh[i,:qcount]=r['query_hidden']
   if failed is None:outerhist[i+1],outersteps[i+1]=packed(m,o)
   norm=lambda st:float(torch.linalg.matrix_norm(st['recurrent.weight_hh_l0'].double(),2)) if st is not None and torch.isfinite(st['recurrent.weight_hh_l0']).all() else np.nan
   summ.append(dict(episode=e,kind=ep['kind'],status=r['status'],inner_steps=r['inner_steps'],outer_executed=r['outer_executed'],query_ce=r['query_ce'],query_gradient_norm=r['query_gradient_norm'],query_unchanged=r['query_unchanged'],policy_reset=r['policy_reset'],inner_optimizer_empty=r['inner_optimizer_empty'],query_count=qcount,Wh_norm_before=norm(r['before']),Wh_norm_adapted=norm(r['adapted']),Wh_norm_after=norm(r['after'])));updates.extend(dict(episode=e,**x) for x in r['updates']);records.append(dict(episode=e,status=r['status'],meta=r['after'],before=r['before'],adapted=r['adapted'],adapted_policy=r['adapted_policy'],failure_snapshot=r.get('failure_snapshot')))
  torch.save(dict(seed=seed,prior=prior,organization=org,arm=arm,source=str(source.relative_to(ROOT)),source_sha=sha(source),records=records,outer_optimizer=copy.deepcopy(o.state_dict()),inner_layout=layout(m,core.policy()),outer_layout=layout(m),inner_optimizer_group=core.optimizer(m,core.policy()).state_dict()['param_groups'][0],outer_optimizer_group=o.state_dict()['param_groups'][0]),path);np.savez_compressed(path.with_suffix('.npz'),inner=history,inner_steps=steps,outer=outerhist,outer_steps=outersteps,query_gradient=grad,query_logits=ql,query_hidden=qh)
  write_frame(path.with_name(path.stem+'_episodes.csv'),summ);write_frame(path.with_name(path.stem+'_inner.csv'),updates);rr=dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm=arm,block=b,dose=b*240,path=str(path.relative_to(ROOT)),sha256=sha(path),packed_sha=sha(path.with_suffix('.npz')),status=failed or 'completed',inner_steps=count,outer_updates=ocount);dump(receipt,rr);blocks.append(rr);n+=count;ng+=ocount;print(f'{rid} +{b*240}: {rr["status"]}',flush=True)
 result=dict(seed=seed,prior=prior,organization=org,arm=arm,status=failed or 'completed',inner_steps=n,outer_updates=ng,wall_seconds=time.monotonic()-start,blocks=blocks);dump(done,result);return result

def main():
 seal=json.loads((OUT/'audit/protocol_seal.json').read_text());core.hashes(OUT,seal['registry']);start=time.monotonic();dump(OUT/'audit/meta_started.json',dict(code=codes(),source_manifest=seal['parent_manifest_sha'],inner_planned=3072000,outer_planned=153600));jobs=[(s,p,o,a) for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']];rows=[];blocks=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(run,jobs):blocks.extend(r.pop('blocks'));rows.append(r);print(f'Runs committed {len(rows)}/160',flush=True)
 write_frame(OUT/'registry/meta_runs.csv',rows);write_frame(OUT/'registry/meta_blocks.csv',blocks);dump(OUT/'audit/meta_complete.json',dict(runs=160,inner_updates=sum(r['inner_steps'] for r in rows),outer_updates=sum(r['outer_updates'] for r in rows),wall_seconds=time.monotonic()-start,code=codes()));append('## FOMAMLstageBcomplete\nAll160runstatusesand960episodeslotsretained;querygradientsdrivepersistentouterAdam,predictionparamsonly. All20innerparameter/policy/Adamstatespackedlosslesslyplusbefore/adapted/afterandouterstates. Fixed+240/+960,nofavorablecheckpointselectionorretrainingofold1920.')
if __name__=='__main__':main()
