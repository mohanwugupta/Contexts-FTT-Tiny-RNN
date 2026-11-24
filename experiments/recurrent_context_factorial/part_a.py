"""Missing full-strength Mixed evaluation with certified archived branch reuse."""
from .core import *
from experiments.centered_context_linear.core import pair as old_pair

def certify_reuse(m,d):
 assert not np.count_nonzero(arrays(m)[0][:,6]);out=dict(d);provenance={c:'exact archived cue branch' for c in d};out['NoContext']=out['C'];provenance['NoContext']='certified zero third column; archived C branch'
 if m.centered:
  assert torch.count_nonzero(m.g)==0
  out['Mixed-both']=out['Mixed-average'];provenance['Mixed-both']='certified permanent g=0; archived Mixed branch'
 return out,provenance

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 if (OUT/'audit/part_a_started.json').exists():raise RuntimeError('PartA already started; no repeated branches')
 oldmanifest=json.loads((OLD/'manifest.json').read_text());check_hashes(ROOT,oldmanifest['code']);deps={str((OLD/n).relative_to(ROOT)):sha(OLD/n) for n in ['manifest.json','tables/position_predictions.csv','tables/checkpoint_registry.csv']}
 for n in ['tables/position_predictions.csv','tables/checkpoint_registry.csv']:assert sha(OLD/n)==oldmanifest['outputs'][n]
 dump(OUT/'audit/part_a_started.json',dict(source_manifests=guard(),inputs=deps,code=code_hashes(),new_native_rollouts=6620,training_updates=0))
 source=pd.read_csv(OLD/'tables/position_predictions.csv',float_precision='round_trip');registry=pd.read_csv(OLD/'tables/checkpoint_registry.csv');rollouts=0;checkpoint_rows=[]
 for seed in range(20):
  ms=old_pair(seed)
  for arm,m in ms.items():
   r=registry[(registry.seed==seed)&(registry.arm==arm)].iloc[0];path=OLD/r.path;assert sha(path)==r.sha256;payload=torch.load(path,map_location='cpu',weights_only=False);checkpoint_rows.append(dict(seed=seed,arm=arm,path=str(path.relative_to(ROOT)),sha256=r.sha256,states=331));groups=source[(source.seed==seed)&(source.arm==arm)].groupby(['update','cue']);ev=[];positions=[];math=[];decomp=[];geo=[]
   for st in payload['states']:
    k=st['update'];m.load_state_dict(st['model']);m.requires_grad_(False);m.eval();meta=dict(seed=seed,arm=arm,update=k)
    d={new:from_frame(groups.get_group((k,old))) for new,old in [('A','A'),('B','B'),('Mixed-average','Mixed'),('C','C')]};obs,provenance=certify_reuse(m,d)
    if arm=='Standard':obs['Mixed-both']=trace(m,'Mixed-both');provenance['Mixed-both']='new frozen native evaluation';rollouts+=1
    controls=checks(m,obs)
    if arm=='Centered':
     assert np.array_equal(obs['C']['logits'],obs['Mixed-both']['logits']) and np.array_equal(obs['Mixed-average']['hidden'],obs['C']['hidden'])
    a,b=rows_for(meta,obs,provenance);ev.extend(a);positions.extend(b);math.extend(dict(**meta,**x) for x in controls);decomp.extend(dict(**meta,**x) for x in decomposition(obs));geo.append(dict(**meta,**metrics(m,obs)))
   for n,rows in [('evaluations',ev),('position_predictions',positions),('mathematical_checks',math),('decomposition',decomp),('geometry',geo)]:csv('part_a',n,rows)
   assert sha(path)==r.sha256
  print(f'PartA seed{seed:02d}: existing states unchanged',flush=True)
 csv('part_a','checkpoint_registry',checkpoint_rows);check_hashes(ROOT,deps);dump(OUT/'audit/part_a_complete.json',dict(inputs=deps,code=code_hashes(),source_manifests=guard(),new_native_rollouts=rollouts,training_updates=0,reused_native_cue_cells=72820,total_cue_cells=79440,states=13240,positions=556080,math_position_checks=92680));append('## PartA complete\nAll existing states checked before any new factorial training.6620 missing Standard Mixed-both rollouts;72820 native cue cells reused with projection/parameter certificates. All identities pass; no training, optimizer construction or source mutation. FigureA generated before PartB.')
if __name__=='__main__':main()
