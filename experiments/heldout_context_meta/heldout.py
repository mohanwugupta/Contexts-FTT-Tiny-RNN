"""Unseen identifiable conflict tasks; frozen initializations, diagnostic adaptation."""
import concurrent.futures,multiprocessing,argparse
from .common import *
from .protocol import support
from .fomaml import packed,layout
from .analysis import summarize

def conflict_cells(ep):return list(dict.fromkeys((r['context'],r['sequence']) for r in support(ep,'P2','Transfer')))
def evaluations(m,ep):
 return {c:core.trace(m,c,ep['paths'][j]) for c,j in conflict_cells(ep)}

def heldout_run(job):
 initialize_process();seed,prior,org,arm,dose,version=job;rid=f's{seed:02d}_{prior}_{org}_{arm}_d{dose}';dest=OUT/('heldout_diagnostics_v2' if version=='v2' else 'heldout_diagnostics')/rid
 if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
 dest.mkdir(parents=True);state,src=initial(seed,prior,org,arm,dose);eps=json.loads((OUT/f'registry/heldout_v2_seed_{seed}.json' if version=='v2' else OUT/f'registry/heldout_seed_{seed}.json').read_text())['episodes'];rows=[];blocks=[];pos=[];hist=np.full((12,21,3,174),np.nan,np.float32);steps=np.full((12,21,8),np.nan,np.float32);count=0;snapshots=[]
 for i,ep in enumerate(eps):
  if state is None:rows.append(dict(task=i,status='blocked_meta',ce=np.nan,accuracy=np.nan,conflict_accuracy=np.nan,appropriate_context_change=np.nan,prefix_centered_separation=np.nan));continue
  m=source_model(state);p=core.policy();o=core.optimizer(m,p);assert not o.state;failed=None;cells=conflict_cells(ep);hist[i,0],steps[i,0]=packed(m,o,p);before=evaluations(m,ep);previous=before
  for k,r in enumerate(support(ep,'P2','Transfer'),1):
   q=core.step(m,p,o,r['context'],ep['paths'][r['sequence']],meta=True);count+=int(q['optimizer_step_executed']);hist[i,k],steps[i,k]=packed(m,o,p)
   if q['status']!='completed':failed=q['status'];break
   if k%5==0:
    after=evaluations(m,ep);other=next(c for c,j in cells if c!=r['context']);own=previous[r['context']]['ce'].mean()-after[r['context']]['ce'].mean();cross=previous[other]['ce'].mean()-after[other]['ce'].mean();blocks.append(dict(task=i,end=k,practiced=r['context'],other=other,own_gain=own,cross_gain=cross,specificity=own-cross));previous=after
  if failed:rows.append(dict(task=i,status=failed,ce=np.nan,accuracy=np.nan,conflict_accuracy=np.nan,appropriate_context_change=np.nan,prefix_centered_separation=np.nan));continue
  out=evaluations(m,ep);xx=[out[c] for c,j in cells];ok=[bool(x['accuracy'][2]) for x in xx];pred=[int(x['logits'][2].argmax()) for x in xx];center=lambda z:z-z.mean(-1,keepdims=True);sep=np.linalg.norm(center(xx[0]['logits'][2])-center(xx[1]['logits'][2]));rows.append(dict(task=i,status='completed',ce=float(np.mean([x['ce'].mean() for x in xx])),accuracy=float(np.mean([x['accuracy'].mean() for x in xx])),conflict_accuracy=float(np.mean(ok)),appropriate_context_change=bool(all(ok) and pred[0]!=pred[1]),prefix_centered_separation=float(sep),full_centered_separation=float(np.linalg.norm(center(xx[0]['logits'])-center(xx[1]['logits']),axis=1).mean())))
  for c,j in cells:
   x=out[c]
   for t in range(7):pos.append(dict(task=i,context=c,sequence=j,position=t+1,target=ep['paths'][j][t+1],ce=x['ce'][t],correct=bool(x['accuracy'][t]),margin=x['margin'][t],**{f'logit_{z}':x['logits'][t,z] for z in range(4)},**{f'probability_{z}':x['probabilities'][t,z] for z in range(4)}))
  snapshots.append(dict(task=i,after=copy.deepcopy(m.state_dict()),policy=copy.deepcopy(p.state_dict()),optimizer=copy.deepcopy(o.state_dict())))
 for n,rr in [('tasks',rows),('blocks',blocks),('positions',pos)]:write_frame(dest/f'{n}.csv',rr)
 np.savez_compressed(dest/'inner_history.npz',inner=hist,steps=steps);torch.save(snapshots,dest/'adapted_states.pt');rr=dict(seed=seed,prior=prior,organization=org,condition=prior+'-'+org,arm=arm,dose=dose,run_id=rid,diagnostic_steps=count,tasks=12,directory=str(dest.relative_to(OUT)),source=str(src.relative_to(ROOT)),source_sha=sha(src));dump(dest/'complete.json',rr);return rr

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v1','v2'],default='v2');version=parser.parse_args().version;assert (OUT/'audit/meta_complete.json').exists();jobs=[(s,p,o,'Historical',0) for s in range(20) for p in ['P1','P2'] for o in ['I','C']]+[(s,p,o,a,d) for d in [240,960] for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']];jobs=[(*j,version) for j in jobs];rows=[];means=[];bm=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
  for row in pool.map(heldout_run,jobs):
   rows.append(row);d=OUT/row['directory'];f=pd.read_csv(d/'tasks.csv');r={k:row[k] for k in ['seed','condition','arm','dose']};means.append(dict(**r,**{n:f[n].astype(float).mean() for n in ['ce','accuracy','conflict_accuracy','appropriate_context_change','prefix_centered_separation']},n_valid=int(f.ce.notna().sum())))
   try:b=pd.read_csv(d/'blocks.csv')
   except pd.errors.EmptyDataError:b=pd.DataFrame()
   if len(b):
    for context,g in b.groupby('practiced'):bm.append(dict(**r,context=context,own_gain=g.own_gain.mean(),cross_gain=g.cross_gain.mean(),specificity=g.specificity.mean()))
 write_frame(OUT/f'registry/heldout_diagnostics_{version}.csv',rows);f=pd.DataFrame(means);write_frame(OUT/'tables/heldout_seed.csv',f);write_frame(OUT/'tables/heldout_summary.csv',summarize(f,['condition','arm','dose'],['ce','accuracy','conflict_accuracy','appropriate_context_change','prefix_centered_separation']));write_frame(OUT/'tables/heldout_block_seed.csv',bm);dump(OUT/f'audit/heldout_{version}_complete.json',dict(checkpoints=400,tasks_per_checkpoint=12,diagnostic_steps=sum(x['diagnostic_steps'] for x in rows),selection=False));append('## Heldoutdiagnosticscomplete\nFixed12unseenidentifiableconflicttasksperseedatall400initializations,20updates/task,freshAdam/policy,noouterupdate. Prefixposition3sameinputs/differenttargetsmakecontextchangefunctional;wholetrajectoryseparationaloneisnotcausalcontextuse. Signedblockown/crosseffectsandallquerypositionsrecorded. Average12taskswithinseedbefore10kCI;no diagnosticselection.')
if __name__=='__main__':main()
