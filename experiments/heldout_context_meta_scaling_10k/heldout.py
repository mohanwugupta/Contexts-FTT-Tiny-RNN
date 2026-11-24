"""Reserved conflict capability: exact source adaptation plus missing wrong-cue traces."""
import concurrent.futures,multiprocessing
from .common import *
from .adapters import heldout_job
from .analysis import summarize,scaling_effects
from experiments.heldout_context_meta.heldout import conflict_cells
DIAGNOSTIC_DOSES=[960,3840,7680,10000]

def recovered_correct_before(final,blocks,context):
 return float(final+blocks.loc[blocks.practiced==context,'own_gain'].sum()+blocks.loc[blocks.other==context,'cross_gain'].sum())

def wrong_context(row):
 """Correct final branches reused; reconstruct their initial CE; evaluate missing wrong branches."""
 initialize_process();dest=OUT/'conflict_wrong_cue'/row['run_id'];dest.mkdir(parents=True,exist_ok=True);done=dest/'complete.json'
 if done.exists():return json.loads(done.read_text())
 d=ROOT/row['directory'] if row['reused'] else OUT/row['directory'];state,source_path=initial(row['seed'],row['prior'],row['organization'],row['arm'],row['dose']);m0=source_model(state);after=torch.load(d/'adapted_states.pt',weights_only=False);models={r['task']:r['after'] for r in after};eps=json.loads((OUT/f'registry/heldout_v2_seed_{row["seed"]}.json').read_text())['episodes'];correct=pd.read_csv(d/'positions.csv',float_precision='round_trip');blocks=pd.read_csv(d/'blocks.csv',float_precision='round_trip');rows=[];pos=[];calls=0
 for task,ep in enumerate(eps):
  if task not in models:
   rows.append(dict(task=task,status='blocked_adaptation',correct_CE_before=np.nan,correct_CE_after=np.nan,wrong_CE_before=np.nan,wrong_CE_after=np.nan,correct_gain=np.nan,wrong_gain=np.nan,context_benefit=np.nan));continue
  model=source_model(models[task]);cells=conflict_cells(ep)
  for context,sequence in cells:
   wrong=next(c for c,j in cells if c!=context);cc=correct[(correct.task==task)&(correct.context==context)&(correct.sequence==sequence)].sort_values('position');bb=blocks[blocks.task==task];assert len(cc)==7 and cc.position.tolist()==list(range(1,8));final=float(cc.ce.mean());before=recovered_correct_before(final,bb,context);wrongobs={stage:core.trace(m,wrong,ep['paths'][sequence]) for stage,m in [('before',m0),('after',model)]};calls+=2
   wb=float(wrongobs['before']['ce'].mean());wa=float(wrongobs['after']['ce'].mean());cb=float(cc.loc[cc.position==3,'ce'].iloc[0]);wc=float(wrongobs['after']['ce'][2]);rows.append(dict(task=task,status='completed',context=context,wrong_context=wrong,sequence=sequence,correct_CE_before=before,correct_CE_after=final,wrong_CE_before=wb,wrong_CE_after=wa,correct_gain=before-final,wrong_gain=wb-wa,context_benefit=wa-final,context_gain_advantage=(before-final)-(wb-wa),conflict_correct_ce=cb,conflict_wrong_ce=wc,conflict_context_benefit=wc-cb))
   for stage,x in wrongobs.items():
    for t in range(7):pos.append(dict(task=task,stage=stage,sequence=sequence,correct_context=context,evaluation_context=wrong,position=t+1,target=ep['paths'][sequence][t+1],ce=x['ce'][t],correct=bool(x['accuracy'][t]),margin=x['margin'][t],**{f'logit_{k}':x['logits'][t,k] for k in range(4)},**{f'probability_{k}':x['probabilities'][t,k] for k in range(4)}))
 write_frame(dest/'cells.csv',rows);write_frame(dest/'positions.csv.gz',pos);receipt={k:row[k] for k in ['seed','prior','organization','condition','arm','dose','run_id']};receipt.update(directory=str(dest.relative_to(OUT)),source_sha=sha(source_path),adapted_states_sha=sha(d/'adapted_states.pt'),forward_calls=calls,optimizer_updates=0);dump(done,receipt);return receipt

def main():
 assert (OUT/'audit/meta_complete.json').exists();jobs=[(s,p,o,a,d) for d in [3840,7680,10000] for s in range(20) for p in ['P1','P2'] for o in ['I','C'] for a in ['Seen','Transfer']];new=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(heldout_job,jobs):r['reused']=False;new.append(r);print(f'Conflict diagnostics {len(new)}/480',flush=True)
 old=pd.read_csv(SOURCE/'registry/heldout_diagnostics_v2.csv');old=old[old.dose.isin([0,960])].copy();old['directory']=[str((SOURCE/d).relative_to(ROOT)) for d in old.directory];old['reused']=True;r=pd.concat([old,pd.DataFrame(new)],ignore_index=True);assert len(r)==720;write_frame(OUT/'registry/heldout_diagnostics_v2.csv',r);means=[];bm=[]
 for row in r.to_dict('records'):
  d=ROOT/row['directory'] if row['reused'] else OUT/row['directory'];tasks=pd.read_csv(d/'tasks.csv');blocks=pd.read_csv(d/'blocks.csv');positions=pd.read_csv(d/'positions.csv');tag={k:row[k] for k in ['seed','condition','arm','dose']};m={n:tasks[n].astype(float).mean() for n in ['ce','accuracy','conflict_accuracy','appropriate_context_change','prefix_centered_separation']};m['conflict_ce']=positions[positions.position==3].ce.mean();means.append(dict(**tag,**m,n_valid=int(tasks.ce.notna().sum())))
  for context,g in blocks.groupby('practiced'):bm.append(dict(**tag,context=context,own_gain=g.own_gain.mean(),cross_gain=g.cross_gain.mean(),specificity=g.specificity.mean()))
 f=pd.DataFrame(means);write_frame(OUT/'tables/heldout_seed.csv',f);write_frame(OUT/'tables/heldout_summary.csv',summarize(f,['condition','arm','dose'],['ce','accuracy','conflict_accuracy','conflict_ce','appropriate_context_change','prefix_centered_separation']));write_frame(OUT/'tables/heldout_block_seed.csv',bm)
 with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  wrong=list(pool.map(wrong_context,r.to_dict('records')))
 write_frame(OUT/'registry/conflict_wrong_cue.csv',wrong);wr=[]
 for row in wrong:
  x=pd.read_csv(OUT/row['directory']/'cells.csv');wr.append({k:row[k] for k in ['seed','condition','arm','dose']}|{k:x[k].mean() for k in ['correct_CE_before','correct_CE_after','wrong_CE_before','wrong_CE_after','correct_gain','wrong_gain','context_benefit','context_gain_advantage','conflict_correct_ce','conflict_wrong_ce','conflict_context_benefit']})
 wf=pd.DataFrame(wr);write_frame(OUT/'tables/heldout_wrong_seed.csv',wf);write_frame(OUT/'tables/heldout_wrong_summary.csv',summarize(wf,['condition','arm','dose'],list(wf.columns[4:])));fx=scaling_effects(f.assign(probe='conflict'),['probe'],['ce','accuracy','conflict_accuracy','conflict_ce','appropriate_context_change','prefix_centered_separation']);write_frame(OUT/'tables/heldout_seed_effects.csv',fx);write_frame(OUT/'tables/heldout_paired_intervals.csv',summarize(fx,['condition','dose','contrast','probe','metric'],['effect']));dump(OUT/'audit/heldout_complete.json',dict(passed=True,checkpoints=720,reused=240,new=480,tasks_per_checkpoint=12,diagnostic_steps=sum(x['diagnostic_steps'] for x in new),wrong_cue_forward_calls=sum(x['forward_calls'] for x in wrong),wrong_cue_optimizer_updates=0,library='corrected_v2_reserved_through10000',selection=False));append('## Conflict capability complete\nCorrected-v2 tasks reservedthroughall10000episodes. Historical/960 diagnosticcopiesreused,480 newcopies at3840/7680/10000,exact20innerupdates/task. Missingwrong-contextnativebranchesindexedseparately;correctbranchesreused,initialcorrectCEreconstructedfromfoursignedblockgains. Whole12tasksaveragedwithinseedbefore20seedCI;noneusedforselection.')
if __name__=='__main__':main()
