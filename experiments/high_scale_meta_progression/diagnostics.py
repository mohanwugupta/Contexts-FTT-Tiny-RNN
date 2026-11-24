"""Original corrected-v2 unseen conflicts, common parent once, no feedback."""
import concurrent.futures,multiprocessing
from .common import *
from experiments.heldout_context_meta import heldout as native
from experiments.heldout_context_meta_scaling_10k import heldout as wrong
from experiments.heldout_context_meta_scaling_10k.adapters import bind

def jobs():return [(s,k,a,d) for s in range(6) for k in [4,5,6] for a in ['R','T','B','O1','O2'] for d in ([1920] if a=='R' else [240,960,1920])]
def validate_reserved_tasks(eps):
 if len(eps)!=12:raise ValueError('Twelve fixed tasks required')
 for e in eps:
  if e['kind']!='conflict' or len(e['paths'])!=2 or e['paths'][0][:3]!=e['paths'][1][:3] or e['paths'][0][3]==e['paths'][1][3]:raise ValueError('Invalid original critical-position target labels')
 return True

def one(job):
 s,k,a,d=job;configure();eps=json.loads((OUT/f'registry/heldout_v2_seed_{s}.json').read_text())['episodes'];validate_reserved_tasks(eps)
 with bind(native,OUT=OUT,initial=initial,initialize_process=configure):r=native.heldout_run((s,'P2','I',f'{k}-{a}',d,'v2'))
 r['reused']=False
 with bind(wrong,OUT=OUT,initial=initial,initialize_process=configure):w=wrong.wrong_context(r)
 folder=OUT/r['directory'];task=pd.read_csv(folder/'tasks.csv',float_precision='round_trip');pos=pd.read_csv(folder/'positions.csv',float_precision='round_trip');block=pd.read_csv(folder/'blocks.csv',float_precision='round_trip');wc=pd.read_csv(OUT/w['directory']/'cells.csv',float_precision='round_trip');assert len(task)==12 and len(pos)==168 and len(block)==48 and len(wc)==24 and task.status.eq('completed').all();assert r['diagnostic_steps']==240 and w['optimizer_updates']==0
 row=dict(seed=s,scale=k,phase=a,dose=d,**{n:task[n].astype(float).mean() for n in ['ce','accuracy','conflict_accuracy','appropriate_context_change','prefix_centered_separation']},critical_ce=pos[pos.position.eq(3)].ce.mean(),n_tasks=12,n_appropriate_tasks=int(task.appropriate_context_change.astype(bool).sum()),own_gain=block.own_gain.mean(),cross_gain=block.cross_gain.mean(),specificity=block.specificity.mean(),**{n:wc[n].mean() for n in ['context_benefit','context_gain_advantage','conflict_context_benefit','correct_CE_before','correct_CE_after','wrong_CE_before','wrong_CE_after']});r.update(scale=k,phase=a);return r,w,row

def main():
 if (OUT/'audit/diagnostics_complete.json').exists():return
 assert json.loads((OUT/'audit/fomaml_verified.json').read_text())['passed'];assert '1 passed' in (OUT/'audit/30_GREEN_diagnostics.log').read_text();rows=[];wr=[];means=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r,w,m in pool.map(one,jobs()):rows.append(r);wr.append(w);means.append(m);print('Reserved conflicts',len(rows),'/234',flush=True)
 write_frame(OUT/'registry/diagnostics.csv',rows);write_frame(OUT/'registry/wrong_cue.csv',wr);(OUT/'tables').mkdir(exist_ok=True);write_frame(OUT/'tables/conflict_seed.csv',means);dump(OUT/'audit/diagnostics_complete.json',dict(passed=True,cells=234,tasks_per_cell=12,adaptation_updates=56160,wrong_cue_optimizer_updates=0,source_critical_position=3,bootstrap_unit='trained seed'));append('## Corrected-v2 conflict diagnostics complete\nAll234 common-parent/four-arm registered-dose cases,12 fixed unseen tasks each,20 native updates/task. Correct/wrong physical cues, all positions and labels preserved. Average tasks within seed; no meta feedback, new task fitting or seed replacement. FOMAML+0 is one common parent, not four independent diagnostics.');print('Conflict diagnostics PASSED',flush=True)
if __name__=='__main__':main()
