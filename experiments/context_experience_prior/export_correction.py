"""Recover recorded scalar rows using the original per-branch field order."""
import csv,shutil
from .core import *
from experiments.centered_context_linear.analysis import time_intervals
COMMON=['seed','organization','prior','condition','update','training_cue','archive_reused']
NEW_ORDER=['ce_before','accuracy_before','execution_loss','total_loss','gradient_norm','hidden_absmax','optimizer_step_executed','status','new_optimizer_step_executed']
OLD_ORDER=['archived_update_reused','trajectory_kind','status','optimizer_step_executed','ce_before','execution_loss','total_loss','accuracy_before','hidden_absmax','gradient_norm','amplitude_fraction_above_95','amplitude_fraction_above_99','new_optimizer_step_executed']
CANONICAL=COMMON+OLD_ORDER

def training_record(common,q):
 d=dict(**common,**q);return {n:d.get(n,np.nan) for n in CANONICAL}

def decode_training(path):
 rows=[]
 with Path(path).open() as f:
  r=csv.reader(f);header=next(r);assert header==CANONICAL
  for row in r:
   columns=CANONICAL if row[2]=='P0' else COMMON+NEW_ORDER;assert len(row)==len(columns);d=dict(zip(columns,row));rows.append(training_record({},d))
 f=pd.DataFrame(rows)
 for n in ['seed','update']:f[n]=pd.to_numeric(f[n]).astype(int)
 for n in ['ce_before','execution_loss','total_loss','accuracy_before','hidden_absmax','gradient_norm','amplitude_fraction_above_95','amplitude_fraction_above_99']:f[n]=pd.to_numeric(f[n])
 for n in ['archive_reused','optimizer_step_executed','new_optimizer_step_executed']:f[n]=f[n].map({'True':True,'False':False});assert f[n].notna().all()
 return f

def main():
 dest=OUT/'audit/export_correction';dest.mkdir();source=OUT/'tables/training.csv';old=dest/'training_original_misaligned.csv';shutil.copyfile(source,old);shutil.copyfile(OUT/'tables/training_intervals.csv',dest/'training_intervals_original_misaligned.csv');before=sha(source);f=decode_training(old);ev=pd.read_csv(OUT/'tables/evaluations.csv',float_precision='round_trip');ev=ev.rename(columns={'cue':'training_cue'});ev['update']+=1;z=f.merge(ev[['seed','condition','update','training_cue','ce']],on=['seed','condition','update','training_cue'],validate='one_to_one');assert len(z)==39600;err=abs(z.ce_before-z.ce);assert (err<=1e-4+1e-5*abs(z.ce_before)).all();np.testing.assert_allclose(z.total_loss,z.ce_before+.2*z.execution_loss,rtol=1e-6,atol=2e-6);assert f.new_optimizer_step_executed.sum()==26400 and f.archive_reused.sum()==13200 and f.status.eq('completed').all() and f.optimizer_step_executed.all();f.to_csv(source,index=False);time_intervals(f,['condition'],'update',['ce_before']).to_csv(OUT/'tables/training_intervals.csv',index=False);dump(dest/'receipt.json',dict(cause='append field order mismatch',original_sha=before,corrected_sha=sha(source),rows=39600,max_preupdate_CE_vs_frozen_error=float(err.max()),new_steps=26400,reused_steps=13200,training_rerun=False,frozen_tables_unchanged=True));append('## Export correction GREEN\nRetained originalmisaligned trainingCSV/intervals andfailedverification. Recovered39600 exactloggedscalarrecords byknownschemas;everyactualCE matches frozenpreupdateweights withinfixedtol,originaltotalobjectiveidentity passes,new/reusedcounts correct. Onlytrainingtable/intervals corrected,no scientificupdates/evaluationreplay. Strictappendcolumn guard anduniformtargetrecord helper preventrecurrence.')
if __name__=='__main__':main()
