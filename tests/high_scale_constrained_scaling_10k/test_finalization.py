import numpy as np,pandas as pd,pytest
from experiments.high_scale_constrained_scaling_10k import finalize
from experiments.high_scale_constrained_scaling_10k.budget import plan

def test_exact_budget_and_no_recount_of_C_prefix():
 p=plan();rows=[]
 for c in ['U','P']:
  for arm in ['T','B','O1','O2']:
   for s in range(6):
    for k in [4,5,6]:rows.append(dict(condition=c,arm=arm,new_episodes_since_source=8080 if c=='U' else 10000,new_inner_updates=(8080 if c=='U' else 10000)*20,new_outer_updates=(8080 if c=='U' else 10000)//(2 if arm=='B' else 1),new_interpolations=0))
 for _ in range(18):rows.append(dict(condition='P',arm='R',new_episodes_since_source=1920,new_inner_updates=38400,new_outer_updates=0,new_interpolations=1920))
 got=finalize.meta_counts(pd.DataFrame(rows));assert got['new_meta_episodes']==p['new_meta_episodes'] and got['new_inner_updates']==p['new_inner_updates'] and got['new_outer_Adam_updates']==p['new_outer_Adam_updates']
 with pytest.raises(AssertionError):finalize.check_meta_counts(pd.DataFrame(rows+rows[:1]))
 assert finalize.FIXTURE_OPTIMIZER_UPDATES==1984

def test_readonly_receipt_resume_cannot_replace_a_different_budget(tmp_path):
 from experiments.high_scale_constrained_scaling_10k.common import sha
 p=tmp_path/'receipt.json';finalize.receipt(p,dict(passed=True,updates=17));h=sha(p)
 finalize.receipt(p,dict(passed=True,updates=17));assert sha(p)==h
 with pytest.raises(AssertionError):finalize.receipt(p,dict(passed=True,updates=18))
 assert sha(p)==h

def test_csv_receipts_preserve_identical_and_reject_changed(tmp_path):
 p=tmp_path/'derived.csv';f=pd.DataFrame([dict(seed=0,value=1.25)])
 finalize.frame_receipt(p,f);original=p.read_bytes()
 finalize.frame_receipt(p,f);assert p.read_bytes()==original
 with pytest.raises(AssertionError):finalize.frame_receipt(p,f.assign(value=-1.25))
 assert p.read_bytes()==original

def test_ledger_pending_hash_is_versioned_and_other_claims_unchanged(tmp_path):
 from experiments.high_scale_constrained_scaling_10k.common import sha
 (tmp_path/'audit').mkdir();source=tmp_path/'exposure.csv';source.write_text('actual,17\n')
 ledger=pd.DataFrame([dict(claim='budget',source='exposure.csv',sha256='pending_finalization',mean=17),dict(claim='fixed',source='exposure.csv',sha256=sha(source),mean=17)])
 p=tmp_path/'CLAIM_LEDGER.csv';ledger.to_csv(p,index=False);original=p.read_bytes()
 finalize.finish_ledger(tmp_path)
 assert (tmp_path/'audit/CLAIM_LEDGER_initial.csv').read_bytes()==original
 result=pd.read_csv(p);assert result.sha256.tolist()==[sha(source)]*2
 accepted=p.read_bytes();finalize.finish_ledger(tmp_path);assert p.read_bytes()==accepted
 source.write_text('actual,18\n')
 with pytest.raises(AssertionError):finalize.finish_ledger(tmp_path)
 assert p.read_bytes()==accepted
