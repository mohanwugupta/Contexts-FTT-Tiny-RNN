import pytest
from experiments.ab_update_transfer_analysis import storage, sources

def test_budget_reserves_failed_attempts_and_refuses_excess(tmp_path):
 ledger=storage.Ledger(tmp_path/'events.jsonl',limits={'diagnostic':2,'reconstruction':1,'fixture':0})
 ledger.reserve('diagnostic','a');ledger.reserve('diagnostic','b')
 assert ledger.counts()['diagnostic']==2
 with pytest.raises(RuntimeError):ledger.reserve('diagnostic','c')
 with pytest.raises(RuntimeError):ledger.reserve('fixture','fixture')
 assert storage.Ledger(tmp_path/'events.jsonl',limits=ledger.limits).counts()['diagnostic']==2

def test_exact_branch_key_and_resume_contract():
 s=dict(model={'x':1},policy={'p':2},optimizer={'step':10},rng={'torch':'archived'},cursor=10)
 a=storage.branch_key(s,{'code':'a'},'A',10,'native')
 assert a==storage.branch_key(s,{'code':'a'},'A',10,'native')
 for key in s:
  bad=dict(s);bad[key]='wrong'
  assert a!=storage.branch_key(bad,{'code':'a'},'A',10,'native')
 assert a!=storage.branch_key(s,{'code':'b'},'A',10,'native')
 assert a!=storage.branch_key(s,{'code':'a'},'B',10,'native')

def test_manifest_resolves_unique_physical_cohort_without_aliases():
 rows=sources.selected_registry();assert len(rows)==342
 assert len(rows[rows.cohort.eq('final')])==144
 assert len(rows[rows.meta_stage.eq('fresh')])==18
 assert len(rows[rows.meta_stage.eq('reptile')])==36
 assert rows.trajectory_id.is_unique and rows.path.is_unique

def test_derived_csv_receipts_are_resumable_but_cannot_change(tmp_path):
 from experiments.ab_update_transfer_analysis.common import csv
 p=tmp_path/'metrics.csv';csv(p,[dict(gain=-2.)]);before=p.read_bytes()
 csv(p,[dict(gain=-2.)]);assert p.read_bytes()==before
 with pytest.raises(AssertionError):csv(p,[dict(gain=2.)])
