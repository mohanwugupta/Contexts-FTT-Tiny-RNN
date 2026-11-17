from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from context_sequence.provenance.hashing import sha256
from experiments.context_failure_selective_repair.reporting import audit_tables, products
from experiments.context_failure_selective_repair.figures import generate
from experiments.context_failure_selective_repair.verify import verify_outputs, safe_destination
from experiments.context_failure_selective_repair.core import ROOT
OUT=ROOT/'results/context_failure_selective_repair/v1_1'


def test_exact_tables_and_no_transition_bootstrap():
 q=audit_tables(OUT);assert q['r1_cells']==360 and q['seed_unit']=='seed' and q['all20_retained']
 assert q['matched_k3_seed_count']==1 and q['reverse_valid_cells']==0


def test_five_figures_deterministic_table_only(tmp_path,monkeypatch):
 import experiments.context_failure_selective_repair.core as core
 monkeypatch.setattr(core,'load_original',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No scientific model calls in regeneration')))
 for name in ['a','b']:
  p=tmp_path/name;p.mkdir();generate(OUT,p)
 files=sorted((tmp_path/'a').glob('*'));assert len(list((tmp_path/'a').glob('*.png')))==5
 assert len(files)==10
 assert all(sha256(p)==sha256(tmp_path/'b'/p.name) for p in files)
 with pytest.raises(FileExistsError):generate(OUT,tmp_path/'a')


def test_seal_detects_extra_missing_and_modified_science(tmp_path):
 p=tmp_path/'x.csv';p.write_text('x\n1\n');m={'x.csv':sha256(p)};assert verify_outputs(tmp_path,m)
 extra=tmp_path/'extra.csv';extra.write_text('unregistered');
 with pytest.raises(ValueError):verify_outputs(tmp_path,m)
 extra.unlink();p.write_text('changed')
 with pytest.raises(ValueError):verify_outputs(tmp_path,m)
 p.unlink()
 with pytest.raises(ValueError):verify_outputs(tmp_path,m)
 with pytest.raises(ValueError):safe_destination(ROOT/'results/output_compatible_trajectories/v1_20261111')


def test_claim_sources_and_regeneration_products(tmp_path):
 q=products(OUT,tmp_path,figures=False)
 assert 'FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md' in q
 text=(tmp_path/'FINAL_CONTEXT_FAILURE_SELECTIVE_REPAIR_REPORT.md').read_text()
 assert 'did not identify a unique disruption' in text and 'no new training' in text.lower()
 c=pd.read_csv(tmp_path/'tables/r5_claim_sources.csv')
 for row in c.itertuples():assert sha256(OUT/row.source_table)==row.source_sha256
 assert c.claim_id.nunique()>=6
