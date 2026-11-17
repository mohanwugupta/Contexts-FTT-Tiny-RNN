from pathlib import Path
import numpy as np
import pytest
from experiments.output_compatible_trajectories.source_audit import load_sources, verify_checkpoint, require_parity
from experiments.output_compatible_trajectories.continuation import frozen_weight_guard
ROOT=Path(__file__).resolve().parents[2]


def test_T01_T02_T06_historical_identity_and_negative_fixture():
    bundle=load_sources(ROOT)
    assert len(bundle['records'])==120 and len(bundle['analysis_records'])==100
    r=bundle['records'][0]
    if not (ROOT/r['path']).is_file():pytest.skip('blocked_missing_source: '+r['path'])
    m=verify_checkpoint(ROOT,r)
    with frozen_weight_guard(m):assert np.all(m.wih[:,6]==0)
    wrong=dict(r,path=bundle['records'][1]['path'])
    with pytest.raises(ValueError):verify_checkpoint(ROOT,wrong)
    assert require_parity(np.array([.1]),np.array([.1]),.002)
    with pytest.raises(ValueError):require_parity(np.array([.1]),np.array([.2]),.002)


def test_T05_manifest_and_duplicate_negative_fixtures(tmp_path):
    import json
    from experiments.output_compatible_trajectories.config import CONTRACT
    from experiments.output_compatible_trajectories.common import verify_stage,unique
    from context_sequence.provenance.hashing import sha256
    import pandas as pd
    out=tmp_path/'out';(out/'manifests').mkdir(parents=True)
    p=out/'data.csv';p.write_text('value\n1\n')
    (out/'manifests/p0.json').write_text(json.dumps({'integrity_passed':True,'contract_sha256':CONTRACT['contract_sha256'],'code_hashes':{},'outputs':{'data.csv':sha256(p)}}))
    verify_stage(tmp_path,out,'p0');p.write_text('value\n2\n')
    with pytest.raises(RuntimeError):verify_stage(tmp_path,out,'p0')
    with pytest.raises(ValueError):unique(pd.DataFrame({'seed':[0,0],'cue':['A','A']}),['seed','cue'])


def test_export_schema_cannot_silently_shift(tmp_path):
    import pandas as pd
    from experiments.output_compatible_trajectories.common import Stream
    (tmp_path/'tables').mkdir()
    s=Stream(tmp_path,'test');s.write(pd.DataFrame({'seed':[0],'ce':[.1]}))
    with pytest.raises(ValueError):s.write(pd.DataFrame({'seed':[1],'other_metric':[.2]}))


def test_optional_cloud_placeholder_is_explicit_without_fabricated_hash(tmp_path,monkeypatch):
    from experiments.output_compatible_trajectories import source_audit
    p=tmp_path/'placeholder.csv';p.write_text('not hydrated')
    def no_read(*a,**kw):raise AssertionError('Dataless historical source must not be read or fabricated')
    monkeypatch.setattr(source_audit,'sha256',no_read)
    row=source_audit.optional_table_audit(p,'expected-historical-hash',flags=0x40000000)
    assert row['status']=='blocked_missing_source_optional'
    assert row['observed_sha256'] is None and row['expected_sha256']=='expected-historical-hash'
