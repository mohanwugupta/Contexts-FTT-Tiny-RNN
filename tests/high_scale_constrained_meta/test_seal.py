import hashlib,pytest
from experiments.high_scale_constrained_meta.seal import check_hashes

def test_tamper_missing_and_escape_rejected(tmp_path):
 p=tmp_path/'fixed';p.write_text('fixed');h=hashlib.sha256(p.read_bytes()).hexdigest();check_hashes(tmp_path,{'fixed':h});p.write_text('changed')
 for r in [{'fixed':h},{'missing':h},{'../fixed':h}]:
  with pytest.raises((ValueError,FileNotFoundError)):check_hashes(tmp_path,r)
