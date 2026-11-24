import hashlib,pytest
from experiments.linear_scale_sweep.figures import destination
from experiments.linear_scale_sweep.seal import check_hashes

def test_hash_tamper_escape(tmp_path):
 p=tmp_path/'a';p.write_text('x');h=hashlib.sha256(p.read_bytes()).hexdigest();check_hashes(tmp_path,{'a':h});p.write_text('y')
 with pytest.raises(ValueError):check_hashes(tmp_path,{'a':h})
 with pytest.raises(ValueError):check_hashes(tmp_path,{'../a':h})

def test_legacy_control_guard_detects_exact_seed18():
 import torch
 from experiments.linear_scale_sweep.common import OUT
 from experiments.linear_scale_sweep.reuse_audit import matches_older_control
 a=torch.load(OUT/'initial/s18_Fresh_2.pt',weights_only=False)['model'];b=torch.load(OUT/'initial/s18_Fresh_4.pt',weights_only=False)['model'];assert matches_older_control(18,a);assert not matches_older_control(18,b)
