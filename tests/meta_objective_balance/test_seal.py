import hashlib
import pytest
from experiments.meta_objective_balance.seal import check_hashes

def test_seal_rejects_tampering_missing_and_path_escape(tmp_path):
 p=tmp_path/'a';p.write_text('frozen');h=hashlib.sha256(p.read_bytes()).hexdigest();check_hashes(tmp_path,{'a':h});p.write_text('changed')
 with pytest.raises(ValueError):check_hashes(tmp_path,{'a':h})
 with pytest.raises(ValueError):check_hashes(tmp_path,{'../a':h})
