from pathlib import Path
import hashlib
import pytest
from experiments.heldout_context_meta_scaling_10k.seal import check_hashes

def test_changed_file_or_path_escape_rejected(tmp_path):
 p=tmp_path/'safe.txt';p.write_text('data');h=hashlib.sha256(p.read_bytes()).hexdigest();check_hashes(tmp_path,{'safe.txt':h});p.write_text('changed')
 with pytest.raises(ValueError):check_hashes(tmp_path,{'safe.txt':h})
 with pytest.raises(ValueError):check_hashes(tmp_path,{'../outside':'x'})
