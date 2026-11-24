import pytest
from experiments.heldout_context_meta.seal import check_hashes
from experiments.heldout_context_meta.common import sha

def test_seal_rejects_modified_or_escaping_source(tmp_path):
 p=tmp_path/'x';p.write_text('a');h={'x':sha(p)};check_hashes(tmp_path,h);p.write_text('b')
 with pytest.raises(ValueError):check_hashes(tmp_path,h)
 with pytest.raises(ValueError):check_hashes(tmp_path,{'../outside':'bad'})
