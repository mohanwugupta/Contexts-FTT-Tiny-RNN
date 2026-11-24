import pytest
from experiments.meta_learning_scaling.seal import validate_hash_map
from experiments.meta_learning_scaling.common import sha

def test_manifest_tamper_and_escape_rejected(tmp_path):
 p=tmp_path/'x';p.write_text('one');h={'x':sha(p)};validate_hash_map(tmp_path,h);p.write_text('two')
 with pytest.raises(ValueError):validate_hash_map(tmp_path,h)
 with pytest.raises(ValueError):validate_hash_map(tmp_path,{'../outside':'fake'})
