from pathlib import Path
import pytest
from experiments.context_experience_prior.seal import verify_hashes
from experiments.context_experience_prior.core import sha

def test_seal_detects_tamper_and_escaping_paths(tmp_path):
 p=tmp_path/'x';p.write_text('one');d={'x':sha(p)};verify_hashes(tmp_path,d);p.write_text('two')
 with pytest.raises(ValueError):verify_hashes(tmp_path,d)
 with pytest.raises(ValueError):verify_hashes(tmp_path,{'../outside':'fake'})
