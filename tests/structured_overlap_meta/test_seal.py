import hashlib,pytest
from experiments.structured_overlap_meta.seal import check_hashes

def test_hash_seal_rejects_mutation_missing_escape(tmp_path):
 p=tmp_path/'a';p.write_text('fixed');h=hashlib.sha256(p.read_bytes()).hexdigest();check_hashes(tmp_path,{'a':h});p.write_text('changed')
 for values in [{'a':h},{'missing':h},{'../a':h}]:
  with pytest.raises((ValueError,FileNotFoundError)):check_hashes(tmp_path,values)
