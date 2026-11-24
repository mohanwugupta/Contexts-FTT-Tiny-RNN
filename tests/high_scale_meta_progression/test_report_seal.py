import pytest
from experiments.high_scale_meta_progression.report import explain_protocol
from experiments.high_scale_meta_progression.seal import check_hashes
from experiments.high_scale_meta_progression.common import sha

def test_report_protocol_and_hash_guard(tmp_path):
 s=explain_protocol();assert all(w in s for w in ['960','1,920','990','C column','not causal mediation','0.75','0.09375','0.0625'])
 p=tmp_path/'x';p.write_text('a');check_hashes(tmp_path,{'x':sha(p)});p.write_text('b')
 with pytest.raises(ValueError):check_hashes(tmp_path,{'x':'wrong'})
 with pytest.raises(ValueError):check_hashes(tmp_path,{'../x':'wrong'})
