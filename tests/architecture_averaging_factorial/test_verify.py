import pytest
from experiments.architecture_averaging_factorial.verify import check_output_files
from experiments.architecture_averaging_factorial.protocol import hash_file


def test_detect_changed_missing_and_extra_evidence(tmp_path):
 p=tmp_path/'results.csv';p.write_text('x\n1\n');m={'results.csv':hash_file(p)};assert check_output_files(tmp_path,m)
 e=tmp_path/'extra.csv';e.write_text('extra')
 with pytest.raises(ValueError):check_output_files(tmp_path,m)
 e.unlink();p.write_text('changed')
 with pytest.raises(ValueError):check_output_files(tmp_path,m)
 p.unlink()
 with pytest.raises(ValueError):check_output_files(tmp_path,m)
