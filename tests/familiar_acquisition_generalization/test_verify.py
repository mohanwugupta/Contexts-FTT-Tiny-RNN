import pytest
from experiments.familiar_acquisition_generalization.verify import check_outputs
from experiments.familiar_acquisition_generalization.audit import sha

def test_missing_modified_and_extra_scientific_bytes(tmp_path):
 p=tmp_path/'table.csv';p.write_text('a\n1\n');files={'table.csv':sha(p)};assert check_outputs(tmp_path,files)
 extra=tmp_path/'extra.csv';extra.write_text('extra')
 with pytest.raises(ValueError):check_outputs(tmp_path,files)
 extra.unlink();p.write_text('changed')
 with pytest.raises(ValueError):check_outputs(tmp_path,files)
 p.unlink()
 with pytest.raises(ValueError):check_outputs(tmp_path,files)
