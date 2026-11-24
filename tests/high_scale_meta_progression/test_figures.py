import pytest
from experiments.high_scale_meta_progression.figures import destination,FIGURE_FAMILIES

def test_sealed_plots_need_external_destination(tmp_path):
 root=tmp_path/'sealed';root.mkdir();(root/'manifest.json').write_text('{}');external=tmp_path/'plots'
 with pytest.raises(ValueError):destination(root,None)
 with pytest.raises(ValueError):destination(root,root/'figures')
 assert destination(root,external)==external and len(FIGURE_FAMILIES)==6
