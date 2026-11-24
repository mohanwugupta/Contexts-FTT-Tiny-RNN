import pytest
from experiments.high_scale_constrained_meta import figures

def test_sealed_figures_need_external_destination(tmp_path,monkeypatch):
 root=tmp_path/'sealed';root.mkdir();(root/'manifest.json').write_text('{}');monkeypatch.setattr(figures,'OUT',root)
 with pytest.raises(ValueError):figures.destination(None)
 assert figures.destination(tmp_path/'outside')==tmp_path/'outside'
