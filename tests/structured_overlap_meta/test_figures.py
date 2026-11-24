import pytest
from experiments.structured_overlap_meta import figures

def test_sealed_regeneration_requires_external_destination(tmp_path,monkeypatch):
 root=tmp_path/'sealed';root.mkdir();(root/'manifest.json').write_text('{}');monkeypatch.setattr(figures,'OUT',root)
 with pytest.raises(ValueError):figures.safe_destination(None)
 with pytest.raises(ValueError):figures.safe_destination(root/'figures')
 assert figures.safe_destination(tmp_path/'regen')==tmp_path/'regen'
