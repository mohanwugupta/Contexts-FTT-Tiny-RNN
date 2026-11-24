from experiments.meta_objective_balance.figures import MAIN,confirmation_cases

def test_registered_figure_families_and_no_selected_intermediate():
 assert MAIN==['figure_1','figure_2','figure_3','figure_4','figure_5']
 assert confirmation_cases(['A','F'])==[(960,'A'),(960,'F'),(960,'C'),(1920,'A'),(1920,'F'),(1920,'C')]

def test_sealed_figures_require_external_destination(tmp_path,monkeypatch):
 import pytest
 from experiments.meta_objective_balance import figures
 monkeypatch.setattr(figures,'OUT',tmp_path/'sealed');figures.OUT.mkdir();(figures.OUT/'manifest.json').write_text('{}')
 with pytest.raises(ValueError):figures.safe_destination(None)
 assert figures.safe_destination(tmp_path/'recreated')==tmp_path/'recreated'
