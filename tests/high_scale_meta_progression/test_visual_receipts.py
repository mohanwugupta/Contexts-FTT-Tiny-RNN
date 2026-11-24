import pytest
from experiments.high_scale_meta_progression.visual_review import validate_figure_sets,expected_names

def test_exact_figure_inventory_and_hashes_required():
 a={n:'hash' for n in expected_names()};assert len(a)==24 and validate_figure_sets(a,a)==24
 b=a.copy();b.pop(next(iter(b)))
 with pytest.raises(ValueError):validate_figure_sets(a,b)
 b=a.copy();b[next(iter(b))]='changed'
 with pytest.raises(ValueError):validate_figure_sets(a,b)
