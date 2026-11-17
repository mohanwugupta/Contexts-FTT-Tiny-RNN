import pytest
from experiments.context_failure_selective_repair.decision import recommendation, validate_claim


def test_negative_mechanism_no_training():
 r=recommendation({'selective_mechanism_gate':False});assert r['recommendation']=='no_new_training' and r['scientific_updates']==0
 assert 'diagnostic' in r and r['design_only'] is True

@pytest.mark.parametrize('claim',['natural retrieval','lost sequence information','global unreachability','human mechanism','attractor'])
def test_unsupported_labels_rejected(claim):
 with pytest.raises(ValueError):validate_claim({'asserts':[claim]})
 assert validate_claim({'asserts':['finite teacher-forced causal cue effect']})
