import pytest
from experiments.linear_C_shared_contribution.verify import check_accounting

def test_wrong_loss_accounting_or_independent_rescue_claim_rejected():
 r=dict(ce_gain=10.,margin_loss_gain=9.,other_alternatives_loss_gain=1.,restored_retrieval_ce_difference=0.)
 assert check_accounting(r)
 with pytest.raises(AssertionError):check_accounting({**r,'margin_loss_gain':8.})
 with pytest.raises(AssertionError):check_accounting({**r,'restored_retrieval_ce_difference':.1})
