import numpy as np
import pandas as pd

from retrieval_modeling.distillation import distill_retrieval_rule


def test_distillation_recovers_known_relative_and_history_rule():
    rng = np.random.default_rng(6)
    n = 400
    relative = rng.normal(size=n)
    evidence = rng.uniform(0.1, 1.5, n)
    hidden = rng.normal(size=n)
    logit = 1.7 * relative + 0.8 * hidden + rng.normal(0, 0.01, n)
    q = 1 / (1 + np.exp(-logit))
    data = pd.DataFrame(
        {
            "q_A": q, "relative_evidence_R": relative,
            "absolute_evidence_E": evidence, "history_state": hidden,
            "split": np.where(np.arange(n) % 5 == 0, "test", "train"),
        }
    )
    result = distill_retrieval_rule(data).iloc[0]
    assert result.beta_R > 0
    assert result.lambda_history > 0
    assert abs(result.beta_E) < 0.05
    assert result.heldout_r2 > 0.99
