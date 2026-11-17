import numpy as np
import pandas as pd

from probe_modeling.metrics import add_mixture_indices


def test_mixture_index_masks_ill_conditioned_ab_counterfactuals():
    data = pd.DataFrame(
        {
            "mean_rt": [500.0, np.sqrt(450.0 * 550.0)],
            "log_rt": np.log([500.0, np.sqrt(450.0 * 550.0)]),
            "pred_A_rt": [499.0, 450.0],
            "pred_B_rt": [501.0, 550.0],
            "pred_A_log_rt": np.log([499.0, 450.0]),
            "pred_B_log_rt": np.log([501.0, 550.0]),
        }
    )
    out = add_mixture_indices(data)
    assert not out.loc[0, "mixture_index_reliable"]
    assert np.isnan(out.loc[0, "mixture_index_log"])
    assert out.loc[1, "mixture_index_reliable"]
    assert out.loc[1, "mixture_index_log"] == 0.5
