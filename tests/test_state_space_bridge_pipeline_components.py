import numpy as np
import pandas as pd

from state_space_bridge.pipeline import _fit_traces
from state_space_bridge.recovery import run_state_space_recovery


def _average_trace(agent="human"):
    context = ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10
    return pd.DataFrame({
        "agent_type": agent,
        "dataset": "Synthetic",
        "condition": "Synthetic",
        "counterbalance_group": "AFirst",
        "seed_or_group": "AVERAGE",
        "fit_scope": "pre_probe_AB" if agent == "human" else "all_AB_training",
        "global_step": range(1, 41),
        "block_id": np.repeat(range(1, 5), 10),
        "analysis_sequence_in_block": np.tile(range(1, 11), 4),
        "context": context,
        "observed": np.linspace(1.0, 0.4, 40),
        "observed_sem": 0.02,
        "participants": 10,
    })


def test_shared_trace_fitter_exports_complete_long_tables():
    tables, lookup = _fit_traces(_average_trace())
    assert set(tables) == {
        "state_space_model_fits", "state_space_bic_comparison",
        "state_space_parameters", "state_space_decoders",
        "state_space_predictions", "state_space_latent_states",
    }
    assert set(tables["state_space_model_fits"].model) == {
        "SS_LUMP", "SS_CROSS", "SS_SPLIT"
    }
    assert len(tables["state_space_predictions"]) == 3 * 40
    assert tables["state_space_model_fits"].probe_rows_used.eq(0).all()
    assert len(lookup) == 1


def test_recovery_preserves_three_by_three_comparison_and_schedule():
    context = np.array(["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10)
    result = run_state_space_recovery(context, replicates=2, noise_sd=.01, random_seed=1)
    assert len(result) == 3 * 2 * 3
    assert set(result.generating_model) == {"SS_LUMP", "SS_CROSS", "SS_SPLIT"}
    assert set(result.fitted_model) == {"SS_LUMP", "SS_CROSS", "SS_SPLIT"}
    assert result.recovery_replicates_per_generating_model.eq(2).all()
