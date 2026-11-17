import pandas as pd
import pytest

from ftt_srn_context_sequence_interaction.models.execution_models import (
    fit_execution_models,
    leave_one_seed_out_splits,
)


def _natural_rows():
    rows = []
    for seed in (0, 1, 2):
        for position in (1, 2, 3):
            for context, z in (("A", 1.0), ("B", -1.0)):
                rows.append({
                    "seed": seed, "checkpoint": "human_matched",
                    "context": context, "token_position": position,
                    "z_context": z,
                    "execution_speed": 0.2 + 0.1 * position + z * 0.02 * position,
                })
    return pd.DataFrame(rows)


def test_leave_one_seed_out_never_leaks_held_out_seed():
    rows = _natural_rows()
    for held_out, training, test in leave_one_seed_out_splits(rows):
        assert held_out not in set(training.seed)
        assert set(test.seed) == {held_out}


def test_models_fit_only_natural_ab_and_interaction_captures_token_dependence():
    result = fit_execution_models(_natural_rows())
    comparison = result.comparison.set_index("model")
    assert comparison.loc["interaction", "mean_held_out_mse"] < comparison.loc[
        "additive", "mean_held_out_mse"
    ]
    assert set(result.predictions.context) == {"A", "B"}
    assert result.predictions.training_contexts.eq("A|B").all()


def test_model_fitting_rejects_mixed_or_novel_outcomes():
    invalid = pd.concat([
        _natural_rows(),
        pd.DataFrame([{
            "seed": 0, "checkpoint": "human_matched", "context": "Mixed_norm",
            "token_position": 1, "z_context": 0.0, "execution_speed": 0.5,
        }]),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="A/B"):
        fit_execution_models(invalid)

