from srn_environmental_pressures.config import (
    EnvironmentalCondition,
    discovery_conditions,
)
from srn_environmental_pressures.confirmation.seed_registry import validate_fresh_seeds


def test_primary_discovery_grid_has_120_unique_environmental_conditions():
    grid = discovery_conditions()
    assert len(grid) == 120
    assert len({condition.condition_id for condition in grid}) == 120
    assert {condition.context_validity for condition in grid} == {1.0, .95, .85, .75, .65, .55}
    assert {condition.cue_accuracy for condition in grid} == {.97, .90, .80, .70, .60}
    assert {condition.state_separation for condition in grid} == {.20, .40, .60, .80}
    assert {condition.state_variant for condition in grid} == {"fixed_global"}


def test_discovery_and_confirmation_seeds_cannot_overlap():
    validate_fresh_seeds((0, 1, 2, 3, 4), tuple(range(1000, 1020)))
    try:
        validate_fresh_seeds((0, 1, 2), (2, 3))
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping confirmation seeds must fail")


def test_condition_identity_includes_only_environmental_dimensions():
    condition = EnvironmentalCondition(.75, .8, .6, .01, "fixed_global")
    assert "v0.75" in condition.condition_id
    assert "cue0.80" in condition.condition_id
    assert "d0.60" in condition.condition_id
    assert "h" not in condition.condition_id
    assert "epoch" not in condition.condition_id

