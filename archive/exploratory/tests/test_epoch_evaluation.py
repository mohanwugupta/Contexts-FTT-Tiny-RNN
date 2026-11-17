from probe_modeling.epoch_data import normalize_epoch_data
from probe_modeling.epoch_fit import epoch_region_labels


def test_v2_evaluation_requires_every_epoch(epoch_keypress_frame):
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)
    regions = set(epoch_region_labels(clean))
    assert regions == {
        "mixed_1", "mixed_2", "post_mixed_return", "mixed_test", "ordinary_AB"
    }

