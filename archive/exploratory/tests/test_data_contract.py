import pandas as pd
import pytest

from probe_modeling.data import (
    EmptyDatasetError,
    UnknownContextLabelError,
    normalize_probe_data,
)


def test_loader_aggregates_and_normalizes_probe_schedule(keypress_probe_frame):
    out = normalize_probe_data(keypress_probe_frame, dataset="Combined", cohort_only=False)

    assert list(out.columns[:16]) == [
        "participant_id",
        "dataset",
        "block_id",
        "sequence_in_block",
        "global_sequence",
        "context",
        "probe_type",
        "is_probe",
        "is_primary_probe",
        "probe_block_number",
        "previous_context",
        "next_context",
        "mean_rt",
        "log_rt",
        "error",
        "break_duration",
    ]
    assert len(out) == 2 * 6 * 2
    assert set(out.loc[out.is_probe, "probe_type"]) == {"Both"}
    assert set(out.loc[out.is_probe, "probe_block_number"].astype(int)) == {1, 2}
    assert set(out.loc[out.is_probe & out.probe_block_number.eq(1), "previous_context"]) == {"B"}
    assert set(out.loc[out.is_probe & out.probe_block_number.eq(2), "next_context"]) == {"A"}
    expected_order = list(range(1, 13))
    assert out.groupby("participant_id").global_sequence.apply(
        lambda x: x.tolist() == expected_order
    ).all()
    assert out.loc[(out.participant_id == "P2") & out.block_id.eq(3) & out.sequence_in_block.eq(1), "error"].iloc[0] == pytest.approx(0.25)


def test_context_labels_are_validated(keypress_probe_frame):
    bad = keypress_probe_frame.copy()
    bad.loc[0, "trialType"] = "Mystery Chamber"
    with pytest.raises(UnknownContextLabelError, match="Mystery Chamber"):
        normalize_probe_data(bad, dataset="Combined", cohort_only=False)


def test_participants_are_never_combined(keypress_probe_frame):
    out = normalize_probe_data(keypress_probe_frame, dataset="Combined", cohort_only=False)
    p1 = out[out.participant_id.eq("P1")]
    p2 = out[out.participant_id.eq("P2")]
    assert p1.global_sequence.max() == p2.global_sequence.max() == 12
    assert p1.mean_rt.mean() != p2.mean_rt.mean()


def test_empty_input_is_explicit():
    empty = pd.DataFrame(columns=["ID", "trialType", "rt"])
    with pytest.raises(EmptyDatasetError):
        normalize_probe_data(empty, dataset="Novel")


def test_novel_aliases_are_mapped_without_one_hot_fallback(keypress_probe_frame):
    novel = keypress_probe_frame.copy()
    novel["trialType"] = novel.trialType.str.replace("Mixed", "Third Context", regex=False)
    out = normalize_probe_data(novel, dataset="Novel", cohort_only=False)
    assert set(out.loc[out.is_probe, "probe_type"]) == {"Novel"}
