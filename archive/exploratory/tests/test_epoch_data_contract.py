import pandas as pd

from probe_modeling.epoch_data import normalize_epoch_data


def test_v2_removes_first_sequence_and_retains_nine_completed_sequences(epoch_keypress_frame):
    assert 0 in set(epoch_keypress_frame.sequenceNumber)
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)

    assert set(clean.original_sequence_in_block) == set(range(2, 11))
    assert set(clean.analysis_sequence_in_block) == set(range(1, 10))
    assert 1 not in set(clean.original_sequence_in_block)
    assert clean.sequence_correct.all()
    counts = clean.groupby(["participant_id", "block_id"]).size()
    assert counts.eq(9).all()
    assert clean.sequence_in_block.equals(clean.analysis_sequence_in_block)


def test_v2_flags_incomplete_block_and_excludes_participant(epoch_keypress_frame):
    pid = epoch_keypress_frame.ID.iloc[0]
    incomplete = epoch_keypress_frame[
        ~(
            epoch_keypress_frame.ID.eq(pid)
            & epoch_keypress_frame.new_trialNumber.eq(8)
            & epoch_keypress_frame.sequenceNumber.eq(5)
        )
    ]
    clean = normalize_epoch_data(incomplete, dataset="Combined", cohort_only=False)
    assert pid not in set(clean.participant_id)
    assert any(item["participant_id"] == pid for item in clean.attrs["invalid_blocks"])


def test_v2_epoch_tagging_is_exact(epoch_keypress_frame):
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)
    for _, participant in clean.groupby("participant_id"):
        counts = participant.epoch.value_counts()
        assert counts["Mixed_1"] == 9
        assert counts["Mixed_2"] == 9
        assert counts["Mixed_Test"] == 9
        assert counts["Post_Mixed_Return"] == 9
        assert participant.loc[participant.epoch.eq("Post_Mixed_Return"), "context"].isin(["A", "B"]).all()
        assert participant.block_id.nunique() == 17


def test_counterbalance_history_is_explicit(epoch_keypress_frame):
    clean = normalize_epoch_data(epoch_keypress_frame, dataset="Combined", cohort_only=False)
    assert set(clean.counterbalance_history) == {"A_before_Mixed", "B_before_Mixed"}
    mixed1 = clean[clean.epoch.eq("Mixed_1")]
    assert (mixed1.previous_context == mixed1.counterbalance_history.str[0]).all()

