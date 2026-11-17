import pandas as pd

from retrieval_modeling.data import normalize_retrieval_data


def _third_context_raw():
    rows = []
    labels = ["Maze Training", "Melody Training", "Novel Training", "No Context Training"]
    for participant in ("N1", "N2"):
        for block, label in enumerate(labels, start=1):
            for sequence in range(10):
                for key in range(8):
                    rows.append(
                        {
                            "ID": participant,
                            "new_trialNumber": block,
                            "sequenceNumber": sequence,
                            "keypressNumber": key,
                            "trialType": label,
                            "counterBalance": "novel_and_default",
                            "correct": True,
                            "rt": 400 + 20 * block + sequence + key,
                        }
                    )
    return pd.DataFrame(rows)


def test_third_context_adapter_retains_novel_and_no_context_probes():
    clean = normalize_retrieval_data(_third_context_raw(), dataset="Novel")
    assert set(clean.retrieval_context) == {"A", "B", "Novel", "Default"}
    assert not clean.original_sequence_in_block.eq(1).any()
    assert set(clean.analysis_sequence_in_block) == set(range(1, 10))
    assert clean.groupby(["participant_id", "block_id"]).size().eq(9).all()
    assert clean.sequence_correct.all()
    assert set(clean.loc[clean.retrieval_context.eq("Default"), "probe_type"]) == {"Default"}


def _actual_three_context_cohorts():
    schedules = {
        "novel_trained": (2, ["Maze Training", "Melody Training", "Space Training", "test spaceA"]),
        "default_trained": (4, ["Maze Training", "Melody Training", "No Context Training", "No Context Test"]),
        "novel_control": (1, ["Maze Training", "Melody Training", "Maze Training", "test spaceA"]),
    }
    rows = []
    for participant, (counterbalance, labels) in schedules.items():
        for block, label in enumerate(labels, start=1):
            for sequence in range(10):
                for key in range(8):
                    rows.append(
                        {
                            "ID": participant, "new_trialNumber": block,
                            "sequenceNumber": sequence, "keypressNumber": key,
                            "trialType": label, "counterBalance": counterbalance,
                            "correct": True, "rt": 400 + block + sequence + key,
                        }
                    )
    return pd.DataFrame(rows)


def test_actual_space_labels_and_exposure_cohorts_remain_distinct():
    clean = normalize_retrieval_data(_actual_three_context_cohorts(), dataset="Novel")
    people = clean[
        ["participant_id", "probe_exposure_cohort", "probe_training_context", "is_probe_control"]
    ].drop_duplicates().set_index("participant_id")
    assert people.loc["novel_trained", "probe_exposure_cohort"] == "Novel_training"
    assert people.loc["default_trained", "probe_exposure_cohort"] == "Default_training"
    assert people.loc["novel_control", "probe_exposure_cohort"] == "Novel_test_only_control"
    assert bool(people.loc["novel_control", "is_probe_control"])

    probes = clean[clean.is_probe]
    assert set(probes.loc[probes.raw_context.eq("Space Training"), "probe_phase"]) == {"training"}
    assert set(probes.loc[probes.raw_context.eq("test spaceA"), "probe_phase"]) == {"test"}
    control_probe = probes[probes.participant_id.eq("novel_control")]
    assert control_probe.is_primary_probe.all()
    trained_test = probes[
        probes.participant_id.eq("novel_trained") & probes.raw_context.eq("test spaceA")
    ]
    assert not trained_test.is_primary_probe.any()
