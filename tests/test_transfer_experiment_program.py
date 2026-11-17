from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ordinal_layout_ecology.inputs import feature_sequence
from transfer_experiment_program.analysis import cluster_bootstrap, strict_acquisition_matches
from transfer_experiment_program.config import load_config
from transfer_experiment_program.experiments import (
    PRIOR_SEQUENCES, TARGET_SEQUENCE, compose_features, factorial_features,
    pretraining_schedule, prior_layouts,
)
from transfer_experiment_program.human import build_human_event_audit, validate_human_analysis_input
from transfer_experiment_program.human_task import (
    assignment_cells, assignment_index, schedule_for_assignment,
)
from transfer_experiment_program.modeling import SRNEngine, same_snapshot_contrasts, sequence_metrics
from transfer_experiment_program.power import simulate_power_planning


ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_disjoint_cohorts_and_budgets():
    config = load_config(ROOT / "configs/transfer_experiment_program.yaml")
    assert config.original_seeds == tuple(range(20))
    assert config.additional_seeds == tuple(range(1000, 1020))
    assert not set(config.original_seeds) & set(config.additional_seeds)
    assert config.bootstrap_draws == 2000
    assert config.simulation_draws == 1000


def test_transition_means_reconstruct_and_measurement_precedes_update():
    engine = SRNEngine("High_monotonic", 42, max_sequences=20)
    transitions, row, _ = engine.score_sequence(
        context="A", branch_id="test:update", update=True,
        metadata={"experiment": "test", "cohort": "constructed"},
    )
    assert tuple(transitions.transition_position) == tuple(range(1, 8))
    assert row["sequence_scored_before_update"]
    assert row["authorized_update_after_score"]
    assert row["parameters_changed"]
    assert row["mean_prediction_NLL"] == pytest.approx(transitions.correct_token_nll.mean(), abs=1e-7)
    assert row["mean_normalized_RT"] == pytest.approx(transitions.normalized_RT.mean(), abs=1e-7)


def test_frozen_branch_isolated_and_optimizer_parent_is_deep_copied():
    parent_engine = SRNEngine("High_monotonic", 8, max_sequences=20)
    parent_engine.score_sequence(context="A", branch_id="parent", update=True,
                                 metadata={"experiment":"test","cohort":"constructed"})
    parent = parent_engine.snapshot()
    first, second = SRNEngine.from_snapshot(parent), SRNEngine.from_snapshot(parent)
    before = second.snapshot()
    first.score_sequence(context="B", branch_id="first", update=True,
                         metadata={"experiment":"test","cohort":"constructed"})
    # A sibling update must not mutate the cached optimizer tensors.
    assert second.snapshot().optimizer_sha256 == before.optimizer_sha256
    second.score_sequence(context="Mixed", branch_id="frozen", update=False,
                          metadata={"experiment":"test","cohort":"constructed"})
    after = second.snapshot()
    assert (before.state_sha256, before.optimizer_sha256, before.rng_sha256) == (
        after.state_sha256, after.optimizer_sha256, after.rng_sha256
    )


def test_reset_frozen_repeats_exactly_while_carry_is_inherited():
    engine = SRNEngine("High_monotonic", 3, max_sequences=20)
    reset = []
    for repetition in (1, 2):
        transitions, _, _ = engine.score_sequence(
            context="Mixed", branch_id=f"reset:{repetition}", update=False,
            metadata={"experiment":"test","cohort":"constructed","repetition":repetition},
        )
        reset.append(transitions.correct_token_probability.to_numpy())
    np.testing.assert_array_equal(reset[0], reset[1])
    _, _, terminal = engine.score_sequence(context="A", branch_id="primer", update=False,
                                            metadata={"experiment":"test","cohort":"constructed"})
    carried, _, _ = engine.score_sequence(context="Mixed", branch_id="carry", update=False,
                                           initial_state=terminal,
                                           metadata={"experiment":"test","cohort":"constructed"})
    assert not np.array_equal(reset[0], carried.correct_token_probability.to_numpy())
    assert not terminal.requires_grad


def test_paired_max_is_taken_within_seed_not_after_aggregation():
    rows = []
    for seed, values in ((0, {"A":1.,"B":3.,"Mixed":4.,"Novel":5.}),
                         (1, {"A":3.,"B":1.,"Mixed":4.,"Novel":5.})):
        for context, value in values.items():
            rows.append({"experiment":"x","cohort":"x","condition":"High_monotonic","seed":seed,
                         "parent_checkpoint":20,"completed_training_sequences":20,"incoming_context":"B",
                         "repetition":1,"primer":"none","weight_policy":"frozen","state_policy":"reset",
                         "context":context,"mean_prediction_NLL":value,"mean_normalized_RT":value})
    result=same_snapshot_contrasts(pd.DataFrame(rows),comparison_type="constructed")
    assert result[result.probe_context.eq("Mixed")].NLL_deficit_over_worse.tolist()==[1.,1.]


def _matching_frame(include_probe=False):
    rows=[]
    for condition, checkpoints in (("High_monotonic",(170,)),("Low_monotonic",(10,20))):
        for checkpoint in checkpoints:
            for context in ("A","B"):
                value = 1.0 if (condition=="High_monotonic" or checkpoint==20) else 1.3
                row={"condition":condition,"seed":1000,"parent_checkpoint":checkpoint,
                     "incoming_context":"A","context":context,"mean_prediction_NLL":value,
                     "token_accuracy":1.0}
                if include_probe: row["Novel_outcome_that_must_not_be_used"]=-999 if checkpoint==10 else 999
                rows.append(row)
    return pd.DataFrame(rows)


def test_matching_is_ab_only_keeps_failed_candidates_and_is_probe_invariant():
    one=strict_acquisition_matches(_matching_frame(),anchors=(170,),calipers=(.05,),accuracy_caliper=1/7)
    two=strict_acquisition_matches(_matching_frame(True),anchors=(170,),calipers=(.05,),accuracy_caliper=1/7)
    assert one.loc[one.selected,"low_candidate_checkpoint"].tolist()==[20]
    assert two.loc[two.selected,"low_candidate_checkpoint"].tolist()==[20]
    assert len(one)==2 and (~one.selected).any()
    assert not one.probe_columns_accessed.any()


def test_factorial_adapter_preserves_width_targets_and_endpoint_encoder():
    for anchor in ("A","B"):
        for family in ("Mixed","Novel"):
            assert np.array_equal(factorial_features(anchor,family,"00"),feature_sequence("stage2_layout",anchor))
            assert np.array_equal(factorial_features(anchor,family,"11"),feature_sequence("stage2_layout",family))
            assert factorial_features(anchor,family,"10").shape==(7,42)
    engine=SRNEngine("Low_monotonic",1,max_sequences=20)
    transitions,_,_=engine.score_sequence(context="Mixed",branch_id="factorial",update=False,
        features_np=factorial_features("A","Mixed","10"),
        metadata={"experiment":"test","cohort":"constructed"})
    assert tuple(transitions.target_token)==TARGET_SEQUENCE[1:]


def test_e5_pretraining_marginals_pairing_and_target_holdout():
    p1,p2=pretraining_schedule(1003,"P1"),pretraining_schedule(1003,"P2")
    assert len(p1)==len(p2)==640
    assert p1.sequence_id.tolist()==p2.sequence_id.tolist()
    assert set(p1.sequence_id.value_counts())=={80} and set(p1.layout_id.value_counts())=={80}
    assert set(p2.sequence_id.value_counts())=={80} and set(p2.layout_id.value_counts())=={80}
    assert set(p1.groupby(["sequence_id","layout_id"]).size())=={10}
    assert set(p2.groupby(["sequence_id","layout_id"]).size())=={80}
    reverse=tuple(reversed(TARGET_SEQUENCE))
    equivalents={seq[i:]+seq[:i] for seq in (TARGET_SEQUENCE,reverse) for i in range(8)}
    assert not equivalents.intersection(PRIOR_SEQUENCES)
    assert len(prior_layouts())==8 and all(layout.shape==(8,2) for layout in prior_layouts())


def test_human_counterbalances_are_balanced_reproducible_and_end_with_preceding_context():
    expected={"e2":4,"e3":16,"e4":16}
    for experiment,count in expected.items():
        cells=assignment_cells(experiment); assert len(cells)==count*4
        assert len({cell["core_cell_index"] for cell in cells})==count
        assert assignment_index("stored-code",len(cells))==assignment_index("stored-code",len(cells))
        for cell in cells:
            schedule=schedule_for_assignment(experiment,cell)
            assert len(schedule)==168
            acquisition=[row for row in schedule if row["phase"]=="acquisition"]
            assert acquisition[-1]["context"]==cell["preceding_context"]
            assert [row["repetition_in_block"] for row in schedule if row["phase"]=="test"]==list(range(1,9))
            assert all(row["response_keys"]==cell["response_keys"] for row in schedule)


def test_one_target_source_has_no_action_valued_future_attributes():
    source=(ROOT/"transfer_experiment_program/human_task/app.js").read_text()
    assert "data-token" not in source and "dataset.token" not in source
    assert "event.repeat" in source and "localStorage" in source
    assert "requestAnimationFrame" in source and 'emit("blur"' in source


def test_cluster_bootstrap_resamples_whole_clusters():
    frame=pd.DataFrame({"seed":np.repeat([0,1,2],2),"time":[20,30]*3,"value":range(6)})
    draws,summary=cluster_bootstrap(frame,cluster="seed",value="value",group=("time",),draws=12,seed=4)
    assert draws.groupby("draw").time.nunique().eq(2).all()
    assert summary.bootstrap_unit.eq("seed").all()


def test_power_is_prospective_and_reports_monte_carlo_uncertainty():
    result=simulate_power_planning(n_options=(64,),effects=(0.,.2),draws=8,seed=3,
                                   residual_sd=.25,participant_sd=.2,
                                   participant_missingness=.1,event_error=.05)
    assert result.synthetic_planning_only.all()
    assert set(result.operating_characteristic)=={"type_I_error","power"}
    assert result.monte_carlo_SE.notna().all()


def test_synthetic_events_cannot_enter_empirical_report():
    events=pd.DataFrame({"participant_uid":["x"],"data_origin":["synthetic_task_validation"],
                         "context":["A"],"absolute_event_order":[1]})
    with pytest.raises(ValueError,match="Synthetic"):
        validate_human_analysis_input(events,allow_synthetic=False)


def test_human_first_exposure_precedes_exclusion_and_error_does_not_rename_it(tmp_path, monkeypatch):
    rows=[]
    def sequence(block, number, label, error=False):
        if error:
            rows.append({"ID":"p1","new_trialNumber":block,"trialNumber":0,
                         "sequenceNumber":number,"keypressNumber":0,"trialType":label,
                         "counterBalance":1,"correct":False,"rt":400})
        for keypress in range(8):
            rows.append({"ID":"p1","new_trialNumber":block,"trialNumber":1 if error else 0,
                         "sequenceNumber":number,"keypressNumber":keypress,"trialType":label,
                         "counterBalance":1,"correct":True,"rt":500+keypress})
    sequence(1,0,"Maze Training")
    sequence(2,0,"Melody Training")
    sequence(3,0,"Mixed Test",error=True)
    sequence(3,1,"Mixed Test")
    path=tmp_path/"events.csv";pd.DataFrame(rows).to_csv(path,index=False)
    normalized=pd.DataFrame({"participant_id":["p1"],"block_id":[3],
                             "original_sequence_in_block":[2]})
    monkeypatch.setattr("transfer_experiment_program.human.load_retrieval_data",
                        lambda path,dataset: normalized.copy())
    events,sequences,audit=build_human_event_audit(path,"Combined")
    probes=sequences[sequences.context.eq("Mixed")].sort_values("first_source_row")
    assert probes.probe_exposure_index.astype(int).tolist()==[1,2]
    assert probes.is_first_ever_probe_family.tolist()==[True,False]
    assert probes.sequence_had_error.tolist()==[True,False]
    assert probes.existing_cleaning_included.tolist()==[False,True]
    assert audit["exposure_counts_computed_before_RT_exclusion"]
