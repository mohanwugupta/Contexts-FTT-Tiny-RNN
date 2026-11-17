import json

import pandas as pd

from ftt_srn_causal_representation.config import smoke_config as causal_smoke_config
from ftt_srn_causal_representation.pipeline import REQUIRED_CSVS, run_causal_pipeline
from ftt_srn_execution_v2.config import ExperimentConfig
from ftt_srn_execution_v2.pipeline import run_pipeline as run_v21_pipeline


def test_causal_pipeline_loads_frozen_v21_checkpoints_and_writes_contract(tmp_path):
    source = tmp_path / "v21"
    source_config = ExperimentConfig(
        n_blocks=17,
        checkpoint_sequences=(20, 40, 80, 170),
        checkpoint_labels=(
            "early_after_initial_ab", "human_matched", "several_switches", "final"
        ),
        seeds=tuple(range(5)),
        make_plots=False,
        show_progress=False,
        profile="causal_fixture",
    )
    run_v21_pipeline(source, config=source_config)
    manifest = pd.read_csv(source / "model_checkpoint_manifest.csv")
    assert not manifest.empty

    output = tmp_path / "causal"
    result = run_causal_pipeline(source, output, config=causal_smoke_config())
    names = {path.name for path in output.iterdir()}
    assert set(REQUIRED_CSVS).issubset(names)
    assert {"causal_run_config.json", "causal_report.md"}.issubset(names)
    assert len(list((output / "figures_causal_v22").glob("*.png"))) == 10
    for filename in REQUIRED_CSVS:
        assert not pd.read_csv(output / filename).empty
    record = json.loads((output / "causal_run_config.json").read_text())
    assert record["optimizer_steps"] == 0
    assert record["backward_calls"] == 0
    assert record["source_splitting_gate_passed"] is True
    assert record["source_execution_grounding_gate_passed"] is True
    assert record["primary_checkpoint"] == "human_matched"
    assert result["ab_directions"].estimation_pass_id.str.contains("estimation").all()
    assert result["steering_results"].evaluation_pass_id.str.contains("evaluation").all()
    assert set(result["steering_results"].target_identity_preserved) == {True}
    assert result["steering_results"].query("alpha == 0").zero_intervention_identity.all()
    assert (result["steering_results"].query("intervention_level == 'recurrent'").horizon > 0).any()
    assert (result["off_manifold_metrics"].intervention_norm > 0).all()
    random = result["random_direction_controls"].query("control_test == 'steering'")
    assert (abs(random.reference_ab_norm - random.control_direction_norm) < 1e-8).all()
    audit = pd.read_csv(output / "frozen_checkpoint_audit.csv")
    assert audit.weights_unchanged.all()
    assert (audit.optimizer_steps == 0).all()
    assert (audit.backward_calls == 0).all()
