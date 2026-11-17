import json

import pandas as pd

from dataclasses import replace

from srn_environmental_pressures.config import smoke_config, with_software_stages
from srn_environmental_pressures.pipeline import REQUIRED_CSVS, run_pipeline


def test_environmental_pressure_pipeline_writes_full_staged_contract(tmp_path):
    config = smoke_config()
    result = run_pipeline(tmp_path, config=config)
    names = {path.name for path in tmp_path.iterdir()}
    assert set(REQUIRED_CSVS).issubset(names)
    assert {"run_config.json", "environmental_pressures_report.md"}.issubset(names)
    assert len(list((tmp_path / "figures_environmental_pressures").glob("*.png"))) == 9
    assert not result["all_seed_results"].empty
    record = json.loads((tmp_path / "run_config.json").read_text())
    assert record["primary_architecture"] == "8-unit Elman SRN"
    assert record["fixed_training"]["block_length"] == 10
    assert record["human_data_used"] is False
    assert record["optimization_dimensions_searched"] == []
    for name in ("confirmation_results.csv", "environmental_ablation_results.csv",
                 "hidden_state_probes.csv", "causal_interventions.csv"):
        frame = pd.read_csv(tmp_path / name)
        assert "status" in frame


def test_forced_software_profile_exercises_confirmation_ablations_and_mechanisms(tmp_path):
    config = replace(with_software_stages(smoke_config()), make_plots=False, show_progress=False)
    result = run_pipeline(tmp_path, config=config)
    assert result["confirmation_results"].status.eq("completed").all()
    assert set(result["confirmation_results"].seed).isdisjoint(config.discovery_seeds)
    assert result["environmental_ablation_results"].status.eq("completed").all()
    assert result["hidden_state_probes"].status.eq("completed").all()
    assert result["causal_interventions"].status.eq("completed").all()
