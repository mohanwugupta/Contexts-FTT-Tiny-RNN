import json

import pandas as pd

from ftt_srn_execution_v2.config import smoke_config
from ftt_srn_execution_v2.pipeline import REQUIRED_CSVS, run_pipeline


def test_execution_v2_pipeline_writes_complete_contract(tmp_path):
    result = run_pipeline(tmp_path, config=smoke_config())
    names = {path.name for path in tmp_path.iterdir()}
    assert set(REQUIRED_CSVS).issubset(names)
    assert {"run_config_v2.json", "execution_control_report.md"}.issubset(names)
    assert len(list((tmp_path / "figures_execution_v2").glob("*.png"))) == 9
    assert set(result["seed_summary"].model_condition) == {"High", "Low", "High_seq_only"}
    assert set(result["mixed_probe_results"].probe_type) == {"Mixed_norm", "Mixed_additive"}
    record = json.loads((tmp_path / "run_config_v2.json").read_text())
    assert record["mixed_novel_training_updates"] == 0
    assert record["external_competence_states"] is False
    assert record["novel_context_column_frozen_zero"] is True
    assert record["token_encoding"] == "strict 4D one-hot"
    for filename in REQUIRED_CSVS:
        assert not pd.read_csv(tmp_path / filename).empty

