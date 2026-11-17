from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from final_task_space_synthesis.analysis import (
    SELECTION_COLUMNS,
    augment_formal_metrics,
    select_aggregate_stable_cell,
    select_seed_stable_cells,
)
from final_task_space_synthesis.config import REQUIRED_SOURCE_CSVS, SynthesisConfig
from final_task_space_synthesis.data import load_phase1c_sources
from final_task_space_synthesis.formal import (
    audit_formal_identities,
    centering_matrix,
    linear_task_target,
    task_error,
    task_space,
)
from final_task_space_synthesis.pipeline import run_pipeline
from srn_mixed_novel_diagnostics.core import load_frozen_network
from srn_nonlinearity_compositionality.math import get_projectors


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "phase1c_pareto_path_sufficiency"
CHECKPOINTS = ROOT / "outputs" / "confirmatory_srn_final" / "confirmatory_checkpoints"


def _sha(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_formal_task_space_identities():
    assert all(audit_formal_identities().values())
    center = centering_matrix(4)
    assert np.allclose(center @ np.ones(4), 0)
    logits = np.arange(28, dtype=float).reshape(7, 4)
    assert np.allclose(task_space(logits), task_space(logits + 31.7))
    a, b = task_space(logits), task_space(logits[::-1])
    assert np.allclose(linear_task_target(a, b, 0), a)
    assert np.allclose(linear_task_target(a, b, 1), b)
    assert task_error((a + b) / 2, a, b, .5) == pytest.approx(0.)


def test_projector_identities_on_frozen_checkpoint():
    if not (CHECKPOINTS / "High_monotonic_seed_0.pt").exists():
        pytest.skip("Frozen confirmatory checkpoint is unavailable")
    frozen = load_frozen_network(CHECKPOINTS, "High_monotonic", 0)
    pout, pnull, _, _ = get_projectors(frozen.model)
    assert np.allclose(pout @ pout, pout, atol=1e-8)
    assert np.allclose(pnull @ pnull, pnull, atol=1e-8)
    assert np.allclose(pout @ pnull, 0, atol=1e-8)


def test_missing_phase1c_artifacts_fail_loudly(tmp_path):
    config = SynthesisConfig(source_dir=tmp_path, output_dir=tmp_path / "out",
                             checkpoint_dir=CHECKPOINTS, bootstrap_samples=100,
                             profile="test")
    with pytest.raises(FileNotFoundError, match="Missing required Phase 1C artifacts") as error:
        load_phase1c_sources(config)
    assert "phase1c_seed_metrics.csv" in str(error.value)
    assert "PHASE1C_PARETO_PATH_SUFFICIENCY_REPORT.md" in str(error.value)


def test_stable_selection_firewall_and_frozen_result():
    if not SOURCE.exists():
        pytest.skip("Completed Phase 1C outputs are unavailable")
    config = SynthesisConfig(source_dir=SOURCE, checkpoint_dir=CHECKPOINTS,
                             bootstrap_samples=100)
    tables, _, _ = load_phase1c_sources(config)
    master = augment_formal_metrics(tables["phase1c_seed_metrics"], 7)
    selection = master.loc[:, list(SELECTION_COLUMNS)]
    aggregate = select_aggregate_stable_cell(config, selection)
    seeds = select_seed_stable_cells(config, selection)
    assert aggregate.cell_id == "P1_correct_logit_alpha_10_anchor_30"
    assert len(seeds) == 20 and seeds.cell_id.notna().all()
    contaminated = selection.assign(Mixed_loss=master.Mixed_loss)
    with pytest.raises(AssertionError, match="received Mixed outcomes"):
        select_seed_stable_cells(config, contaminated)
    high = master[master.family.eq("P0_frozen_high")]
    assert np.allclose(high.normalized_logit_path_residual, 1.)
    assert np.allclose(high.normalized_output_path_residual, 1.)
    assert np.allclose(high.normalized_null_path_residual, 1.)
    low = master[master.family.eq("P5_frozen_low")]
    assert low.rescue_fraction_per_seed.isna().all()


@pytest.mark.integration
def test_full_read_only_synthesis(tmp_path):
    if not SOURCE.exists() or not CHECKPOINTS.exists():
        pytest.skip("Frozen Phase 1C sources are unavailable")
    source_files = [SOURCE / name for name in REQUIRED_SOURCE_CSVS]
    before = {path.name: _sha(path) for path in source_files}
    output = tmp_path / "final"
    result = run_pipeline(SynthesisConfig(
        source_dir=SOURCE, output_dir=output, checkpoint_dir=CHECKPOINTS,
        bootstrap_samples=100))
    after = {path.name: _sha(path) for path in source_files}
    assert before == after
    assert len(result["seed_summary"]) == 20
    assert result["seed_summary"].passes_full_feasible_gate.mean() == pytest.approx(.20)
    assert result["manifest"]["synthesis_optimizer_steps"] == 0
    assert result["manifest"]["new_models_trained"] == 0
    assert result["manifest"]["invariants_all_passed"] is True

    required = [
        "audit/PHASE1C_LABEL_AUDIT.md", "audit/PHASE1C_LABEL_AUDIT.csv",
        "audit/WRONG_PATH_MATCHING_AUDIT.csv",
        "tables/FINAL_seed_level_sufficiency_summary.csv",
        "tables/FINAL_TABLE01_formal_mechanism_summary.csv",
        "tables/FINAL_statistical_summary.csv",
        "figures/FINAL_FIG01_mechanism_triangle.png",
        "figures/FINAL_FIG02_correct_vs_wrong_path_specificity.png",
        "figures/FINAL_FIG03_seed_level_sufficiency.png",
        "figures/FINAL_FIG04_causal_chain.png",
        "text/FINAL_MECHANISTIC_RESULTS_SUMMARY.md",
        "text/FINAL_PLAIN_LANGUAGE_MECHANISM.md", "text/FINAL_FIGURE_CAPTIONS.md",
        "FINAL_TASK_SPACE_COMPOSITIONALITY_REPORT.md", "experiment_manifest.json",
    ]
    required += [f"timeseries/FINAL_TS0{i}_{name}.png" for i, name in enumerate((
        "full_familiar_sequence_loss", "full_familiar_correct_token_probability",
        "full_familiar_execution_proxy", "full_probe_sequence_loss",
        "full_probe_correct_token_probability", "full_probe_execution_proxy",
        "midpoint_over_finetuning", "cue_sweep_sequence_loss",
        "cue_sweep_task_path_residual"), start=1)]
    assert not [name for name in required if not (output / name).exists()]
    assert pd.read_csv(output / "audit/WRONG_PATH_MATCHING_AUDIT.csv").all_exact.all()
    assert pd.read_csv(output / "audit/SYNTHESIS_INVARIANT_AUDIT.csv").passed.all()

    label = pd.read_csv(output / "audit/PHASE1C_LABEL_AUDIT.csv")
    assert label.qualified_label.is_unique
    inferential = label[label.original_label.eq("new synthesis label")]
    assert inferential.selection_firewall_passed.all()
    report = (output / "FINAL_TASK_SPACE_COMPOSITIONALITY_REPORT.md").read_text()
    for section in range(1, 14):
        assert f"## {section}." in report
    formal = pd.read_csv(output / "tables/FINAL_TABLE01_formal_mechanism_summary.csv")
    high = formal[formal.condition.eq("Baseline High")].iloc[0]
    assert f"{high.Mixed_sequence_loss:.3f}" in report
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    assert manifest["claims"]["feasible_seed_fraction"] == pytest.approx(.20)
