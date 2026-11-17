"""PRD T1--T12 for emergent retrieval/context inference."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from confirmatory_srn.model import MonotonicExecutionPolicy, normalized_rt
from emergent_retrieval_program.config import load_retrieval_config
from emergent_retrieval_program.controls import build_control, train_baseline
from emergent_retrieval_program.inputs import (
    EVIDENCE_DIM, baseline_features, context_evidence, familiar_curriculum,
)
from emergent_retrieval_program.modeling import curriculum_sha256, train_emergent_model
from emergent_retrieval_program.models import EmergentRetrievalModel
from emergent_retrieval_program.pipeline import REQUIRED_OUTPUTS, run_retrieval_program
from human_svg_geometry_transfer.geometry import geometry_features, load_coordinate_registry


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retrieval_program_config.json"
REGISTRY = ROOT / "configs/model_variant_registry.json"


def _loaded(output=None):
    config = load_retrieval_config(CONFIG, REGISTRY, smoke=True, output=output)
    registry = load_coordinate_registry(config.coordinate_registry, config.layout_assignment)
    return config, registry


def test_t1_t2_firewall_and_no_probe_objective_imports():
    config, registry = _loaded()
    run = train_emergent_model(config, registry, "E1", "Low_monotonic", 0)
    assert run.audit["num_train_mixed"] == run.audit["num_train_novel"] == 0
    assert not run.audit["interpolation_objective_used"]
    source = (ROOT / "emergent_retrieval_program/modeling.py").read_text()
    tree = ast.parse(source)
    total_loss = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "total_loss" for target in node.targets)]
    assert len(total_loss) == 1
    expression = ast.unparse(total_loss[0].value).lower()
    assert all(word not in expression for word in ("mixed", "novel", "midpoint", "corridor", "lambda"))
    assert "from .analysis" not in source


def test_t3_t4_byte_identical_curriculum_and_exact_g2_adapter():
    config, registry = _loaded()
    assert familiar_curriculum(config.training_sequences) == tuple(["A"] * 10 + ["B"] * 10)
    assert len({curriculum_sha256(config.training_sequences) for _ in range(7)}) == 1
    for context in ("A", "B", "Mixed", "Novel"):
        evidence = context_evidence(registry, context)
        assert evidence.shape == (EVIDENCE_DIM,)
        np.testing.assert_array_equal(
            baseline_features(registry, context),
            geometry_features("G2_shared_x", context, registry),
        )


def test_t5_t6_fixed_rt_mapping_and_two_dimensional_latent():
    confidence = torch.tensor([[.2, .8]])
    for variant in ("E1", "E2", "E3", "E4"):
        model = EmergentRetrievalModel(variant, .02)
        assert model.latent_dim == 2
        policy = MonotonicExecutionPolicy()
        speed = policy(confidence)
        torch.testing.assert_close(normalized_rt(speed), 1 - speed)


def test_t7_history_reset_and_carry_semantics():
    evidence = torch.zeros(EVIDENCE_DIM)
    previous = torch.tensor([.75, -.25])
    for variant in ("E2", "E3"):
        torch.manual_seed(10)
        model = EmergentRetrievalModel(variant, .02)
        reset_first = model.initial_context(evidence, None)
        reset_second = model.initial_context(evidence, None)
        torch.testing.assert_close(reset_first, reset_second)
        if variant == "E2":
            carried = model.initial_context(evidence, previous)
            assert not torch.equal(carried, reset_first)
        else:
            carried = model.initial_context(evidence, previous, ablation="carry_override")
            torch.testing.assert_close(carried, previous)


def test_t8_t9_t10_frozen_probes_and_honest_labels():
    config, registry = _loaded()
    baseline = train_baseline(config, registry, "Low_monotonic", 0)
    c1 = build_control(config, registry, baseline, "C1")
    emergent = train_emergent_model(config, registry, "E4", "Low_monotonic", 0)
    assert baseline.audit["probe_parameters_unchanged"]
    assert c1.audit["probe_parameters_unchanged"]
    assert emergent.audit["probe_parameters_unchanged"]
    assert c1.probes.averaging_imposed.all()
    assert not emergent.probes.averaging_imposed.any()


def test_t11_t12_analysis_boundary_and_seed_pairing():
    config, _ = _loaded()
    model_source = (ROOT / "emergent_retrieval_program/models.py").read_text()
    training_source = (ROOT / "emergent_retrieval_program/modeling.py").read_text()
    assert ".analysis" not in model_source + training_source
    assert config.payload["seeds"]["screen"] == list(range(5))
    assert config.payload["seeds"]["confirmatory"] == list(range(20))


def test_smoke_pipeline_writes_contract(tmp_path):
    config, _ = _loaded(tmp_path)
    tables = run_retrieval_program(config)
    assert all((tmp_path / name).exists() for name in REQUIRED_OUTPUTS)
    audit = tables["retrieval_training_audit.csv"]
    assert audit.num_train_mixed.eq(0).all() and audit.num_train_novel.eq(0).all()
    assert tables["retrieval_v0_reproduction.csv"].within_tolerance.all()
