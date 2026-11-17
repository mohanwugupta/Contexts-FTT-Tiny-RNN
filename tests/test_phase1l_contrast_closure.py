from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from confirmatory_srn.training import state_hash
from ftt_srn_execution_v2.inputs.token_onehot import encode_sequence
from ftt_srn_execution_v2.training.ftt_schedule import make_training_schedule
from phase1i_context_composition.cue_encoding import CUES
from phase1i_context_composition.mechanics import centered
from phase1l_contrast_closure.config import C0, C1, C2, C3, CONDITIONS, HIGH, smoke_config
from phase1l_contrast_closure.evaluation import add_novel_mean, evaluate_checkpoint, natural_rollout
from phase1l_contrast_closure.model import ContrastBottleneckSRN, initialize_model
from phase1l_contrast_closure.plotting import enforce_figures
from phase1l_contrast_closure.projection import cue_projection_audit, fit_familiar_contrast, project_context
from phase1l_contrast_closure.training import closure_components, train_run


@pytest.fixture(scope="module")
def config(): return smoke_config(".", "/tmp/phase1l_test")


@pytest.fixture(scope="module")
def audit(config): return cue_projection_audit(CUES, CUES["A"], CUES["B"], config.epsilon)


@pytest.fixture(scope="module")
def trained(config, audit, tmp_path_factory):
    root = tmp_path_factory.mktemp("phase1l_models"); projection_hash = hashlib.sha256(audit.to_csv(index=False).encode()).hexdigest()
    result = {}
    for condition in CONDITIONS:
        path = root / f"{condition}.pt"; model, acquisition, payload = train_run(config, condition, HIGH, 0, CUES, path)
        transition, neutral, provenance = evaluate_checkpoint(config, condition, HIGH, 0, model.eval(), payload, path, CUES, projection_hash)
        result[condition] = {"model": model, "acquisition": acquisition, "payload": payload, "transition": transition, "neutral": neutral, "provenance": provenance}
    return result


def test_1_identical_canonical_training_stream(trained, config):
    expected = make_training_schedule(config.n_blocks, config.block_length)
    assert len(expected) == 170
    reference = None
    for condition in CONDITIONS:
        stream = trained[condition]["acquisition"][["training_sequence_index", "block_index", "sequence_in_block", "training_context"]].reset_index(drop=True)
        if reference is None: reference = stream
        assert stream.equals(reference)
        assert stream.training_context.tolist() == expected.context.tolist()


def test_2_familiar_projection_identity(audit):
    values = audit.set_index("cue_name").alpha
    assert np.isclose(values.A, 1, atol=1e-7) and np.isclose(values.B, -1, atol=1e-7)


def test_3_projection_function_has_no_probe_label_special_cases():
    source = inspect.getsource(project_context)
    assert all(label not in source for label in ("Mixed", "NoContext", "Novel"))
    assert set(inspect.signature(fit_familiar_contrast).parameters) == {"cue_a", "cue_b"}


def test_4_physical_mixed_midpoint_projects_to_zero(audit):
    assert np.isclose(audit.set_index("cue_name").loc["Mixed", "alpha"], 0, atol=1e-7)


def test_5_nocontext_and_novel_use_computed_coordinates_unchanged(audit, config):
    cbar, direction = fit_familiar_contrast(CUES["A"], CUES["B"]); lookup = audit.set_index("cue_name").alpha
    model = initialize_model(C1, HIGH, 0, config, cbar, direction)
    for cue in ("NoContext", "Novel1", "Novel2", "Novel3", "Novel4"):
        tensor = torch.as_tensor(CUES[cue]).unsqueeze(0)
        assert np.isclose(float(model.context_coordinate(tensor)), lookup[cue], atol=1e-7)


def test_6_context_pathway_is_rank_one_for_C1_C3(trained):
    for condition in (C1, C3):
        model = trained[condition]["model"]
        assert isinstance(model, ContrastBottleneckSRN)
        assert torch.linalg.matrix_rank(model.effective_context_matrix()).item() == 1
        drives = torch.stack([model.context_drive(torch.as_tensor(CUES[name]).unsqueeze(0)).squeeze(0) for name in CUES])
        assert torch.linalg.matrix_rank(drives, tol=1e-5).item() <= 1


def test_7_context_u_uses_nonzero_high_scale_initialization(trained):
    for condition in (C1, C3):
        payload = trained[condition]["payload"]; initial = payload["context_u_initial"]
        assert payload["u_initialization_scale"] == 2.0 and initial is not None
        assert not torch.equal(initial, torch.zeros_like(initial)) and float(initial.std()) > .1


def _closure_setup(config, condition=C3):
    cbar, direction = fit_familiar_contrast(CUES["A"], CUES["B"])
    model = initialize_model(condition, HIGH, 3, config, cbar, direction)
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0)
    a = torch.as_tensor(CUES["A"]).unsqueeze(0); b = torch.as_tensor(CUES["B"]).unsqueeze(0); neutral = torch.as_tensor(cbar).unsqueeze(0)
    return model, tokens, a, b, neutral


def test_8_closure_midpoint_target_is_stop_gradient(config):
    model, tokens, a, b, neutral = _closure_setup(config)
    _, target, denominator, _, _ = closure_components(model, tokens, a, b, neutral, config.epsilon)
    assert not target.requires_grad and not denominator.requires_grad


def test_9_training_module_has_strict_probe_firewall():
    import phase1l_contrast_closure.training as module
    source = inspect.getsource(module)
    assert all(label not in source for label in ("Mixed", "NoContext", "Novel1", "Novel2", "Novel3", "Novel4"))


def test_10_primary_lambda_is_fixed_without_probe_selection(config):
    assert config.lambda_closure == 1.0
    import phase1l_contrast_closure.training as module
    source = inspect.getsource(module.train_run)
    assert "lambda_closure" in source and "select" not in source.lower()


def test_11_neutral_rollout_is_autoregressive_in_hidden_state(config):
    model, tokens, _, _, neutral = _closure_setup(config)
    with torch.no_grad(): _, hidden, _ = model(tokens, neutral)
    drive = model.context_drive(neutral); first = torch.tanh(tokens[:, 0] @ model.weight_seq.T + drive + model.bias_ih + model.bias_hh)
    second = torch.tanh(tokens[:, 1] @ model.weight_seq.T + drive + first @ model.weight_hh.T + model.bias_ih + model.bias_hh)
    assert torch.equal(hidden[:, 0], first) and torch.equal(hidden[:, 1], second)


def test_12_closure_target_is_exactly_AB_only_midpoint(config):
    model, tokens, a, b, neutral = _closure_setup(config)
    with torch.no_grad(): _, ha, _ = model(tokens, a); _, hb, _ = model(tokens, b)
    _, target, _, _, _ = closure_components(model, tokens, a, b, neutral, config.epsilon)
    assert torch.equal(target, .5 * (ha + hb))


def test_13_closure_denominator_is_detached_AB_separation(config):
    model, tokens, a, b, neutral = _closure_setup(config)
    with torch.no_grad(): _, ha, _ = model(tokens, a); _, hb, _ = model(tokens, b)
    _, _, denominator, _, _ = closure_components(model, tokens, a, b, neutral, config.epsilon)
    expected = (ha - hb).square().sum(-1) + config.epsilon
    assert torch.equal(denominator, expected) and not denominator.requires_grad


def test_14_evaluation_has_no_hidden_patch_path():
    import phase1l_contrast_closure.evaluation as module
    source = inspect.getsource(module.natural_rollout)
    assert "patch" not in source.lower() and "intervention" not in source.lower()


def test_15_every_final_probe_has_exactly_seven_raw_CEs(trained):
    full = add_novel_mean(pd.concat([value["transition"] for value in trained.values()], ignore_index=True))
    counts = full.groupby(["condition", "family", "seed", "probe"]).raw_ce.count()
    assert counts.eq(7).all()


def test_16_novelmean_is_nested_within_seed(trained):
    full = add_novel_mean(pd.concat([value["transition"] for value in trained.values()], ignore_index=True))
    derived = full[full.probe.eq("NovelMean")]; variants = full[full.probe.str.match(r"Novel[1-4]")]
    assert len(derived) * 4 == len(variants)
    for identity, group in variants.groupby(["condition", "family", "seed", "transition"]):
        value = derived[(derived.condition.eq(identity[0])) & derived.family.eq(identity[1]) & derived.seed.eq(identity[2]) & derived.transition.eq(identity[3])].raw_ce.iloc[0]
        assert np.isclose(value, group.raw_ce.mean())


def test_17_closure_target_never_averages_probabilities():
    source = inspect.getsource(closure_components)
    assert "softmax" not in source and "probab" not in source.lower()


def test_18_centered_logits_have_zero_class_mean(trained, config):
    run = natural_rollout(trained[C3]["model"], config.sequence, CUES["A"])
    assert np.allclose(centered(run["logits"]).mean(axis=1), 0, atol=1e-12)


def test_19_acquisition_losses_are_logged_pre_update(trained, config):
    first = trained[C0]["acquisition"].iloc[0]
    cbar, direction = fit_familiar_contrast(CUES["A"], CUES["B"]); model = initialize_model(C0, HIGH, 0, config, cbar, direction)
    tokens = encode_sequence(config.sequence[:-1], as_tensor=True).unsqueeze(0); targets = torch.as_tensor(config.sequence[1:])
    logits, _, _ = model(tokens, torch.as_tensor(CUES["A"]).unsqueeze(0)); ce = F.cross_entropy(logits.squeeze(0), targets)
    assert np.isclose(first.raw_sequence_ce_pre_update, float(ce.detach()), atol=1e-7)


def test_20_parameter_hashes_change_at_every_update(trained):
    for condition in CONDITIONS:
        trace = trained[condition]["acquisition"]
        assert trace.parameters_changed.all() and trace.parameter_hash.nunique() == 170


def test_21_evaluation_rows_use_final_checkpoint_hash(trained):
    for condition in CONDITIONS:
        expected = trained[condition]["payload"]["final_parameter_hash"]
        assert set(trained[condition]["transition"].parameter_hash) == {expected}
        assert trained[condition]["provenance"]["final_parameter_hash"] == expected


def test_22_all_ten_figures_are_mandatory(tmp_path):
    with pytest.raises(RuntimeError, match="Missing Phase 1L mandatory figures"):
        enforce_figures(tmp_path)
