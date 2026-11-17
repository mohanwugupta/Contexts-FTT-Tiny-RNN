import subprocess
import sys
from pathlib import Path
import pytest
import torch
from context_sequence.dynamics.pipeline import check_mode, evaluation_firewall
ROOT=Path(__file__).resolve().parents[2]


def test_E1_E2_no_implicit_training_mode():
    for stage in ['e1','e2']:
        assert check_mode(stage,'eval_only') is None
        with pytest.raises(ValueError):check_mode(stage,'confirmatory')
        with pytest.raises(ValueError):check_mode(stage,'ab_pilot')
    with evaluation_firewall():
        with pytest.raises(RuntimeError):torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
        with pytest.raises(RuntimeError):torch.ones(1,requires_grad=True).sum().backward()


def test_R_cli_help_has_explicit_modes():
    r=subprocess.run([sys.executable,str(ROOT/'scripts/run_dynamical_generalization.py'),'--help'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert 'eval_only' in r.stdout and 'ab_pilot' in r.stdout and 'confirmatory' in r.stdout


def test_missing_sources_are_named_without_training(tmp_path):
    from context_sequence.dynamics.registry import audit
    config={'source_files':{'source_missing.py':'expected-source-digest'},'input_files':[],
            'checkpoints':[{'path':'checkpoint_missing.pt','checkpoint_sha256':'expected-checkpoint-digest'}]}
    with evaluation_firewall():result=audit(tmp_path,config)
    assert set(result.path)=={'source_missing.py','checkpoint_missing.pt'}
    assert result.status.eq('blocked_missing_source').all()
    assert result.observed_sha256.isna().all()
