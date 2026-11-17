#!/usr/bin/env python3
"""Verify one completed, immutable evaluation-only paper freeze."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from context_sequence.provenance.hashing import sha256, source_hash
from context_sequence.paper.pipeline import load_contract
from context_sequence.provenance.registry import audit_inputs
from context_sequence.task.novelty import read_registry


def verify(config_path, result_path):
    root, contract = load_contract(config_path)
    out = Path(result_path).resolve()
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['freeze_id'] == contract['freeze_id']
    assert manifest['evaluation_only'] is True and manifest['optimizer_updates'] == 0
    assert manifest['contract_sha256'] == contract['contract_sha256']
    assert manifest['source_code_sha256'] == source_hash(root)
    registry = read_registry(root / contract['novelty_registry'], contract['novelty_registry_sha256'])
    assert manifest['registry_sha256'] == registry['registry_sha256']
    assert manifest['checkpoints'] == contract['checkpoints']
    assert manifest['input_tables'] == contract['input_tables']
    for record in contract['input_tables']:
        assert sha256(root / record['path']) == record['sha256'], record['path']
    checkpoints = audit_inputs(root, contract['checkpoints'])
    assert len(checkpoints) == 120 and checkpoints.passed.all()
    for name, expected in manifest['outputs'].items():
        target = out / name
        assert target.is_file() and sha256(target) == expected, name
    actual = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()} - {'manifest.json'}
    assert actual == set(manifest['outputs']), 'Output file set differs from manifest'
    for panel in manifest['figure_panels']:
        assert (out / panel['panel_data']).is_file(), panel['panel_data']
    assert {r['analysis'] for r in manifest['analysis_status']} == {f'A{i}' for i in range(1, 9)}
    assert {r['evidence_status'] for r in manifest['analysis_status'] if r['analysis'] == 'A2'} == {'insufficient_common_support'}
    for number in (1, 2, 3, 4):
        for suffix in ('png', 'pdf', 'svg'):
            assert (out / 'figures' / f'figure_{number}.{suffix}').is_file()
        assert (out / 'captions' / f'figure_{number}.md').is_file()
    print(f"Verified {manifest['freeze_id']}: {len(manifest['outputs'])} output hashes, {len(checkpoints)} checkpoints, {len(manifest['figure_panels'])} panels, A1–A8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs/paper_freeze_v2.json'))
    parser.add_argument('--result', default=None)
    args = parser.parse_args()
    contract = json.loads(Path(args.config).read_text())
    verify(args.config, args.result or ROOT / 'results/paper' / contract['freeze_id'])
