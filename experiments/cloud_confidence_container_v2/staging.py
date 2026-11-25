"""Curated, byte-preserving staging; never mounts a host repository."""
import json
from pathlib import Path
import shutil

from .contract import ROOT, SELECTED, BUNDLE_MANIFEST_SHA, sha, verify_bundle, evidence_entries, compare_scientific_body


def overlay():
    files = {}
    for name, digest in SELECTED.items():
        rel = 'experiments/cloud_confidence_selected_v1/' + name
        if sha(ROOT / rel) != digest:
            raise ValueError('Changed selected implementation: ' + rel)
        files[rel] = digest
    for directory in ['cloud_confidence_container_v2', 'worker_isolation_preflight_v1']:
        for p in (ROOT / 'experiments' / directory).glob('*.py'):
            files[str(p.relative_to(ROOT))] = sha(p)
    compare_scientific_body()
    return files


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def stage(destination, evidence_root=None, selection=None, proposal=None, authorization=None):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Existing staging must not be overwritten')
    source_files = {}
    if evidence_root is not None:
        evidence_root = Path(evidence_root).resolve()
        manifest_path = evidence_root / 'CLOUD_INPUT_MANIFEST.json'
        if manifest_path.is_symlink() or sha(manifest_path) != BUNDLE_MANIFEST_SHA:
            raise ValueError('Wrong original bundle manifest')
        manifest = json.loads(manifest_path.read_text())
        files = evidence_entries(manifest)
        if len(files) != 1609:
            raise ValueError('Wrong original bundle coverage')
        verify_bundle(evidence_root, files)
        source_files.update({rel: evidence_root / rel for rel in files})
        source_files['CLOUD_INPUT_MANIFEST.json'] = manifest_path
    for rel in overlay():
        if rel in source_files and sha(source_files[rel]) != sha(ROOT / rel):
            raise ValueError('Overlay collision with frozen source: ' + rel)
        source_files[rel] = ROOT / rel
    if evidence_root is not None:
        for name, source in [('selection', selection), ('proposal', proposal), ('authorization', authorization)]:
            source = Path(source)
            if source.is_symlink() or not source.is_file():
                raise ValueError('Missing/unsafe selected request')
            source_files['selected_request/' + name + '.json'] = source
    destination.mkdir(parents=True)
    hashes = {}
    for rel, source in source_files.items():
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
        hashes[rel] = sha(source)
    for directory in destination.rglob('*'):
        if directory.is_dir():
            directory.chmod(0o755)
    destination.chmod(0o755)
    write_json(destination / 'STAGED_FILES.json', hashes)
    verify_bundle(destination, hashes)
    return hashes
