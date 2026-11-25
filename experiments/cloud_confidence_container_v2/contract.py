"""Pure staging/authorization contracts, without scientific execution."""
import ast
import datetime
import hashlib
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_MANIFEST_SHA = '199556216472f99eb1d250d757f1bd6163edc50b39eb31e189fab3fec5fc8ebe'
SELECTED = {
    'analysis.py': '60bd5397bfd550d12b9e36b8be12173fbc747fae8b10706c73c376babc8fe3bb',
    '__init__.py': '1f2c73a7df6843e1e4477e763f6efd35ee2c342b7bff9c05dae509a544c3d0d6',
    'sandbox.py': '829df1d98274552f1a7a48b2cb8ecc8da92a729b266ccab7875bcccd5f68edec',
    'figures.py': 'ebbeedfd5da9d3ef5d92e41e1930b5d94296897ee727569b51e0e888c24faf4d',
    'driver.py': 'ebe3d0d294efc85de6c78bb716aecbcd9e3a8ac14dee7b22fc690c4b1d3a7c26',
}
FORBIDDEN = {'.git', '.aws', '.codex', '.venv', '__pycache__', 'raw',
             'benchmark_banks', 'sealed', 'agent_evidence', 'runtime_validation'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_relative(value):
    p = PurePosixPath(value)
    if not value or '\\' in value or p.is_absolute() or '..' in p.parts or str(p) != value:
        raise ValueError('Unsafe relative path')
    if any(x in FORBIDDEN for x in p.parts) or p.name in {'auth.json', '.env'}:
        raise ValueError('Forbidden private path')
    if p.suffix in {'.pt', '.npy', '.npz', '.sqlite', '.pem', '.key'} or p.parts[0] == 'data':
        raise ValueError('Checkpoints, credentials and human data are not staged')
    return value


def evidence_entries(manifest):
    files = manifest['files']
    if not files or not isinstance(files, dict):
        raise ValueError('Empty evidence manifest')
    for name, expected in files.items():
        checked_relative(name)
        if not re.fullmatch('[a-f0-9]{64}', expected):
            raise ValueError('Invalid source hash')
        if name.startswith('research_search/database/') and not name.startswith('research_search/database/controlled_pilot_v2/'):
            raise ValueError('Unregistered database namespace')
    for name, expected in manifest['native_consumed_inputs'].items():
        if files.get(name) != expected:
            raise ValueError('Native source outside manifest or different hash')
    return files


def verify_bundle(root, files):
    root = Path(root).resolve()
    for name, expected in files.items():
        checked_relative(name)
        p = root / name
        if p.is_symlink() or any(q.is_symlink() for q in p.parents if q != root):
            raise ValueError('Symlink in source path')
        if not p.is_file() or not p.resolve().is_relative_to(root) or sha(p) != expected:
            raise ValueError('Changed/missing input: ' + name)


def validate_authorization(record, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    exact = dict(version='container-execution-v2', owner_approved=True,
                 scope='one_cached_confidence_execution', CPU_cap_seconds=120,
                 wall_cap_seconds=300, new_model_forwards=0,
                 new_optimizer_steps=0, new_agent_calls=0)
    for key, value in exact.items():
        if type(record.get(key)) is not type(value) or record[key] != value:
            raise ValueError('Missing/incorrect explicit authorization: ' + key)
    for field in ['proposal_sha256', 'selection_sha256']:
        if not re.fullmatch('[a-f0-9]{64}', record.get(field, '')):
            raise ValueError('Missing selected-request hash')
    deadline = datetime.datetime.fromisoformat(record['deadline_utc'].replace('Z', '+00:00'))
    if deadline.tzinfo is None or now.tzinfo is None:
        raise ValueError('Timezone required')
    remaining = (deadline - now).total_seconds()
    if remaining <= 0:
        raise ValueError('Explicit execution deadline passed; never reset it')
    return min(300, remaining)


def compare_scientific_body():
    """Allow only relative-import relocation, never algorithm changes."""
    old_path = ROOT / 'experiments/cloud_confidence_selected_v1/driver.py'
    if sha(old_path) != SELECTED['driver.py']:
        raise ValueError('Frozen selected driver changed')
    old = next(n for n in ast.parse(old_path.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == 'child')
    start = next(i for i, n in enumerate(old.body)
                 if isinstance(n, ast.ImportFrom) and n.module == 'experiments.research_search_cloud_v1.port')
    wanted = old.body[start:]
    for node in wanted:
        for n in ast.walk(node):
            if isinstance(n, ast.ImportFrom) and n.level:
                if n.module not in ['analysis', 'figures'] or n.level != 1:
                    raise ValueError('Unexpected relative import in source')
                n.module = 'experiments.cloud_confidence_selected_v1.' + n.module
                n.level = 0
    new = next(n for n in ast.parse(Path(__file__).with_name('compute.py').read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == 'compute')
    canonical = lambda body: ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
    if canonical(wanted) != canonical(new.body):
        raise ValueError('Scientific body differs from frozen source')
    return True
