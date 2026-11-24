"""Portable saved-evidence staging; no Mac CLI/auth or model execution."""
import hashlib,json,os,platform,shutil,sys,tarfile,time,tempfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2]
CODE_DIRS={p.name for p in ROOT.iterdir() if p.is_dir() and (p/'__init__.py').is_file()} | {'experiments','src','confirmatory_srn','execution_objective_search','research_search','tests','scripts','ftt_srn_execution_v2','ftt_srn_causal_representation','ftt_srn_context_sequence_interaction'}
FORBIDDEN={'.codex','.aws','.git','.venv','__pycache__','raw','agent_evidence','workspaces','runtime_validation','sealed','benchmark_banks'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def permitted(path,curated):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(x in FORBIDDEN for x in p.parts):return False
    if p.suffix in {'.pt','.npz','.npy','.sqlite','.pem','.key'} or p.name in {'auth.json','.env'}:return False
    if path.startswith('research_search/database/'):
        return path in curated and path.startswith('research_search/database/controlled_pilot_v2/') and p.suffix in {'.csv','.json'}
    if path.startswith('data/'):return False
    if path in curated:return p.suffix in {'.csv','.json','.md'}
    if p.parts[0] in CODE_DIRS and p.suffix=='.py':return True
    return path in {'requirements.lock','pyproject.toml'} or path.startswith(('configs/','docs/research_search_cloud_v1/')) and p.suffix in {'.json','.yaml','.md','.txt','.sh'}


def verify(root,manifest):
    root=Path(root).resolve()
    for rel,expected in manifest['files'].items():
        p=PurePosixPath(rel)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe bundle path')
        target=root/rel
        if target.is_symlink() or any(x.is_symlink() for x in target.parents if x!=root) or not target.resolve().is_relative_to(root):raise ValueError('Symlink/escape in bundle')
        if not target.is_file() or digest(target)!=expected:raise ValueError('Changed or missing evidence: '+rel)
    return len(manifest['files'])


def gate_agent_usage(usage,remaining):
    if not isinstance(usage,dict):raise ValueError('Missing provider-reported cloud token telemetry; stop')
    for name in ['input_tokens','output_tokens','cached_input_tokens']:
        if type(usage.get(name)) is not int or usage[name]<0:raise ValueError('Missing provider-reported cloud token telemetry; stop')
    if usage['cached_input_tokens']>usage['input_tokens']:raise ValueError('Invalid cached subset')
    total=usage['input_tokens']+usage['output_tokens']
    if 'total_tokens' in usage:
        if type(usage['total_tokens']) is not int or usage['total_tokens']<total:raise ValueError('Invalid aggregate')
        total=usage['total_tokens']
    if total>remaining:raise ValueError('Reported cloud token budget exhausted')
    return total


def export(destination):
    destination=Path(destination).resolve()
    if destination.exists():raise ValueError('Use a new empty export destination')
    parent=json.loads((ROOT/'research_search/database/integration_v1_1/definition.json').read_text())
    curated=parent['sources'];paths=set(curated)
    for directory in CODE_DIRS:
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*.py') if permitted(str(p.relative_to(ROOT)),set()))
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'configs').glob('*') if p.suffix in {'.yaml','.json'})
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'docs/research_search_cloud_v1').glob('*') if p.is_file())
    paths.update(['requirements.lock','pyproject.toml'])
    rejected=[p for p in paths if not permitted(p,set(curated))]
    if rejected:raise ValueError('Unapproved export entries: '+str(rejected))
    for rel in paths:
        source=ROOT/rel
        if source.is_symlink() or not source.is_file():raise ValueError('Nonregular source: '+rel)
        if rel in curated and digest(source)!=curated[rel]:raise ValueError('Historical source changed: '+rel)
    destination.mkdir(parents=True)
    for rel in sorted(paths):
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
    manifest=dict(version='cloud-preflight-v1',files={p:digest(ROOT/p) for p in sorted(paths)},native_consumed_inputs=curated,model_checkpoints_included=False,credentials_included=False,confirmation_included=False,source_software_status='Original runtime arm64; cloud runtime requires separate validation, never claim original software seal passes on Linux.',research_agent_dispatch='blocked_until_remote_isolation_and_reported_token_receipts_validated',new_training_authorized=False,full72h_campaign_authorized=False)
    (destination/'CLOUD_INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verify(destination,manifest)
    archive=destination.with_suffix('.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(destination.rglob('*')):
            if p.is_file():tar.add(p,arcname=str(p.relative_to(destination)),recursive=False)
    result=dict(destination=str(destination),archive=str(archive),archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,files=len(paths),uncompressed_bytes=sum((destination/p).stat().st_size for p in paths),private_evidence=True,do_not_commit_archive_to_public_repo=True)
    (destination.parent/(destination.name+'_receipt.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result


def preflight(root):
    root=Path(root).resolve();start=time.monotonic();cpu=time.process_time()
    manifest=json.loads((root/'CLOUD_INPUT_MANIFEST.json').read_text());count=verify(root,manifest)
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='srn-cloud-mpl-'))
    import numpy,pandas,torch,yaml,jsonschema,matplotlib
    # Import the unchanged cached evaluator only; no model construction/forward.
    from experiments.research_search_integration_v1_1.cached import execute
    from experiments.research_search_integration_v1_1.evaluation import four_outcome
    assert callable(execute) and callable(four_outcome)
    result=dict(bundle_hashes_pass=True,verified_files=count,cached_evaluator_imports=True,system=platform.system(),architecture=platform.machine(),python=platform.python_version(),software=dict(numpy=numpy.__version__,pandas=pandas.__version__,torch=torch.__version__,matplotlib=matplotlib.__version__),original_arm64_runtime_seal_claimed=False,new_model_forwards=0,new_optimizer_steps=0,remote_agent_dispatch_ready=False,remote_agent_token_telemetry=None,remote_isolation_acceptance='not_yet_validated',next_action='Stop after preflight. Obtain actual cloud per-call token receipts and validate role isolation before any research-agent dispatch.',wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu)
    return result
