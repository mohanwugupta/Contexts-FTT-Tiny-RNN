"""Local acceptance and immutable scientific identities. Never starts a campaign."""
import json,os,platform,sys,tempfile
import numpy as np,pandas as pd,torch
from .common import *

def software():return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,platform=platform.machine(),execution_dtype='float32',device='CPU',threads=1)

def code_files():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.rglob('*.py')) if not any(x in ['database','audit','experiments','__pycache__'] for x in p.relative_to(PACKAGE).parts)}

def freeze():
    codes=code_files();configuration={str(p.relative_to(ROOT)):sha(p) for p in (PACKAGE/'config').glob('*.yaml')};sources=read(PACKAGE/'history/SOURCE_MANIFEST.json')['files'].copy();base=ROOT/'results/a06_novel_cue_followup/v1';m=read(base/'manifest.json');sources|=m['inputs']|m['code']|{str((base/n).relative_to(ROOT)):h for n,h in m['outputs'].items()};sources[str((base/'manifest.json').relative_to(ROOT))]=sha(base/'manifest.json')
    for n,h in sources.items():
        if sha(ROOT/n)!=h:raise ValueError('Changed historical source '+n)
    roles={str(p.relative_to(ROOT)):sha(p) for p in (PACKAGE/'agents').glob('*.md')};value=dict(code=codes,configuration=configuration,roles=roles,sources=sources,software=software(),runner_hash=fingerprint(codes),metrics_hash=fingerprint({n:h for n,h in codes.items() if '/evaluation/' in n}),scientific_objective='ordinary native CE+.2 detached execution; no splitting/transfer reward',confirmation_commitment=read(PACKAGE/'audit/registered_bank_commitments.json'))
    write(PACKAGE/'audit/evaluator_seal.json',value,True);return value

def verify_seal():
    m=read(PACKAGE/'audit/evaluator_seal.json')
    for group in ['code','configuration','roles','sources']:
        for n,h in m[group].items():
            if sha(ROOT/n)!=h:raise ValueError('Changed sealed '+group+': '+n)
    if code_files()!=m['code']:raise ValueError('Scientific code file set changed')
    if software()!=m['software']:raise ValueError('Library/runtime version changed; versioned revalidation required')
    return dict(passed=True,files=sum(len(m[k]) for k in ['code','configuration','roles','sources']),runner_hash=m['runner_hash'])

def acceptance():
    # Reuses completed accepted fixtures. Never reruns attempted optimization.
    from .history.build import build
    from .evaluation.parity import run as parity
    from .evaluation.training_parity import run as training
    from .evaluation.development import register
    from .runners.benchmark import benchmark
    from .runners.fixture import run as fixture
    from .runners.native_fixture import run as native
    from .runners.worker_fixture import run as worker
    from .orchestration.isolation import certify
    build();a,b=register(PACKAGE/'database/benchmark_banks');p=parity();t=training();bench=benchmark();f=fixture();n=native();w=worker()
    proof=PACKAGE/'audit/isolation_final.json'
    if proof.exists():isolation=read(proof)
    else:
        isolation=certify(PACKAGE/'audit/isolation_final')
        # Keep unavailability/failure separate from a passing accepted proof.
        write(PACKAGE/'audit/isolation_latest_attempt.json',isolation)
        if isolation.get('passed'):write(proof,isolation,True)
    r=dict(infrastructure_passed=all(x.get('passed',False) for x in [p,t,bench,f,n,w,isolation]),scientific_campaign_started=False,real_agent_model_calls=0,live_backend_verified=False,live_backend_status='blocked_by_automatic_approval_review; explicit user approval needed for fixed JSON-only ping',unattended_token_ceiling_guaranteed=False,manual_checkpoints_required=True,implementation_setup_tokens=None,setup_token_planning_reserve=100000,baseline_models=p['models'],baseline_states=p['states'],baseline_native_forwards=480,baseline_double_forwards=480,practice_losses_exact=t['checked'],practice_states_missing=t['missing'],actual_fixture_optimizer_calls=bench['native_steps']+f['actual_optimizer_steps']+n['actual_optimizer_steps'],simulated_uncertain_reservations=1,development_tasks=len(a['tasks']),development_seeds=len(a['seeds']),confirmation_tasks=len(b['tasks']),confirmation_seeds=len(b['seeds']),budgets=read(PACKAGE/'config/budgets.yaml'),isolation=isolation,source_and_evaluator_sealed=False)
    write(PACKAGE/'audit/readiness_local.json',r,True);return r

def readiness():
    r=read(PACKAGE/'audit/readiness_local.json');proof=PACKAGE/'runtime_validation/backend_v2/proof.json'
    if proof.exists():
        p=read(proof)
        if p.get('runner_hash')!=read(PACKAGE/'audit/evaluator_seal.json')['runner_hash']:raise ValueError('Backend proof belongs to another code identity')
        r=r|dict(live_backend_verified=bool(p.get('passed')),live_backend_status=p.get('status'),real_agent_model_calls=p.get('real_agent_fixture_calls'))
    return r|dict(source_and_evaluator_sealed=(PACKAGE/'audit/evaluator_seal.json').exists(),ready_for_scientific_launch=bool(r['infrastructure_passed'] and r['live_backend_verified']))

def seal_artifacts():
    verify_seal();paths=[]
    for p in sorted(PACKAGE.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:continue
        rel=p.relative_to(PACKAGE)
        if rel.parts[0] in ['database','experiments','runtime_validation'] or p.name=='IMPLEMENTATION_MANIFEST.json':continue
        paths.append(p)
    write(PACKAGE/'IMPLEMENTATION_MANIFEST.json',dict(status='local infrastructure complete; live backend blocked; scientific campaign not started',files={str(p.relative_to(PACKAGE)):sha(p) for p in paths},evaluator_seal_sha=sha(PACKAGE/'audit/evaluator_seal.json')),True)
    return dict(passed=True,files=len(paths))

def verify_artifacts():
    m=read(PACKAGE/'IMPLEMENTATION_MANIFEST.json')
    for n,h in m['files'].items():
        if sha(PACKAGE/n)!=h:raise ValueError('Implementation evidence changed '+n)
    return verify_seal()|dict(implementation_artifacts=len(m['files']))
