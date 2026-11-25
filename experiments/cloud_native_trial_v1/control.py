"""Frozen scope, source closure and preparation. No model construction here."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
LIMITS=dict(tokens=0,cpu_seconds=18000,optimizer_steps=159120,wall_seconds=21600,proposals=1,active_branches=1)
META_CEILING=7680
DEFAULT=dict(model=dict(hidden_dim=8,activation='linear',nominal_scale=5.,recurrent_norm=.9,centered=False,cue_timing='continuous',gate=False),prior=dict(episodes=1920,kind='P2',cue_rotation=False),target=dict(steps=330,readout_initial_multiplier=1.,update_rule='native'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n');path.chmod(0o644)

def configs():
    a=copy.deepcopy(DEFAULT);b=copy.deepcopy(a);b['prior']['cue_rotation']=True
    return a,b

def definition():
    a,b=configs()
    return dict(version='cloud-native-trial-v1',hypothesis='prior-cue-coverage',
      hypothesis_status='registered open hypothesis, not a newly discovered mechanism',
      control=a,candidate=b,seeds=[21000,21001],tasks=4,limits=copy.deepcopy(LIMITS),
      meta_ceiling=META_CEILING,agent_calls=0,confirmation_evaluations=0,
      automatic_promotion=False,original_72h_campaign_modified=False,
      claim='development only; n=2 independent seeds; late functional and screen status unassessed',
      success='retain original scorer flags; additionally same-network original early phenotype and useful A/B/C/Mixed-average/NoContext at330',
      old_token_holds=117000,old_provider_usage=None,parent_agent_isolation_certified=False)

def validate_definition(obj):
    if obj!=definition():raise ValueError('Trial scope, criteria or budgets changed; create a new version')

def source_files():
    """Static local-import closure, including imports inside functions; never data dirs."""
    pending=['experiments.cloud_native_trial_v1.worker','experiments.cloud_native_trial_v1.launcher',
             'research_search.runners.evaluate','research_search.evaluation.scoring']
    seen=set();files={}
    def locate(name):
        for root in [ROOT,ROOT/'src']:
            for path in [root/Path(*name.split('.')).with_suffix('.py'),root/Path(*name.split('.'))/'__init__.py']:
                if path.is_file():return path
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);path=locate(name)
        if path is None:continue
        rel=str(path.relative_to(ROOT))
        if any(x in Path(rel).parts for x in ['database','sealed','data','.git']):raise ValueError('Unsafe source closure')
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=ROOT and ROOT in p.parents):raise ValueError('Symlinked source forbidden')
        files[rel]=sha(path)
        package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):pending.extend(n.name for n in node.names)
            elif isinstance(node,ast.ImportFrom):
                base=node.module or ''
                if node.level:
                    parts=package.split('.');base='.'.join(parts[:len(parts)-node.level+1]+([base] if base else []))
                if base:pending.append(base)
                pending.extend(base+'.'+n.name for n in node.names if n.name!='*')
        for i in range(1,len(name.split('.'))):pending.append('.'.join(name.split('.')[:i]))
    seal=json.loads((ROOT/'research_search/audit/evaluator_seal.json').read_text())
    for rel,h in seal['code'].items():
        if sha(ROOT/rel)!=h:raise ValueError('Original scientific code changed: '+rel)
        files[rel]=h
    for rel in ['configs/novelty_registry_v1.json','research_search/audit/evaluator_seal.json']:
        files[rel]=sha(ROOT/rel)
    assets=HERE/'assets/FILES.json'
    if assets.exists():files[str(assets.relative_to(ROOT))]=sha(assets)
    return dict(sorted(files.items()))

def verify_source():
    expected=json.loads((HERE/'SOURCE_FILES.json').read_text())
    if source_files()!=expected:raise ValueError('Pinned source closure changed; do not launch')
    return expected

def prepare(out):
    out=Path(out).absolute()
    if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):raise ValueError('New nonsymlink output directory required')
    hashes=verify_source();assets=json.loads((HERE/'assets/FILES.json').read_text())
    for rel,h in assets.items():
        if sha(HERE/'assets'/rel)!=h:raise ValueError('Changed registered task library '+rel)
    out.mkdir(parents=True);out.chmod(0o755)
    source=out/'source';source.mkdir()
    for rel,h in hashes.items():
        p=source/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,p);p.chmod(0o644)
    for name in [*assets,'FILES.json']:
        p=source/'trial_assets'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/'assets'/name,p);p.chmod(0o644)
    for p in source.rglob('*'):
        if p.is_dir():p.chmod(0o755)
    save(source/'STAGED_FILES.json',{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()})
    save(out/'definition.json',definition());save(source/'definition.json',definition())
    for part in ['fixture/source','fixture/private','native','exports']:(out/part).mkdir(parents=True)
    for p in [out/'fixture',out/'fixture/source']:p.chmod(0o755)
    (out/'fixture/private').chmod(0o700);(out/'native').chmod(0o777);(out/'exports').chmod(0o777)
    save(out/'fixture/source/protected.txt',None)
    (out/'fixture/source/protected.txt').write_text('UNCHANGED_SOURCE');(out/'fixture/private/secret.txt').write_text('SYNTHETIC_PRIVATE_FIXTURE');(out/'fixture/private/secret.txt').chmod(0o600)
    return dict(prepared=True,training_started=False,out=str(out),limits=LIMITS,source_files=len(hashes),next_action='explicit run using accepted immutable image')
