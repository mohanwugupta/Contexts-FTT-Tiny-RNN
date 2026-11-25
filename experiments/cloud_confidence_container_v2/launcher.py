"""Explicit container lifecycle with separate smoke/approved-execution modes."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from experiments.worker_isolation_preflight_v1.preflight import SOCKETS, create_command
from .contract import ROOT, sha, verify_bundle, validate_authorization
from .staging import stage, write_json

IMAGE = re.compile(r'(?:[a-z0-9]+(?:[./_-][a-z0-9]+)*@)?sha256:[a-f0-9]{64}')
CMD = ['python', '-I', '-B', '/workspace/experiments/cloud_confidence_container_v2/worker.py']


def command(docker, image, job, mode):
    if not IMAGE.fullmatch(image) or mode not in ['smoke', 'execute']:
        raise ValueError('Immutable image and explicit mode required')
    base = create_command(docker, 'python@sha256:' + '0' * 64, job / 'fixture')[:-5]
    replacement = {'--memory=256m': '--memory=2g', '--memory-swap=256m': '--memory-swap=2g',
                   '--ulimit=cpu=20:20': '--ulimit=cpu=120:120',
                   '--tmpfs=/tmp:rw,noexec,nosuid,size=16m': '--tmpfs=/tmp:rw,noexec,nosuid,size=128m'}
    base = [replacement.get(x, x) for x in base]
    return [*base, '--mount', f'type=bind,src={job / "source"},dst=/workspace,readonly',
            image, *CMD, '--mode', mode]


def verify_config(configuration, job, image, mode):
    c, h = configuration['Config'], configuration['HostConfig']
    if c['User'] != '65534:65534' or c['Image'] != image or c.get('Entrypoint') or c['Cmd'] != CMD + ['--mode', mode]:
        raise ValueError('Wrong container command, image or identity')
    required = dict(ReadonlyRootfs=True, NetworkMode='none', Privileged=False,
                    PidsLimit=32, Memory=2147483648, MemorySwap=2147483648,
                    NanoCpus=1000000000, PidMode='', IpcMode='none',
                    SecurityOpt=['no-new-privileges:true'],
                    Ulimits=[dict(Name='cpu', Soft=120, Hard=120)],
                    Tmpfs={'/tmp': 'rw,noexec,nosuid,size=128m'})
    if any(h.get(k) != v for k, v in required.items()) or set(h.get('CapDrop') or []) != {'ALL'} or h.get('CapAdd'):
        raise ValueError('Effective container protections differ')
    mounts = {
        '/worker': (str(job / 'fixture/source'), False),
        '/private': (str(job / 'fixture/private'), False),
        '/output': (str(job / 'fixture/output'), True),
        '/workspace': (str(job / 'source'), False),
    }
    actual = [m for m in configuration['Mounts'] if not (m['Type'] == 'tmpfs' and m['Destination'] == '/tmp')]
    if len(actual) != 4 or {m['Destination'] for m in actual} != set(mounts):
        raise ValueError('Unexpected container mounts')
    for m in actual:
        if m['Type'] != 'bind' or (m['Source'], m['RW']) != mounts[m['Destination']]:
            raise ValueError('Wrong source or output mount')


def build_context(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Existing build context must not be overwritten')
    out.mkdir(parents=True)
    original = ROOT / 'docs/research_search_cloud_v1/requirements-cloud.txt'
    shutil.copyfile(original, out / 'requirements.txt')
    shutil.copyfile(Path(__file__).with_name('Dockerfile'), out / 'Dockerfile')
    write_json(out / 'BUILD_CONTEXT.json', dict(files={p.name: sha(p) for p in out.iterdir()},
        evidence_included=False, source='Pinned Cloud requirements; CPU-only torch2.8.0',
        purpose='Runtime dependencies only; no worker or evidence is baked into the image'))


def run(out, image, mode, socket_path=SOCKETS[0], evidence_root=None,
        selection=None, proposal=None, authorization=None):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise ValueError('Use a new output path without symlinks')
    if not IMAGE.fullmatch(image) or socket_path not in SOCKETS or mode not in ['smoke', 'execute']:
        raise ValueError('Explicit reviewed backend/image/mode required')
    approved = None
    if mode == 'execute':
        if any(x is None for x in [evidence_root, selection, proposal, authorization]):
            raise ValueError('Real execution requires explicit source/request/authorization')
        approved = json.loads(Path(authorization).read_text())
        validate_authorization(approved)
        for name, path in [('proposal', proposal), ('selection', selection)]:
            if sha(path) != approved[name + '_sha256']:
                raise ValueError('Selected request differs from authorization')
    elif any(x is not None for x in [evidence_root, selection, proposal, authorization]):
        raise ValueError('Smoke must not receive scientific data or authorization')
    out.mkdir(parents=True); out = out.resolve(); out.chmod(0o755)
    start = time.monotonic(); cid = None; logs = []; lock = None
    receipt = dict(passed=False, mode=mode, image=image, socket=socket_path,
                   new_agent_calls=0, reported_model_token_usage=None,
                   full_parent_agent_isolation_certified=False, no_automatic_retry=True)
    env = {k: os.environ[k] for k in ['PATH', 'HOME', 'LANG'] if k in os.environ}
    docker = [shutil.which('docker') or 'docker', '--host=unix://' + socket_path]

    def call(argv, timeout=15, check=True):
        result = subprocess.run(argv, env=env, text=True, capture_output=True,
                                timeout=timeout, close_fds=True)
        logs.append(dict(command=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        if check and result.returncode:
            raise RuntimeError('Docker lifecycle command failed')
        return result

    try:
        import fcntl
        # Lock lives outside container-writable output and spans its lifecycle.
        lock = (out.parent / '.container_worker.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not shutil.which('docker'):
            raise RuntimeError('Docker CLI unavailable')
        receipt['daemon_version'] = call([*docker, 'info', '--format', '{{.ServerVersion}}']).stdout.strip()
        hashes = stage(out / 'source', evidence_root, selection, proposal, authorization)
        fixture = out / 'fixture'
        for part in ['source', 'private', 'output']:
            (fixture / part).mkdir(parents=True)
        fixture.chmod(0o755); (fixture / 'source').chmod(0o755)
        (fixture / 'private').chmod(0o700); (fixture / 'output').chmod(0o777)
        shutil.copyfile(ROOT / 'experiments/worker_isolation_preflight_v1/container_probe.py', fixture / 'source/probe.py')
        (fixture / 'source/probe.py').chmod(0o644)
        (fixture / 'source/protected.txt').write_text('UNCHANGED_SOURCE'); (fixture / 'source/protected.txt').chmod(0o644)
        (fixture / 'private/secret.txt').write_text('SYNTHETIC_PRIVATE_FIXTURE'); (fixture / 'private/secret.txt').chmod(0o600)
        sentinels = [fixture / 'source/probe.py', fixture / 'source/protected.txt', fixture / 'private/secret.txt']
        before = [sha(p) for p in sentinels]
        cid = call(command(docker, image, out, mode)).stdout.strip()
        if not re.fullmatch('[a-f0-9]{64}', cid):
            raise ValueError('Invalid container ID')
        configuration = json.loads(call([*docker, 'inspect', cid]).stdout)[0]
        verify_config(configuration, out, image, mode)
        receipt['image_id'] = configuration['Image']
        timeout = validate_authorization(approved) if approved is not None else 300
        result = call([*docker, 'start', '--attach', cid], timeout=timeout, check=False)
        state = json.loads(call([*docker, 'inspect', '--format', '{{json .State}}', cid]).stdout)
        if result.returncode or state.get('ExitCode') != 0 or state.get('Running'):
            raise RuntimeError('Worker blocked or failed; preserve its logs')
        output = fixture / 'output'
        expected = 'SMOKE.json' if mode == 'smoke' else 'RESULT.json'
        if not (output / expected).is_file():
            raise ValueError('Missing worker completion artifact')
        proof = json.loads((output / 'WORKER_ISOLATION.json').read_text())
        from experiments.worker_isolation_preflight_v1.preflight import verify_checks
        verify_checks(proof['checks'])
        if proof.get('passed') is not True or proof.get('evidence_write_denied') is not True:
            raise ValueError('Actual evidence isolation failed')
        usage = json.loads((output / 'PROCESS_USAGE.json').read_text())
        if usage['CPU_seconds'] > 120:
            raise ValueError('Worker CPU ceiling exceeded')
        verify_bundle(out / 'source', hashes)
        if before != [sha(p) for p in sentinels]:
            raise ValueError('Protected fixture changed')
        receipt.update(passed=True, CPU_seconds=usage['CPU_seconds'], CPU_cap_seconds=120,
                       actual_restriction_checks_passed=19, evidence_write_denied=True,
                       staged_input_hashes_unchanged=True, new_model_forwards=0,
                       new_optimizer_steps=0)
    except Exception as error:
        receipt.update(error_type=type(error).__name__, reason=str(error))
    finally:
        if cid and re.fullmatch('[a-f0-9]{64}', cid):
            try:
                receipt['container_removed'] = call([*docker, 'rm', '--force', cid], timeout=10, check=False).returncode == 0
                if not receipt['container_removed']:
                    receipt['passed'] = False
            except Exception as error:
                receipt.update(passed=False, cleanup_error=str(error))
        if lock is not None:
            lock.close()
        receipt['wall_seconds'] = time.monotonic() - start
        write_json(out / 'docker_commands.json', logs); write_json(out / 'EXECUTION_RECEIPT.json', receipt)
        write_json(out / 'FILES.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'FILES.json'})
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build-context', 'smoke', 'execute'])
    p.add_argument('--out', type=Path, required=True); p.add_argument('--image')
    p.add_argument('--socket', choices=SOCKETS, default=SOCKETS[0])
    for name in ['evidence-root', 'selection', 'proposal', 'authorization']:
        p.add_argument('--' + name, type=Path)
    a = p.parse_args()
    if a.action == 'build-context':
        build_context(a.out); return 0
    receipt = run(a.out, a.image or '', a.action, a.socket, a.evidence_root, a.selection, a.proposal, a.authorization)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 3


if __name__ == '__main__':raise SystemExit(main())
