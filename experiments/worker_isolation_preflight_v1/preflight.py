"""Bounded Docker fixture. Never imports research evaluation or reads evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

IMAGE_RE = re.compile(r'[a-z0-9]+(?:[./_-][a-z0-9]+)*@sha256:[a-f0-9]{64}')
SOCKETS = ['/var/run/docker.sock', '/run/codex-docker/docker.sock']
REQUIRED_CHECKS = {
    'nonroot_user', 'nonroot_group', 'capabilities_dropped', 'no_new_privileges',
    'seccomp_installed', 'no_host_credentials', 'source_read_allowed',
    'source_write_denied', 'source_truncate_denied', 'source_unlink_denied',
    'root_write_denied', 'private_read_denied', 'symlink_read_denied',
    'hardlink_escape_denied', 'inet_socket_denied', 'unix_socket_denied',
    'exec_denied', 'output_write_allowed', 'temp_write_allowed',
}


def create_command(docker, image, fixture):
    if not IMAGE_RE.fullmatch(image):
        raise ValueError('An immutable name@sha256 image reference is required')
    fixture = Path(fixture)
    if ',' in str(fixture):
        raise ValueError('Comma in fixture path is not supported by Docker mount syntax')
    command = [*docker, 'create', '--pull=never', '--read-only', '--network=none',
               '--cap-drop=ALL', '--user=65534:65534',
               '--security-opt=no-new-privileges:true', '--pids-limit=32',
               '--memory=256m', '--memory-swap=256m', '--cpus=1',
               '--ulimit=cpu=20:20', '--ipc=none',
               '--tmpfs=/tmp:rw,noexec,nosuid,size=16m', '--workdir=/output']
    for source, destination, readonly in [('source', '/worker', True),
                                          ('private', '/private', True),
                                          ('output', '/output', False)]:
        mount = f'type=bind,src={fixture / source},dst={destination}'
        command += ['--mount', mount + (',readonly' if readonly else '')]
    return [*command, image, 'python', '-I', '-B', '/worker/probe.py']


def verify_config(config, fixture, image):
    host = config['HostConfig']
    expected = dict(ReadonlyRootfs=True, NetworkMode='none', Privileged=False,
                    PidsLimit=32, Memory=268435456, MemorySwap=268435456, NanoCpus=1000000000,
                    PidMode='', IpcMode='none')
    if config['Config']['User'] != '65534:65534' or config['Config']['Image'] != image:
        raise ValueError('Unexpected image or user')
    if config['Config'].get('Entrypoint') or config['Config'].get('Cmd') != ['python', '-I', '-B', '/worker/probe.py']:
        raise ValueError('Unexpected container entrypoint or command')
    if any(host.get(k) != v for k, v in expected.items()):
        raise ValueError('Effective Docker restrictions differ from contract')
    if set(host.get('CapDrop') or []) != {'ALL'} or host.get('CapAdd'):
        raise ValueError('Capabilities not fully dropped')
    if host.get('SecurityOpt') != ['no-new-privileges:true']:
        raise ValueError('Missing no-new-privileges')
    if host.get('Ulimits') != [dict(Name='cpu', Soft=20, Hard=20)]:
        raise ValueError('Missing hard CPU limit')
    if host.get('Tmpfs') != {'/tmp': 'rw,noexec,nosuid,size=16m'}:
        raise ValueError('Unexpected writable temporary filesystem')
    expected_mounts = {
        '/worker': (str(fixture / 'source'), False),
        '/private': (str(fixture / 'private'), False),
        '/output': (str(fixture / 'output'), True),
    }
    # Some Docker versions also expose tmpfs in Mounts. Its configuration was
    # already checked above; no other bind/volume/device is admitted here.
    binds = [m for m in config['Mounts'] if not (m['Type'] == 'tmpfs' and m['Destination'] == '/tmp')]
    if len(binds) != len(expected_mounts):
        raise ValueError('Unexpected mount count')
    if {m['Destination'] for m in binds} != set(expected_mounts):
        raise ValueError('Unexpected mount destinations')
    for mount in binds:
        if mount['Type'] != 'bind' or (mount['Source'], mount['RW']) != expected_mounts[mount['Destination']]:
            raise ValueError('Unexpected mount source or write permission')


def verify_checks(checks):
    if set(checks) != REQUIRED_CHECKS or any(v is not True for v in checks.values()):
        raise ValueError('Missing, false or malformed actual restriction checks')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(out, image, socket_path=SOCKETS[0]):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('Existing output must not be overwritten or replayed')
    if any(p.is_symlink() for p in out.parents):
        raise ValueError('Output path may not traverse symlinks')
    if not IMAGE_RE.fullmatch(image) or socket_path not in SOCKETS:
        raise ValueError('Explicit immutable image and reviewed local socket required')
    out.mkdir(parents=True)
    out = out.resolve()
    started = time.monotonic()
    receipt = dict(passed=False, status='blocked', image=image, socket=socket_path,
                   scope='Synthetic container capability test only',
                   scientific_inputs_read=0, new_model_forwards=0,
                   new_optimizer_steps=0, new_agent_calls=0,
                   parent_agent_isolation_certified=False,
                   scientific_execution_authorized=False)
    container_id = None
    logs = []
    # Do not forward tokens, proxies, runtime variables or Docker remote context.
    env = {k: os.environ[k] for k in ['PATH', 'HOME', 'LANG'] if k in os.environ}
    docker = [shutil.which('docker') or 'docker', '--host=unix://' + socket_path]

    def call(arguments, timeout=15, check=True):
        result = subprocess.run(arguments, env=env, capture_output=True, text=True,
                                timeout=timeout, close_fds=True)
        logs.append(dict(command=arguments, returncode=result.returncode,
                         stdout=result.stdout, stderr=result.stderr))
        if check and result.returncode:
            raise RuntimeError('Docker command failed: ' + arguments[len(docker)])
        return result

    try:
        if not shutil.which('docker'):
            raise RuntimeError('Docker CLI unavailable; no daemon is installed or started by this check')
        receipt['daemon_version'] = call([*docker, 'info', '--format', '{{.ServerVersion}}']).stdout.strip()
        fixture = out / 'fixture'
        for directory in ['source', 'private', 'output']:
            (fixture / directory).mkdir(parents=True)
        # The synthetic fixture must be reachable by a rootless daemon's
        # unprivileged container identity. No research paths are mounted.
        out.chmod(0o755); fixture.chmod(0o755)
        (fixture / 'source').chmod(0o755)
        (fixture / 'output').chmod(0o777)
        (fixture / 'private').chmod(0o700)
        probe = Path(__file__).with_name('container_probe.py')
        shutil.copyfile(probe, fixture / 'source/probe.py')
        (fixture / 'source/probe.py').chmod(0o644)
        (fixture / 'source/protected.txt').write_text('UNCHANGED_SOURCE')
        (fixture / 'source/protected.txt').chmod(0o644)
        (fixture / 'private/secret.txt').write_text('SYNTHETIC_PRIVATE_FIXTURE')
        (fixture / 'private/secret.txt').chmod(0o600)
        sentinel_paths = [fixture / 'source/probe.py', fixture / 'source/protected.txt', fixture / 'private/secret.txt']
        before = {str(p.relative_to(out)): digest(p) for p in sentinel_paths}
        save(out / 'FIXTURE_INPUT_HASHES.json', before)
        result = call(create_command(docker, image, fixture))
        container_id = result.stdout.strip()
        if not re.fullmatch(r'[a-f0-9]{64}', container_id):
            raise ValueError('Invalid created container ID')
        configuration = json.loads(call([*docker, 'inspect', container_id]).stdout)[0]
        verify_config(configuration, fixture, image)
        receipt['effective_configuration_verified'] = True
        # Keep a restricted configuration snapshot, never the full Docker env.
        save(out / 'EFFECTIVE_CONFIG.json', {k: configuration[k] for k in ['HostConfig', 'Mounts']})
        result = call([*docker, 'start', '--attach', container_id], timeout=35, check=False)
        state = json.loads(call([*docker, 'inspect', '--format', '{{json .State}}', container_id]).stdout)
        if result.returncode or state.get('ExitCode') != 0 or state.get('Running'):
            raise RuntimeError('Container probe failed; inspect docker_commands.json')
        proof = json.loads(result.stdout)
        verify_checks(proof['checks'])
        if proof.get('passed') is not True:
            raise ValueError('Probe did not pass')
        save(out / 'ACTUAL_RESTRICTION_CHECKS.json', proof)
        after = {str(p.relative_to(out)): digest(p) for p in sentinel_paths}
        if before != after:
            raise ValueError('Protected synthetic fixture changed')
        if (fixture / 'output/allowed.txt').read_text() != 'ALLOWED_OUTPUT':
            raise ValueError('Allowed output not produced')
        receipt.update(passed=True, status='passed_container_fixture',
                       protected_hashes_unchanged=True, checks_passed=len(proof['checks']),
                       platform=proof['platform'], image_id=configuration['Image'])
    except Exception as error:
        receipt.update(error_type=type(error).__name__, errno=getattr(error, 'errno', None), reason=str(error))
    finally:
        if container_id and re.fullmatch(r'[a-f0-9]{64}', container_id):
            try:
                cleanup = call([*docker, 'rm', '--force', container_id], timeout=10, check=False)
                receipt['container_removed'] = cleanup.returncode == 0
                if not receipt['container_removed']:
                    receipt.update(passed=False, status='blocked_cleanup_failed')
            except Exception as error:
                receipt.update(passed=False, status='blocked_cleanup_failed', cleanup_error=str(error))
        receipt['wall_seconds'] = time.monotonic() - started
        save(out / 'docker_commands.json', logs)
        save(out / 'RECEIPT.json', receipt)
        files = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'FILES.json'}
        save(out / 'FILES.json', files)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image', required=True, help='An already pulled name@sha256 image')
    parser.add_argument('--socket', choices=SOCKETS, default=SOCKETS[0])
    args = parser.parse_args()
    receipt = run(args.out, args.image, args.socket)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
