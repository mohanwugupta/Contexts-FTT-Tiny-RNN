"""Verified CA file-secret build, then unchanged synthetic container smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PEM = re.compile(br'-----BEGIN CERTIFICATE-----\s+.*?-----END CERTIFICATE-----', re.S)
SOCKETS = ['/var/run/docker.sock', '/run/codex-docker/docker.sock']


def certificates(data):
    if len(data) > 4 * 1024 * 1024 or b'PRIVATE KEY' in data:
        raise ValueError('Certificate-only bundle required; keys/oversize input rejected')
    blocks = PEM.findall(data)
    residue = PEM.sub(b'', data)
    if not blocks or any(line.strip() and not line.lstrip().startswith(b'#') for line in residue.splitlines()):
        raise ValueError('Invalid certificate bundle')
    result = []
    for block in blocks:
        try:
            text = block.decode('ascii')
            der = ssl.PEM_cert_to_DER_cert(text)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.load_verify_locations(cadata=text)
        except (ValueError, UnicodeError, ssl.SSLError) as error:
            raise ValueError('Malformed X509 certificate') from error
        result.append((ssl.DER_cert_to_PEM_cert(der).encode(), hashlib.sha256(der).hexdigest()))
    return result


def combined_bundle(system_path, session_path=None):
    paths = [Path(system_path)] + ([Path(session_path)] if session_path is not None else [])
    unique = {}
    for path in paths:
        if not path.is_file() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError('Missing or oversized configured CA bundle')
        for pem, fingerprint in certificates(path.read_bytes()):
            unique.setdefault(fingerprint, pem)
    bundle = b''.join(unique.values())
    return bundle, dict(unique_certificates=len(unique), sha256=hashlib.sha256(bundle).hexdigest(),
                        certificate_sha256=list(unique), source_files=len(paths), TLS_verification=True)


def configured_bundle(explicit=None):
    if explicit is not None:
        return combined_bundle(explicit)
    system = Path('/etc/ssl/certs/ca-certificates.crt')
    if not system.is_file():
        default = ssl.get_default_verify_paths().cafile
        if not default:
            raise ValueError('No system CA bundle; supply --ca-bundle explicitly')
        system = Path(default)
    bundle, metadata = combined_bundle(system)
    sources = [system]
    # Add only explicitly configured local trust stores; never fetch a new CA.
    for name in ['CODEX_PROXY_CERT', 'SSL_CERT_FILE', 'REQUESTS_CA_BUNDLE', 'PIP_CERT']:
        value = os.environ.get(name)
        if value:
            path = Path(value)
            if path not in sources:
                sources.append(path)
    unique = {}
    for path in sources:
        data, _ = combined_bundle(path)
        for pem, fingerprint in certificates(data):
            unique.setdefault(fingerprint, pem)
    bundle = b''.join(unique.values())
    metadata.update(unique_certificates=len(unique), sha256=hashlib.sha256(bundle).hexdigest(),
                    certificate_sha256=list(unique), source_files=len(sources))
    return bundle, metadata


def prepare_context(out):
    from experiments.cloud_confidence_container_v2.launcher import build_context
    out = Path(out)
    build_context(out)
    (out / 'Dockerfile').write_bytes(Path(__file__).with_name('Dockerfile').read_bytes())
    metadata = dict(version='runtime-tls-v1', evidence_included=False, CA_in_context=False,
                    files={name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                           for name in ['Dockerfile', 'requirements.txt']})
    (out / 'BUILD_CONTEXT.json').write_text(json.dumps(metadata, indent=2) + '\n')


def docker_build_command(socket_path, context, image_id, ca_file):
    if socket_path not in SOCKETS or any(',' in str(p) or '\n' in str(p) for p in [context, image_id, ca_file]):
        raise ValueError('Unreviewed socket or unsupported build path')
    return ['docker', '--host=unix://' + socket_path, 'build', '--no-cache',
            '--secret', 'id=proxy_ca,src=' + str(ca_file), '--iidfile', str(image_id), str(context)]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(out, socket_path=SOCKETS[0], ca_bundle=None):
    out = Path(out).absolute()
    if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):
        raise ValueError('Use a new output directory without symlinks')
    out.mkdir(parents=True)
    out = out.resolve()
    start = time.monotonic()
    receipt = dict(passed=False, phase='CA_preparation', scientific_inputs_read=0,
                   research_agent_calls=0, new_model_forwards=0, new_optimizer_steps=0,
                   parent_agent_isolation_certified=False, TLS_verification=True,
                   CA_secret_copy_policy='BuildKit mount only; not copied', retry=False)
    # Preserve managed proxies, registry configuration and CA settings. Remove
    # only remote Docker selectors; --host always names the reviewed local socket.
    env = os.environ.copy()
    for name in ['DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH']:
        env.pop(name, None)
    env['DOCKER_BUILDKIT'] = '1'
    temporary_path = None
    try:
        bundle, metadata = configured_bundle(ca_bundle)
        save(out / 'CA_METADATA.json', metadata)
        prepare_context(out / 'build-context')
        receipt['phase'] = 'Docker_build'
        # A sibling temporary directory, never beneath the Docker build context.
        with tempfile.TemporaryDirectory(prefix='srn-ca-build-', dir=out.parent) as private:
            temporary_path = Path(private)
            temporary_path.chmod(0o700)
            cert = temporary_path / 'ca.pem'; cert.write_bytes(bundle); cert.chmod(0o600)
            command = docker_build_command(socket_path, out / 'build-context', out / 'image-id', cert)
            with (out / 'image-build.log').open('w') as log:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            receipt['build_returncode'] = completed.returncode
            if completed.returncode:
                raise RuntimeError('Verified TLS image build failed; preserve image-build.log')
        receipt['temporary_CA_removed'] = not temporary_path.exists()
        receipt['CA_mount_and_certificate_env_absence_asserted_in_build'] = True
        image = (out / 'image-id').read_text().strip()
        if not re.fullmatch('sha256:[a-f0-9]{64}', image):
            raise ValueError('Invalid immutable image ID')
        receipt['image_id'] = image
        receipt['phase'] = 'synthetic_smoke'
        from experiments.cloud_confidence_container_v2.launcher import run as smoke
        result = smoke(out / 'smoke', image, 'smoke', socket_path)
        receipt['smoke_passed'] = result['passed']
        receipt['passed'] = result['passed']
        receipt['phase'] = 'completed' if result['passed'] else 'synthetic_smoke_blocked'
    except Exception as error:
        receipt.update(error_type=type(error).__name__, reason=str(error))
    finally:
        if temporary_path is not None:
            receipt['temporary_CA_removed'] = not temporary_path.exists()
        receipt['wall_seconds'] = time.monotonic() - start
        save(out / 'TLS_BUILD_RECEIPT.json', receipt)
        files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in out.rglob('*') if p.is_file() and not p.is_symlink() and p.name != 'FILES.json'}
        save(out / 'FILES.json', files)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--socket', choices=SOCKETS, default=SOCKETS[0])
    p.add_argument('--ca-bundle', type=Path)
    args = p.parse_args()
    result = run(args.out, args.socket, args.ca_bundle)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 3


if __name__ == '__main__':raise SystemExit(main())
