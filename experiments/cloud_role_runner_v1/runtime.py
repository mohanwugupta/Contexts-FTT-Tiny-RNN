"""Prepare pins/source only. No authentication read or network/model call."""
import json,shutil,ssl
from pathlib import Path
from .contract import sha,save
from .docker import IMAGE,SOCKETS,proxy_environment_sha
HERE=Path(__file__).resolve().parent

def prepare(out,image,codex=None,ca=None,socket=SOCKETS[0],synthetic=False):
 out=Path(out).absolute()
 if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):raise ValueError('Fresh canonical runtime path required')
 if not IMAGE.fullmatch(image) or socket not in SOCKETS:raise ValueError('Explicit accepted image/socket required')
 binary=None
 if not synthetic:
  binary=Path(codex or shutil.which('codex') or '').resolve()
  head=binary.read_bytes()[:20]
  if len(head)<20 or head[:6]!=b'\x7fELF\x02\x01' or int.from_bytes(head[18:20],'little')!=62:raise ValueError('Reviewed x86_64 Linux Codex executable required')
 ca=Path(ca or ssl.get_default_verify_paths().cafile or '/etc/ssl/certs/ca-certificates.crt').resolve();pem=ca.read_bytes()
 if b'PRIVATE KEY' in pem:raise ValueError('CA bundle contains a private key')
 ssl.create_default_context(cafile=str(ca))
 out.mkdir(parents=True,mode=0o700);source=out/'source';source.mkdir(mode=0o755)
 for p in HERE.glob('*.py'):shutil.copyfile(p,source/p.name);(source/p.name).chmod(0o644)
 shutil.copyfile(ca,source/'ca.pem');(source/'ca.pem').chmod(0o644)
 if binary:shutil.copyfile(binary,source/'codex');(source/'codex').chmod(0o555)
 value=dict(version='linux-role-runtime-v1',directory=str(out),source=str(source),image=image,socket=socket,synthetic=synthetic,codex_path=str(binary) if binary else None,codex_sha256=sha(binary) if binary else None,source_hashes={p.name:sha(p) for p in source.iterdir()},proxy_environment_sha=proxy_environment_sha(),live_usage_verified=False,parent_agent_isolation_certified=False,new_agent_calls=0)
 save(out/'RUNTIME.json',value);return value
