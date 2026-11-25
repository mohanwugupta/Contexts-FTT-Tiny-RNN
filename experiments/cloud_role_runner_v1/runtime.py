"""Prepare pins/source only. No authentication read or network/model call."""
import json,os,shutil,ssl
from pathlib import Path
from .contract import sha,save
from .docker import IMAGE,SOCKETS,proxy_environment_sha
HERE=Path(__file__).resolve().parent

def resolve_cli(explicit=None):
 entry=Path(explicit or shutil.which('codex') or '').resolve()
 def elf(p):
  if not p.is_file() or not os.access(p,os.X_OK):return False
  with p.open('rb') as f:head=f.read(20)
  return len(head)==20 and head[:6]==b'\x7fELF\x02\x01' and int.from_bytes(head[18:20],'little')==62
 if elf(entry):return entry,entry
 package=entry.parent.parent
 manifest=package/'package.json'
 if entry.name!='codex.js' or not manifest.is_file() or json.loads(manifest.read_text()).get('name')!='@openai/codex':raise ValueError('Installed x86_64 Linux Codex ELF required; no download or fallback')
 roots=[package,package/'node_modules/@openai/codex-linux-x64',package.parent/'codex-linux-x64']
 found=set()
 for root in roots:
  for triplet in ['x86_64-unknown-linux-musl','x86_64-unknown-linux-gnu']:
   p=root/'vendor'/triplet/'codex/codex'
   if elf(p):found.add(p.resolve())
 if len(found)!=1:raise ValueError('Installed npm CLI binary missing or ambiguous; pass exact --codex ELF path')
 return found.pop(),entry

def prepare(out,image,codex=None,ca=None,socket=SOCKETS[0],synthetic=False):
 out=Path(out).absolute()
 if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):raise ValueError('Fresh canonical runtime path required')
 if not IMAGE.fullmatch(image) or socket not in SOCKETS:raise ValueError('Explicit accepted image/socket required')
 binary=entry=None
 if not synthetic:binary,entry=resolve_cli(codex)
 from experiments.cloud_runtime_tls_v1.build import configured_bundle
 pem,ca_metadata=configured_bundle(ca)
 out.mkdir(parents=True,mode=0o700);source=out/'source';source.mkdir(mode=0o755)
 for p in HERE.glob('*.py'):shutil.copyfile(p,source/p.name);(source/p.name).chmod(0o644)
 (source/'ca.pem').write_bytes(pem);(source/'ca.pem').chmod(0o644)
 for name in ['requirements.toml','managed_config.toml']:
  policy=Path('/etc/codex')/name
  if policy.is_file():shutil.copyfile(policy,source/name);(source/name).chmod(0o644)
 if binary:shutil.copyfile(binary,source/'codex');(source/'codex').chmod(0o555)
 value=dict(version='linux-role-runtime-v1',TLS_verification=True,CA_metadata=ca_metadata,directory=str(out),source=str(source),image=image,socket=socket,synthetic=synthetic,codex_path=str(binary) if binary else None,codex_sha256=sha(binary) if binary else None,launcher_path=str(entry) if entry else None,launcher_sha256=sha(entry) if entry else None,source_hashes={p.name:sha(p) for p in source.iterdir()},proxy_environment_sha=proxy_environment_sha(),live_usage_verified=False,parent_agent_isolation_certified=False,new_agent_calls=0)
 save(out/'RUNTIME.json',value);return value
