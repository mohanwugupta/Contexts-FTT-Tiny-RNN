"""macOS sandbox for whole untrusted process trees; no permissive fallback."""
import json,os,shutil,signal,subprocess,sys,time,resource
from pathlib import Path
from research_search.common import ROOT,write,sha

def safe_workspace(path,root):
    p=Path(path).absolute();base=Path(root).resolve()
    if p.is_symlink() or not p.resolve().is_relative_to(base) or any(x.is_symlink() for x in p.parents if x!=base):raise ValueError('Workspace/symlink escapes root')
    p.mkdir(parents=True,exist_ok=True);return p.resolve()

def profile(workspace,protected=None,network=False,executable=None):
    w=Path(workspace).resolve();quoted=lambda p:json.dumps(str(Path(p).resolve()))
    protected=list(protected or [ROOT])
    lines=['(version 1)','(allow default)','(deny file-link)',f'(deny file-write* (require-all (require-not (subpath {quoted(w)})) (require-not (literal "/dev/null"))))']
    for p in protected:
        lines.append(f'(deny file-read* (require-all (subpath {quoted(p)}) (require-not (subpath {quoted(w)}))))')
    for p in [Path.home()/'.codex/sessions',Path.home()/'.codex/memories',Path.home()/'.codex/attachments',Path.home()/'.codex/skills',Path.home()/'.codex/plugins']:
        lines.append(f'(deny file-read* (subpath {quoted(p)}))')
    lines.append(f'(deny file-read* (literal {quoted(Path.home()/".codex/AGENTS.md")}))')
    if not network:lines.append('(deny network*)')
    if executable:lines.append(f'(deny process-exec (require-not (literal {quoted(executable)})))')
    return '\n'.join(lines)

def run(command,workspace,protected=None,timeout=60,cpu_seconds=30,network=False,executable=None,env=None,stdin=None,cancel_check=None):
    binary=shutil.which('sandbox-exec')
    if not binary:raise RuntimeError('OS sandbox unavailable; execution blocked')
    w=Path(workspace).resolve();w.mkdir(parents=True,exist_ok=True);policy=profile(w,protected,network,executable);path=w/'sandbox.sb';path.write_text(policy)
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(cpu_seconds)),max(1,int(cpu_seconds))))
    started=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    process=subprocess.Popen([binary,'-f',str(path),*map(str,command)],cwd=w,env=env,stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,preexec_fn=limits)
    timed_out=False;cancelled=False;first=True
    try:
        while True:
            remaining=timeout-(time.monotonic()-started)
            if remaining<=0 or cancel_check and cancel_check():
                timed_out=remaining<=0;cancelled=not timed_out;os.killpg(process.pid,signal.SIGKILL);out,err=process.communicate();break
            try:out,err=process.communicate(stdin if first else None,timeout=min(.5,remaining));break
            except subprocess.TimeoutExpired:first=False
    finally:
        # An agent may not leave detached descendants consuming unaccounted CPU.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(returncode=process.returncode,stdout=out,stderr=err,timed_out=timed_out,cancelled=cancelled,wall_seconds=time.monotonic()-started,cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,policy_sha=sha(path))

def certify(directory):
    """Actual denied reads/writes (including symlinks), not a mocked policy test."""
    base=Path(directory).resolve();base.mkdir(parents=True,exist_ok=True);w=base/'workspace';vault=base/'vault';w.mkdir(exist_ok=True);vault.mkdir(exist_ok=True);secret=vault/'confirmation.json';secret.write_text('sealed-fixture');evaluator=vault/'evaluator.py';evaluator.write_text('unchanged')
    link=w/'leak';
    if not link.exists():link.symlink_to(secret)
    code='''import pathlib,json,sys
w=pathlib.Path(sys.argv[1]);secret=pathlib.Path(sys.argv[2]);evaluator=pathlib.Path(sys.argv[3]);r={}
for name,path in [('private_read_denied',secret),('symlink_read_denied',w/'leak')]:
 try:path.read_text();r[name]=False
 except PermissionError:r[name]=True
try:evaluator.write_text('tampered');r['evaluator_write_denied']=False
except PermissionError:r['evaluator_write_denied']=True
try:
 import os
 os.link(secret,w/'hardlink');(w/'hardlink').read_text();r['hardlink_escape_denied']=False
except PermissionError:r['hardlink_escape_denied']=True
(w/'allowed.txt').write_text('allowed');r['workspace_write_allowed']=True
print(json.dumps(r))'''
    r=run([sys.executable,'-c',code,w,secret,evaluator],w,[ROOT,vault],timeout=15,cpu_seconds=5)
    if r['returncode']!=0:return dict(sandbox_available='sandbox_apply:' not in r['stderr'],passed=False,reason=r['stderr'],returncode=r['returncode'])
    flags=json.loads(r['stdout']);assert evaluator.read_text()=='unchanged' and secret.read_text()=='sealed-fixture'
    return dict(sandbox_available=True,passed=all(flags.values()),**flags,policy_sha=r['policy_sha'])
