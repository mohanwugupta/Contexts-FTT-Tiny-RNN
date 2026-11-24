"""External private runtime and unchanged repository protections for Codex."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import selectors
import shutil
import signal
import subprocess
import time

from research_search.common import ROOT,sha
from research_search.orchestration.agent_backend import DISABLED
from research_search.orchestration.isolation import profile as original_profile


@dataclass(frozen=True)
class Runtime:
    workspace: Path
    home: Path
    tmp: Path
    state: Path
    logs: Path
    auth_source: Path


def create_runtime(workspace,auth_source):
    lexical=Path(workspace).absolute()
    if lexical.is_symlink() or any(p.is_symlink() for p in lexical.parents):
        raise ValueError('Runtime paths must not traverse symlinks')
    w=lexical.resolve()
    if w.is_relative_to(ROOT) or ROOT.is_relative_to(w):
        raise ValueError('Runtime must be outside the repository and its ancestors')
    auth=Path(auth_source).absolute()
    if auth.is_symlink() or not auth.is_file():
        raise ValueError('Authentication source must be an existing regular file')
    if auth.resolve().is_relative_to(w):
        raise ValueError('Authentication must remain in its original home')
    if w.exists() and any(w.iterdir()):
        raise ValueError('Use a fresh empty runtime; never reuse an uncertain attempt')
    w.mkdir(parents=True,mode=0o700,exist_ok=True)
    if w.stat().st_uid!=os.getuid():
        raise ValueError('Private runtime must belong to the current user')
    w.chmod(0o700)
    rt=Runtime(w,w/'codex-home',w/'tmp',w/'state',w/'logs',auth.resolve())
    for p in [rt.home,rt.tmp,rt.state,rt.logs]:
        p.mkdir(mode=0o700)
    # The sandbox forbids changes through this link or to the source. Never
    # serialize credentials, copy auth.json or allow automatic credential writes.
    (rt.home/'auth.json').symlink_to(rt.auth_source)
    return rt


def environment(rt,inherited=None):
    inherited=os.environ if inherited is None else inherited
    allowed={'PATH','HOME','USER','LOGNAME','SHELL','LANG','LC_ALL','LC_CTYPE','TZ',
             'SSL_CERT_FILE','SSL_CERT_DIR','REQUESTS_CA_BUNDLE','CURL_CA_BUNDLE',
             'HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','NO_PROXY'}
    env={k:v for k,v in inherited.items() if k in allowed}
    env.update(CODEX_HOME=str(rt.home),CODEX_SQLITE_HOME=str(rt.state),
               TMPDIR=str(rt.tmp),TMP=str(rt.tmp),TEMP=str(rt.tmp),
               RUST_LOG='error',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    return env


def profile(rt,executable,network=False):
    # Preserve the complete old repository/private/symlink/hardlink/write/process
    # policy. The external cwd removes ancestor canonicalization failures.
    p=original_profile(rt.workspace,[ROOT],network,executable)
    q=lambda x:json.dumps(str(x))
    home=rt.auth_source.parent
    # Permit metadata on the original home itself for path traversal, but no
    # listing or unrelated state/config/session reads. Only auth.json data.
    p+='\n'+f'(deny file-read* (require-all (subpath {q(home)}) (require-not (literal {q(home)})) (require-not (literal {q(rt.auth_source)}))))'
    p+='\n'+f'(deny file-read-data (literal {q(home)}))'
    p+='\n'+f'(deny file-write* (literal {q(rt.home/"auth.json")}))'
    return p


def overrides(rt):
    values={'sqlite_home':str(rt.state),'log_dir':str(rt.logs),
            'analytics.enabled':False,'feedback.enabled':False,
            'check_for_update_on_startup':False,'mcp_servers':{},
            'project_doc_max_bytes':0,'model_reasoning_effort':'low'}
    args=[]
    for key,value in values.items():
        args+=['-c',key+'='+('{}' if value=={} else json.dumps(value))]
    return args


def exec_argv(codex,rt,schema=None):
    args=[str(codex),'exec','--ignore-user-config','--ignore-rules','--ephemeral',
          '--skip-git-repo-check','--sandbox','workspace-write','--json','-C',str(rt.workspace)]
    args+=overrides(rt)
    for feature in DISABLED:
        args+=['--disable',feature]
    if schema is not None:
        args+=['--output-schema',str(schema)]
    return args+['-']


def app_server_argv(codex,rt):
    args=[str(codex),'app-server','--stdio']+overrides(rt)
    for feature in DISABLED:
        args+=['--disable',feature]
    return args


def run(command,rt,*,network=False,stdin='',timeout=15,cpu_seconds=10,cancel_check=None):
    """Bounded whole-process isolation. No network unless explicitly requested."""
    sandbox=shutil.which('sandbox-exec')
    if sandbox is None:
        raise RuntimeError('macOS sandbox missing; no fallback')
    executable=Path(command[0]).resolve()
    policy=rt.workspace/'sandbox.sb'
    policy.write_text(profile(rt,executable,network))
    def limits():
        cap=max(1,int(cpu_seconds))
        resource.setrlimit(resource.RLIMIT_CPU,(cap,cap))
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    process=subprocess.Popen([sandbox,'-f',str(policy),*map(str,command)],
        cwd=rt.workspace,env=environment(rt),stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
        start_new_session=True,preexec_fn=limits)
    first=True;timed_out=False;cancelled=False
    try:
        while True:
            remaining=timeout-(time.monotonic()-start)
            if remaining<=0 or cancel_check and cancel_check():
                timed_out=remaining<=0;cancelled=not timed_out
                os.killpg(process.pid,signal.SIGKILL)
                stdout,stderr=process.communicate();break
            try:
                stdout,stderr=process.communicate(stdin if first else None,timeout=min(.25,remaining));break
            except subprocess.TimeoutExpired:
                first=False
    finally:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(returncode=process.returncode,stdout=stdout,stderr=stderr,
        timed_out=timed_out,cancelled=cancelled,wall_seconds=time.monotonic()-start,
        cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        policy_sha=sha(policy),network_allowed=network)


def rpc_quota(rt,command=None,*,network=False,timeout=10):
    """Initialize plus one fixed read-only quota RPC. No turn methods."""
    command=app_server_argv(shutil.which('codex'),rt) if command is None else command
    sandbox=shutil.which('sandbox-exec')
    if sandbox is None:raise RuntimeError('macOS sandbox missing')
    policy=rt.workspace/'quota-sandbox.sb'
    policy.write_text(profile(rt,Path(command[0]).resolve(),network))
    def limits():resource.setrlimit(resource.RLIMIT_CPU,(10,10))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
    p=subprocess.Popen([sandbox,'-f',str(policy),*map(str,command)],cwd=rt.workspace,
        env=environment(rt),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
        bufsize=0,start_new_session=True,preexec_fn=limits)
    os.set_blocking(p.stdout.fileno(),False)
    selector=selectors.DefaultSelector();selector.register(p.stdout,selectors.EVENT_READ)
    buffer=bytearray();deadline=start+timeout
    def send(value):p.stdin.write((json.dumps(value)+'\n').encode());p.stdin.flush()
    def receive(id):
        while time.monotonic()<deadline:
            while b'\n' in buffer:
                line,_,rest=buffer.partition(b'\n');buffer[:]=rest
                if not line.strip():continue
                value=json.loads(line)
                if value.get('id')==id:return value
            if selector.select(max(0,deadline-time.monotonic())):
                chunk=os.read(p.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('Quota RPC channel closed')
                buffer.extend(chunk)
                if len(buffer)>1024*1024:raise ValueError('Quota framing bound exceeded')
        raise TimeoutError('Quota read timed out')
    try:
        send(dict(id=0,method='initialize',params=dict(clientInfo=dict(name='srn-backend-v3',version='1'),capabilities=dict(experimentalApi=False))))
        if 'error' in receive(0):raise RuntimeError('Quota initialization rejected')
        send(dict(method='initialized',params={}))
        send(dict(id=1,method='account/rateLimits/read',params=dict(supportsLunaReserve=False,excludeResetCreditDetails=True)))
        reply=receive(1)
        if 'error' in reply:raise RuntimeError('Quota read rejected')
        value=reply['result'];buckets=value.get('rateLimitsByLimitId') or {'default':value.get('rateLimits',{})}
        safe={k:{f:b.get(f) for f in ['primary','secondary','rateLimitReachedType','spendControlReached','individualLimit']} for k,b in buckets.items()}
        result=dict(available=True,ordinary_usage_allowed=value.get('ordinaryUsageAllowed'),buckets=safe)
    except Exception as ex:
        result=dict(available=False,ordinary_usage_allowed=None,reason=type(ex).__name__)
    finally:
        selector.close()
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=3)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    return result|dict(paid_fallback_requested=False,network_allowed=network,
        CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        wall_seconds=time.monotonic()-start,policy_sha=sha(policy))
