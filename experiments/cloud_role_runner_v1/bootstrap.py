"""Trusted wrapper plus child-only inherited process filter; no Landlock dependency."""
import base64,ctypes,errno,io,json,os,platform,resource,selectors,signal,socket,subprocess,sys,threading,time
from pathlib import Path
sys.path.insert(0,'/runtime')
try:from contract import auth_bytes,codex_argv,redact,save
except ImportError:from .contract import auth_bytes,codex_argv,redact,save
RULES={'fork':57,'vfork':58,'clone':56,'clone3':435}

def restrict_processes():
 if platform.machine()!='x86_64':raise RuntimeError('Reviewed x86_64 filter required')
 class F(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]
 class P(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(F))]
 # Reject x32/foreign architecture; deny fork/vfork, return ENOSYS for clone3,
 # permit clone only with CLONE_THREAD. No process creation after CLI starts.
 v=[F(0x20,0,0,4),F(0x15,1,0,0xc000003e),F(0x06,0,0,0x80000000),F(0x20,0,0,0),F(0x45,0,1,0x40000000),F(0x06,0,0,0x80000000)]
 for n in [57,58]:v += [F(0x15,0,1,n),F(0x06,0,0,0x50000|errno.EPERM)]
 v += [F(0x15,0,1,435),F(0x06,0,0,0x50000|errno.ENOSYS),F(0x15,0,3,56),F(0x20,0,0,16),F(0x45,1,0,0x10000),F(0x06,0,0,0x50000|errno.EPERM),F(0x06,0,0,0x7fff0000)]
 a=(F*len(v))(*v);p=P(len(v),a);lib=ctypes.CDLL(None,use_errno=True)
 if lib.prctl(38,1,0,0,0) or lib.prctl(22,2,ctypes.byref(p),0,0):raise OSError(ctypes.get_errno(),'Process filter unavailable')

def read_frame():
 def exact(n):
  b=b''
  while len(b)<n:
   q=os.read(0,n-len(b))
   if not q:raise EOFError('Missing private input frame')
   b+=q
  return b
 n=int.from_bytes(exact(4),'big')
 if not 0<n<=131072:raise ValueError('Private frame bound')
 return json.loads(exact(n))

def connect(proxy,target):
 s=socket.create_connection((proxy,3128),timeout=3);s.sendall(('CONNECT '+target+' HTTP/1.1\r\nHost: '+target+'\r\n\r\n').encode());head=b''
 while b'\r\n\r\n' not in head:
  q=s.recv(1)
  if not q:raise EOFError('Relay closed')
  head+=q
 return s,head

def proof(proxy,synthetic):
 flags={};status=dict(x.split(':',1) for x in Path('/proc/self/status').read_text().splitlines() if ':' in x)
 flags['nonroot']=os.getuid()==os.getgid()==65534
 flags['no_capabilities']=all(int(status[k].strip(),16)==0 for k in ['CapInh','CapPrm','CapEff','CapBnd','CapAmb'])
 flags['no_new_privileges']=status['NoNewPrivs'].strip()=='1';flags['seccomp']=status['Seccomp'].strip()=='2'
 def denied(name,fn):
  try:fn()
  except (OSError,PermissionError):flags[name]=True
  else:flags[name]=False
 denied('root_write_denied',lambda:Path('/forbidden').write_text('bad'))
 denied('runner_write_denied',lambda:Path('/runtime/contract.py').write_text('bad'))
 for name,path in [('evaluator_unavailable','/protected/evaluator.py'),('confirmation_unavailable','/protected/confirmation.json'),('other_role_unavailable','/other-role/request.json'),('host_home_unavailable','/root/.codex/auth.json')]:denied(name,lambda p=path:Path(p).read_bytes())
 link=Path('/work/escape');link.symlink_to('/protected/confirmation.json');denied('symlink_escape_denied',link.read_bytes);link.unlink()
 denied('hardlink_escape_denied',lambda:os.link('/protected/confirmation.json','/work/hardlink'))
 denied('fork_denied',os.fork)
 denied('child_process_denied',lambda:subprocess.run(['/bin/true'],check=True))
 values=[];t=threading.Thread(target=lambda:values.append(True));t.start();t.join();flags['threads_work']=values==[True]
 def direct():
  with socket.create_connection(('1.1.1.1',443),timeout=.4):pass
 denied('direct_egress_denied',direct)
 s,head=connect(proxy,'evil.invalid:443');s.close();flags['unapproved_connect_denied']=head.startswith(b'HTTP/1.1 403')
 if synthetic:
  s,head=connect(proxy,'chatgpt.com:443');s.sendall(b'SYNTHETIC_ROLE');flags['approved_tunnel_works']=head.startswith(b'HTTP/1.1 200') and s.recv(128)==b'SYNTHETIC_ROLE';s.close()
 Path('/work/allowed').write_text('allowed');flags['workspace_write_allowed']=True
 flags['no_api_credentials']=not any(k in os.environ for k in ['OPENAI_API_KEY','CODEX_API_KEY'])
 save('/work/ISOLATION.json',dict(passed=all(flags.values()),checks=flags,synthetic=synthetic,parent_agent_isolation_certified=False))
 if not all(flags.values()):raise RuntimeError('Actual role isolation failed')


def child(mode,proxy,synthetic):
 proof(proxy,synthetic)
 if synthetic:
  print(json.dumps(dict(type='turn.started')))
  print(json.dumps(dict(type='item.completed',item=dict(type='agent_message',text='{"ok":true}'))))
  print(json.dumps(dict(type='turn.completed',usage=dict(input_tokens=100,output_tokens=20,cached_input_tokens=80))))
  if Path('/runtime/codex').is_file():
   sys.stdout.flush();os.execve('/runtime/codex',['/runtime/codex','exec','--help'],dict(os.environ))
  return
 os.execve('/runtime/codex',codex_argv(mode),dict(os.environ))


def main():
 if len(sys.argv)>1 and sys.argv[1]=='child':return child(sys.argv[2],sys.argv[3],sys.argv[4]=='fixture')
 frame=read_frame();mode=frame['mode'];synthetic=mode=='fixture';proxy=frame['proxy'];auth=base64.b64decode(frame.get('auth','')) if not synthetic else b'{}'
 if not synthetic:auth_bytes(auth)
 os.environ.clear();os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin',HOME='/run/private',CODEX_HOME='/run/private/codex',CODEX_SQLITE_HOME='/tmp/state',TMPDIR='/tmp',LANG='C.UTF-8',HTTP_PROXY='http://'+proxy+':3128',HTTPS_PROXY='http://'+proxy+':3128',NO_PROXY=frame.get('no_proxy',''),SSL_CERT_FILE='/runtime/ca.pem',REQUESTS_CA_BUNDLE='/runtime/ca.pem',RUST_LOG='error')
 home=Path(os.environ['CODEX_HOME']);home.mkdir(mode=0o700)
 if not synthetic:
  p=home/'auth.json';p.write_bytes(auth);p.chmod(0o400);home.chmod(0o500)
 start=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN);cap=frame['cpu_seconds']
 def limits():resource.setrlimit(resource.RLIMIT_CPU,(cap,cap));restrict_processes()
 argv=[sys.executable,'-I','-B','/runtime/bootstrap.py','child',mode,proxy,'fixture' if synthetic else 'live']
 p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,preexec_fn=limits)
 # Frame is consumed with unbuffered os.read: all remaining stdin belongs to CLI.
 if mode=='quota':
  # Trusted RPC bridge: only initialize and one account/rateLimits/read; never turn/start.
  rpc=[dict(id=0,method='initialize',params=dict(clientInfo=dict(name='srn-linux-role',version='1'),capabilities=dict(experimentalApi=False))),dict(method='initialized',params={}),dict(id=1,method='account/rateLimits/read',params=dict(supportsLunaReserve=False,excludeResetCreditDetails=True))]
  # Kept interactive below: initialize first, then send initialized/read after acknowledgement.
  payload=(json.dumps(rpc[0])+'\n').encode()
 else:payload=sys.stdin.buffer.read(65537)
 if len(payload)>65536:raise ValueError('Prompt bound')
 p.stdin.write(payload);p.stdin.flush()
 if mode!='quota':p.stdin.close()
 sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ);sel.register(p.stderr,selectors.EVENT_READ);out=bytearray();err=bytearray();timed=False;rpc_sent=False;quota=None
 try:
  while sel.get_map():
   if time.monotonic()-start>frame['wall_seconds']:timed=True;break
   for key,_ in sel.select(.1):
    b=os.read(key.fd,65536)
    if not b:sel.unregister(key.fileobj);continue
    target=out if key.fileobj is p.stdout else err;target.extend(b)
    if len(out)>4194304 or len(err)>1048576:raise ValueError('Event stream bound')
   if mode=='quota':
    for line in out.splitlines():
     try:r=json.loads(line)
     except ValueError:continue
     if r.get('id')==0 and not rpc_sent:
      if 'error' in r:raise ValueError('Quota initialize rejected')
      p.stdin.write(('\n'.join(json.dumps(q) for q in rpc[1:])+'\n').encode());p.stdin.flush();rpc_sent=True
     if r.get('id')==1:
      if 'error' in r:raise ValueError('Quota read rejected')
      quota=r['result'];break
    if quota is not None:break
  if mode=='quota' or timed:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  p.wait(timeout=3)
 finally:
  sel.close()
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait(timeout=3)
 after=resource.getrusage(resource.RUSAGE_CHILDREN);r=dict(returncode=0 if quota is not None else p.returncode,timed_out=timed,CPU_seconds=time.process_time()+after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,wall_seconds=time.monotonic()-start,synthetic=synthetic)
 if quota is not None:
  buckets=quota.get('rateLimitsByLimitId') or {'default':quota.get('rateLimits',{})}
  r['quota']=dict(available=True,ordinary_usage_allowed=quota.get('ordinaryUsageAllowed'),buckets={k:{f:b.get(f) for f in ['primary','secondary','rateLimitReachedType','spendControlReached','individualLimit']} for k,b in buckets.items()})
 save('/work/PROCESS.json',r)
 (Path('/work')/'events.jsonl').write_text(redact(out.decode(errors='replace'),auth));(Path('/work')/'stderr.log').write_text(redact(err.decode(errors='replace'),auth))
 print(json.dumps(r))
 if r['returncode'] or timed:raise SystemExit(3)
if __name__=='__main__':main()
