"""Fresh internal network, trusted CONNECT relay and nonroot per-role container."""
import base64,fcntl,json,os,re,selectors,shutil,signal,socket,subprocess,time,uuid
from pathlib import Path
from .contract import ROOT,sha,save,redact
SOCKETS=['/var/run/docker.sock','/run/codex-docker/docker.sock']
IMAGE=re.compile(r'sha256:[a-f0-9]{64}')

def frame(value):
 b=json.dumps(value,separators=(',',':')).encode()
 if len(b)>131072:raise ValueError('Private input frame too large')
 return len(b).to_bytes(4,'big')+b

def validate_mode(runtime,mode):
 if runtime.get('synthetic') and mode!='fixture':raise ValueError('Synthetic runtime cannot make live calls')

def command(docker,image,job,kind,name,network,proxy_ip=None):
 if kind not in ['role','proxy'] or not IMAGE.fullmatch(image):raise ValueError('Immutable reviewed role runtime required')
 job=Path(job)
 if ',' in str(job):raise ValueError('Unsupported mount path')
 argv=[*docker,'create','--name',name,'--pull=never','--interactive','--read-only','--cap-drop=ALL','--user=65534:65534','--security-opt=no-new-privileges:true','--pids-limit=32','--memory=2g','--memory-swap=2g','--cpus=1','--ulimit=cpu='+('60:60' if kind=='role' else '30:30'),'--ipc=none','--network='+network,'--tmpfs=/tmp:rw,noexec,nosuid,size=64m,mode=1777','--tmpfs=/run/private:rw,noexec,nosuid,size=4m,mode=0700,uid=65534,gid=65534','--workdir=/work']
 if kind=='role':argv+=['--dns=127.0.0.1']
 for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy']:argv+=['--env',key+'=']
 for src,dst,ro in [(job/'source','/runtime',True),(job/'request','/request',True),(job/('output' if kind=='role' else 'relay'),'/work',False)]:
  argv+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
 for name in ['requirements.toml','managed_config.toml']:
  if (job/'source'/name).is_file():argv+=['--mount',f'type=bind,src={job / "source" / name},dst=/etc/codex/{name},readonly']
 return argv+[image,'python','-I','-B','/runtime/'+('bootstrap.py' if kind=='role' else 'proxy.py')]

def verify(configuration,argv,kind,image,network,job):
 c,h=configuration['Config'],configuration['HostConfig']
 if configuration['Image']!=image or c['Image']!=image or c['User']!='65534:65534' or c.get('Entrypoint') or c['Cmd']!=argv[-4:]:raise ValueError('CLI command/image/user changed')
 wanted=dict(ReadonlyRootfs=True,NetworkMode=network,Privileged=False,PidsLimit=32,Memory=2147483648,MemorySwap=2147483648,NanoCpus=1000000000,PidMode='',IpcMode='none',SecurityOpt=['no-new-privileges:true'],Ulimits=[dict(Name='cpu',Soft=60 if kind=='role' else 30,Hard=60 if kind=='role' else 30)])
 if any(h.get(k)!=v for k,v in wanted.items()) or set(h.get('CapDrop') or [])!={'ALL'} or h.get('CapAdd'):raise ValueError('Container restrictions changed')
 expected_tmp={'/tmp':'rw,noexec,nosuid,size=64m,mode=1777','/run/private':'rw,noexec,nosuid,size=4m,mode=0700,uid=65534,gid=65534'}
 if h.get('Tmpfs')!=expected_tmp or (kind=='role' and h.get('Dns')!=['127.0.0.1']):raise ValueError('Temporary filesystem or DNS policy changed')
 expected={'/runtime':(str(job/'source'),False),'/request':(str(job/'request'),False),'/work':(str(job/('output' if kind=='role' else 'relay')),True)}
 mounts=[m for m in configuration['Mounts'] if m['Type']!='tmpfs']
 for name in ['requirements.toml','managed_config.toml']:
  if (job/'source'/name).is_file():expected['/etc/codex/'+name]=(str(job/'source'/name),False)
 if len(mounts)!=len(expected) or {m['Destination'] for m in mounts}!=set(expected):raise ValueError('Unexpected role mount')
 for m in mounts:
  if m['Type']!='bind' or (m['Source'],m['RW'])!=expected[m['Destination']]:raise ValueError('Mount permission changed')
 if kind=='role' and set(configuration['NetworkSettings']['Networks'])!={network}:raise ValueError('Role has extra network')


def launch(runtime,out,mode,schema,auth=b'{}',prompt='',timeout=300,cancel_check=None):
 out=Path(out).absolute()
 if out.exists() or out.is_symlink() or any(p.is_symlink() for p in out.parents):raise ValueError('New canonical attempt path required; no retry')
 validate_mode(runtime,mode)
 for name,h in runtime['source_hashes'].items():
  if sha(Path(runtime['source'])/name)!=h:raise ValueError('Runtime source/binary changed')
 if runtime['socket'] not in SOCKETS or not IMAGE.fullmatch(runtime['image']):raise ValueError('Wrong daemon/image')
 if runtime.get('proxy_environment_sha')!=proxy_environment_sha():raise ValueError('Managed proxy environment changed; prepare a new runtime for review')
 out.mkdir(parents=True,mode=0o700);out=out.resolve();job=out/'stage';job.mkdir(mode=0o755)
 shutil.copytree(runtime['source'],job/'source');(job/'request').mkdir();save(job/'request/schema.json',schema)
 for p in [job,job/'source',job/'request']:p.chmod(0o755)
 for name in ['output','relay']:(job/name).mkdir(mode=0o777);(job/name).chmod(0o777)
 docker=[shutil.which('docker') or 'docker','--host=unix://'+runtime['socket']];env={k:os.environ[k] for k in ['PATH','HOME','LANG','DOCKER_CONFIG'] if k in os.environ}
 tag='srn-role-'+uuid.uuid4().hex;net=tag+'-net';role=tag+'-agent';relay=tag+'-relay';calls=[];relay_process=None;start=time.monotonic();receipt=dict(passed=False,mode=mode,parent_agent_isolation_certified=False,live_usage_verified=False)
 marker=Path(runtime['directory'])/'UNRESOLVED_CLEANUP.json'
 if marker.exists():raise RuntimeError('Prior role resources require cleanup adjudication')
 lock=(Path(runtime['directory'])/'worker.lock').open('a+')
 def call(args,cap=60,check=True):
  try:r=subprocess.run(args,env=env,capture_output=True,timeout=cap,close_fds=True)
  except subprocess.TimeoutExpired:calls.append(dict(command=args,timeout=cap));raise
  # Never persist inspect environments, stdin, authentication or proxy URLs.
  calls.append(dict(command=args,returncode=r.returncode))
  if check and r.returncode:raise RuntimeError('Docker lifecycle command failed: '+args[len(docker)])
  return r
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  call([*docker,'network','create','--internal','--driver','bridge',net])
  nc=json.loads(call([*docker,'network','inspect',net]).stdout)[0]
  if nc['Internal'] is not True or nc['Driver']!='bridge':raise ValueError('Internal network not enforced')
  pc=command(docker,runtime['image'],job,'proxy',relay,'bridge');call(pc)
  relay_process=subprocess.Popen([*docker,'start','--attach','--interactive',relay],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  proxy_url=upstream();address=None
  if proxy_url and mode!='fixture':
   from urllib.parse import urlsplit
   u=urlsplit(proxy_url);address=socket.getaddrinfo(u.hostname,u.port or (443 if u.scheme=='https' else 80),socket.AF_INET,socket.SOCK_STREAM)[0][4][0]
  relay_process.stdin.write(frame(dict(synthetic=mode=='fixture',upstream=proxy_url,upstream_connect_host=address)));relay_process.stdin.flush()
  selector=selectors.DefaultSelector();selector.register(relay_process.stdout,selectors.EVENT_READ)
  if not selector.select(10):raise TimeoutError('Restricted relay did not start')
  ready=json.loads(relay_process.stdout.readline());selector.close()
  if ready.get('ready') is not True:raise RuntimeError('Restricted relay startup failed')
  call([*docker,'network','connect',net,relay])
  conf=json.loads(call([*docker,'inspect',relay]).stdout)[0];verify(conf,pc,'proxy',runtime['image'],'bridge',job)
  if set(conf['NetworkSettings']['Networks'])!={'bridge',net}:raise ValueError('Unexpected relay network')
  ip=conf['NetworkSettings']['Networks'][net]['IPAddress']
  if not re.fullmatch(r'\d+\.\d+\.\d+\.\d+',ip):raise ValueError('Internal relay address absent')
  rc=command(docker,runtime['image'],job,'role',role,net,ip);call(rc)
  conf=json.loads(call([*docker,'inspect',role]).stdout)[0];verify(conf,rc,'role',runtime['image'],net,job)
  if timeout-(time.monotonic()-start)<=1 or cancel_check and cancel_check():raise TimeoutError('Call deadline/cancellation reached during setup; no model dispatched')
  value=dict(mode=mode,proxy=ip,auth=base64.b64encode(auth).decode() if mode!='fixture' else '',no_proxy=os.environ.get('NO_PROXY',os.environ.get('no_proxy','')),cpu_seconds=60 if mode=='exec' else 10,wall_seconds=max(1,timeout-(time.monotonic()-start)))
  proc=subprocess.Popen([*docker,'start','--attach','--interactive',role],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
  timed=False;cancelled=False;first=True
  try:
   while True:
    left=timeout+5-(time.monotonic()-start)
    if left<=0 or cancel_check and cancel_check():timed=left<=0;cancelled=not timed;break
    try:stdout,stderr=proc.communicate(frame(value)+prompt.encode() if first else None,timeout=min(.5,left));break
    except subprocess.TimeoutExpired:first=False
   if timed or cancelled:call([*docker,'kill',role],check=False);proc.kill();stdout,stderr=proc.communicate()
  finally:
   if proc.poll() is None:proc.kill();proc.wait(timeout=3)
  receipt.update(returncode=proc.returncode,timed_out=timed,cancelled=cancelled,effective_configuration_verified=True,image=runtime['image'])
  process=json.loads((job/'output/PROCESS.json').read_text());proof=json.loads((job/'output/ISOLATION.json').read_text())
  if process['returncode'] or process['timed_out'] or proof['passed'] is not True:raise RuntimeError('CLI, resource guard or actual role restrictions failed')
  receipt.update(passed=not timed and not cancelled,CPU_seconds=process['CPU_seconds'],process=process,isolation=proof,stdout=redact((job/'output/events.jsonl').read_text(),auth),stderr=redact((job/'output/stderr.log').read_text(),auth))
 except Exception as e:receipt.update(error_type=type(e).__name__,reason=str(e),returncode=receipt.get('returncode',1))
 finally:
  try:call([*docker,'stop','--time','2',relay],cap=10,check=False)
  except Exception:pass
  if relay_process:
   try:relay_process.communicate(timeout=5)
   except subprocess.TimeoutExpired:relay_process.kill();relay_process.wait(timeout=3)
  removed=[]
  for name in [role,relay]:
   try:
    r=call([*docker,'rm','--force',name],check=False);removed.append(r.returncode==0 or b'No such container' in r.stderr)
   except Exception:removed.append(False)
  try:
   r=call([*docker,'network','rm',net],check=False);removed.append(r.returncode==0 or b'not found' in r.stderr)
  except Exception:removed.append(False)
  receipt['resources_removed']=all(removed)
  if not all(removed):save(marker,dict(network=net,containers=[role,relay]));receipt['passed']=False
  if (job/'relay/PROXY_USAGE.json').exists():receipt['relay_CPU_seconds']=json.loads((job/'relay/PROXY_USAGE.json').read_text())['CPU_seconds']
  elif receipt['passed']:receipt.update(passed=False,reason='Relay CPU receipt missing')
  receipt['wall_seconds']=time.monotonic()-start
  save(out/'RECEIPT.json',{k:v for k,v in receipt.items() if k not in ['stdout','stderr']});save(out/'DOCKER_COMMANDS.json',calls);lock.close()
 return receipt

def upstream():
 from urllib.parse import urlsplit
 value=next((os.environ.get(k) for k in ['HTTPS_PROXY','https_proxy','HTTP_PROXY','http_proxy','ALL_PROXY','all_proxy'] if os.environ.get(k)),None)
 if value and urlsplit(value).scheme not in ['http','https']:raise ValueError('Unsupported managed proxy; no direct fallback')
 return value
def proxy_environment_sha():
 import hashlib
 return hashlib.sha256(json.dumps({k:os.environ.get(k) for k in ['HTTPS_PROXY','https_proxy','HTTP_PROXY','http_proxy','ALL_PROXY','all_proxy','NO_PROXY','no_proxy']},sort_keys=True).encode()).hexdigest()
