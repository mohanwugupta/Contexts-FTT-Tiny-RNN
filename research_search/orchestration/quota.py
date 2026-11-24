"""Read-only installed Codex app-server quota RPC. Never redeems credits."""
import json,os,selectors,signal,subprocess,time

def read_quota(codex,env,timeout=15):
    p=subprocess.Popen([codex,'app-server','--stdio','-c','mcp_servers={}','--disable','hooks','--disable','plugins','--disable','apps','--disable','multi_agent','--disable','multi_agent_v2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,bufsize=0,start_new_session=True,env=env)
    os.set_blocking(p.stdout.fileno(),False)
    selector=selectors.DefaultSelector();selector.register(p.stdout,selectors.EVENT_READ);deadline=time.monotonic()+timeout
    buffer=bytearray()
    def send(value):p.stdin.write((json.dumps(value)+'\n').encode());p.stdin.flush()
    def receive(id):
        while time.monotonic()<deadline:
            while b'\n' in buffer:
                line,_,rest=buffer.partition(b'\n');buffer[:]=rest
                if not line.strip():continue
                value=json.loads(line)
                if value.get('id')==id:return value
            ready=selector.select(max(0,deadline-time.monotonic()))
            if ready:
                chunk=os.read(p.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('Quota channel closed')
                buffer.extend(chunk)
                if len(buffer)>4*1024*1024:raise ValueError('Quota response exceeds bounded framing buffer')
        raise TimeoutError('Quota read timed out')
    try:
        send(dict(id=0,method='initialize',params=dict(clientInfo=dict(name='cognitive-search-preflight',version='1'),capabilities=dict(experimentalApi=False))))
        init=receive(0)
        if 'error' in init:raise RuntimeError(str(init['error']))
        send(dict(method='initialized',params={}))
        # Explicitly decline reserve-credit fallback capability. No reset,
        # consume, purchase or paid-overage method is invoked.
        send(dict(id=1,method='account/rateLimits/read',params=dict(supportsLunaReserve=False,excludeResetCreditDetails=True)))
        value=receive(1)
        if 'error' in value:raise RuntimeError(str(value['error']))
        r=value['result'];buckets=r.get('rateLimitsByLimitId') or {'default':r.get('rateLimits',{})};safe={}
        for key,b in buckets.items():safe[key]={k:b.get(k) for k in ['primary','secondary','rateLimitReachedType','spendControlReached','individualLimit']}
        return dict(available=True,ordinary_usage_allowed=r.get('ordinaryUsageAllowed'),buckets=safe,observed_at=time.time(),provider='Codex account/rateLimits/read',paid_fallback_requested=False)
    except Exception as ex:return dict(available=False,ordinary_usage_allowed=None,reason=str(ex),paid_fallback_requested=False)
    finally:
        selector.close()
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=5)
