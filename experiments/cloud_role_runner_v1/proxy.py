"""Trusted CONNECT relay. Role containers have no external network route."""
import base64,json,os,resource,select,signal,socket,socketserver,ssl,sys,time
from pathlib import Path
from urllib.parse import urlsplit,unquote
sys.path.insert(0,'/runtime')
try:from contract import authority,save
except ImportError:from .contract import authority,save

def read_frame():
 def exact(n):
  b=b''
  while len(b)<n:
   q=os.read(0,n-len(b))
   if not q:raise EOFError('Missing private input frame')
   b+=q
  return b
 n=int.from_bytes(exact(4),'big')
 if not 0<n<=131072:raise ValueError('Input frame bound')
 return json.loads(exact(n))

def connect_upstream(host,port,config):
 upstream=config.get('upstream')
 if config.get('synthetic'):
  # No provider request: deterministic tunnel fixture inside this trusted process.
  a,b=socket.socketpair()
  import threading
  def echo():
   with b:
    while True:
     data=b.recv(4096)
     if not data:return
     b.sendall(data)
  threading.Thread(target=echo,daemon=True).start();return a
 if not upstream:return socket.create_connection((host,port),timeout=10)
 u=urlsplit(upstream)
 if u.scheme not in ['http','https'] or not u.hostname or u.path not in ['','/'] or u.query or u.fragment:raise ValueError('Unsupported managed proxy configuration')
 s=socket.create_connection((config.get('upstream_connect_host') or u.hostname,u.port or (443 if u.scheme=='https' else 80)),timeout=10)
 if u.scheme=='https':s=ssl.create_default_context(cafile='/runtime/ca.pem').wrap_socket(s,server_hostname=u.hostname)
 auth=''
 if u.username is not None:auth='Proxy-Authorization: Basic '+base64.b64encode((unquote(u.username)+':'+unquote(u.password or '')).encode()).decode()+'\r\n'
 s.sendall(f'CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n{auth}\r\n'.encode())
 head=b''
 while b'\r\n\r\n' not in head:
  q=s.recv(1)
  if not q:raise EOFError('Upstream CONNECT closed')
  head+=q
  if len(head)>8192:raise ValueError('Upstream response bound')
 if head.split(b'\r\n',1)[0].split()[1]!=b'200':s.close();raise RuntimeError('Managed proxy denied CONNECT')
 return s

class Handler(socketserver.BaseRequestHandler):
 def handle(self):
  self.request.settimeout(10);head=b''
  try:
   while b'\r\n\r\n' not in head:
    q=self.request.recv(1)
    if not q:return
    head+=q
    if len(head)>8192:raise ValueError('Header bound')
   method,target,version=head.split(b'\r\n',1)[0].decode('ascii').split()
   if method!='CONNECT' or version!='HTTP/1.1':raise ValueError('CONNECT required')
   host,port=authority(target)
  except (ValueError,UnicodeError):self.request.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n');return
  try:s=connect_upstream(host,port,self.server.configuration)
  except Exception:self.request.sendall(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n');return
  with s:
   self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n');deadline=time.monotonic()+310
   while time.monotonic()<deadline:
    ready,_,_=select.select([self.request,s],[],[],1)
    for r in ready:
     data=r.recv(65536)
     if not data:return
     (s if r is self.request else self.request).sendall(data)

class Server(socketserver.ThreadingTCPServer):
 allow_reuse_address=True
 daemon_threads=True

def main():
 config=read_frame();start=time.process_time()
 def stop(*_):raise SystemExit(0)
 signal.signal(signal.SIGTERM,stop)
 try:
  with Server(('0.0.0.0',3128),Handler) as s:
   s.configuration=config;print(json.dumps({'ready':True}),flush=True);s.serve_forever(poll_interval=.1)
 finally:save('/work/PROXY_USAGE.json',dict(CPU_seconds=time.process_time()))
if __name__=='__main__':main()
