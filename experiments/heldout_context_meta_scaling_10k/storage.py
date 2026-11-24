"""One durable transaction per episode: summary and exact restart state together."""
import hashlib,io,os,sqlite3
from pathlib import Path
import torch

def encode(value):
 b=io.BytesIO();torch.save(value,b);return b.getvalue()
def decode(blob,digest):
 if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Restart state checksum mismatch')
 return torch.load(io.BytesIO(blob),weights_only=False)
def atomic_save(path,value):
 path=Path(path);blob=encode(value);digest=hashlib.sha256(blob).hexdigest()
 if path.exists():
  if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Refusing changed checkpoint '+str(path))
  return digest
 temp=path.with_name(path.name+'.pending')
 with temp.open('wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
 return digest

class Journal:
 def __init__(self,path):
  self.db=sqlite3.connect(path,timeout=60)
  self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
  self.db.execute('CREATE TABLE IF NOT EXISTS latest(id INTEGER PRIMARY KEY CHECK(id=1), episode INTEGER, data BLOB, hash TEXT)')
  self.db.execute('CREATE TABLE IF NOT EXISTS episodes(episode INTEGER PRIMARY KEY,data BLOB,hash TEXT)')
  self.db.execute('CREATE TABLE IF NOT EXISTS checkpoints(episode INTEGER PRIMARY KEY,data BLOB,hash TEXT)');self.db.commit()
 def initialize(self,value):
  if self.db.execute('SELECT COUNT(*) FROM latest').fetchone()[0]:return
  b=encode(value);h=hashlib.sha256(b).hexdigest()
  with self.db:self.db.execute('INSERT INTO latest VALUES(1,?,?,?)',(value['episode'],b,h))
 def latest(self):
  r=self.db.execute('SELECT episode,data,hash FROM latest').fetchone()
  if r is None:return None
  x=decode(r[1],r[2]);assert x['episode']==r[0];return x
 def checkpoint(self,episode):
  r=self.db.execute('SELECT data,hash FROM checkpoints WHERE episode=?',(episode,)).fetchone()
  if r is None:raise ValueError('Missing registered checkpoint')
  return decode(*r)
 def commit(self,value,summary,registered=False):
  e=value['episode'];previous=self.db.execute('SELECT episode FROM latest').fetchone()[0]
  if e!=previous+1 or summary['episode']!=e:raise ValueError('Would repeat or skip an episode')
  blob=encode(value);digest=hashlib.sha256(blob).hexdigest();s=encode(summary);h=hashlib.sha256(s).hexdigest()
  with self.db:
   self.db.execute('INSERT INTO episodes VALUES(?,?,?)',(e,s,h));self.db.execute('UPDATE latest SET episode=?,data=?,hash=? WHERE id=1',(e,blob,digest))
   if registered:self.db.execute('INSERT INTO checkpoints VALUES(?,?,?)',(e,blob,digest))
 def summaries(self):
  return [decode(b,h) for b,h in self.db.execute('SELECT data,hash FROM episodes ORDER BY episode')]
 def close(self):
  self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.db.close()
